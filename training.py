import datetime
import os
import signal

import tensorflow as tf
import numpy as np

from constants import *

from preprocessing import convert_audio_array_to_spec_array, load_audio_array, split_equal_size, load_spec_array
from losses import *
from architecture import build, load, extract_image, assemble_image, get_networks, get_networks2
from testing_network import save_end, use_generator


# TODO: Sample laenge (Eingabe) vergroessern (5 sekunden)

def handler(signum, frame):
    print("Aborting process, ctrl-C")
    logfile.flush()
    logfile.close()


signal.signal(signal.SIGINT, handler)


""" zufaellige ausschnitte der groesse 3 * shape ausstanzen 
    das ausstanzen geshieht um auf die richtige Groesse fuer
    den Inputlayer (groesse shape=24 ) zu kommen, 
    hier wird auf 3 * shape gecroppt, weil fuer das training
    nochmals in 3 Einzelspektrogramme gesplittet und wieder
    zusammengesetzt wird
"""
@tf.function
def proc(x):
    return tf.image.random_crop(x, size=[GL_HOP, 3 * GL_SHAPE, 1])


""" Set learning rate """


def update_lr(lr):
    opt_gen.learning_rate = lr
    opt_disc.learning_rate = lr


""" ########## Training Functions ################# """

""" Train Generator, Siamese and Critic """
""" Ein Trainingsdurchlauf für alle 3 Netze """


@tf.function
def train_all(orig, remix, validation):
    # splitting spectrogram in 3 parts
    orig1, orig2, orig3 = extract_image(orig)
    remix1, remix2, remix3 = extract_image(remix)
    valid1, valid2, valid3 = extract_image(validation)

    with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:
        # translating A to B
        gen_orig1 = gen(orig1, training=True)
        gen_orig2 = gen(orig2, training=True)
        gen_orig3 = gen(orig3, training=True)
        # identity mapping B to B                                                        COMMENT THESE 3 LINES IF THE IDENTITY LOSS TERM IS NOT NEEDED
        gen_remix1 = gen(remix1, training=True)
        gen_remix2 = gen(remix2, training=True)
        gen_remix3 = gen(remix3, training=True)
        # concatenate/assemble converted spectrograms
        gen_orig = assemble_image([gen_orig1, gen_orig2, gen_orig3])

        # feed concatenated spectrograms to critic
        critic_gen_orig = critic(gen_orig, training=True)
        critic_remix = critic(remix, training=True)
        # feed 2 pairs (A,G(A)) extracted spectrograms to Siamese
        siam_gen_orig1 = siam(gen_orig1, training=True)
        siam_gen_orig3 = siam(gen_orig3, training=True)
        siam_orig1 = siam(orig1, training=True)
        siam_orig3 = siam(orig3, training=True)

        gen_valid1 = gen(valid1, training=False)
        gen_valid2 = gen(valid2, training=False)
        gen_valid3 = gen(valid3, training=False)
        gen_valid = assemble_image([gen_valid1, gen_valid2, gen_valid3])
        siam_gen_valid1 = siam(gen_valid1, training=False)
        siam_gen_valid3 = siam(gen_valid3, training=False)
        siam_valid1 = siam(valid1, training=False)
        siam_valid3 = siam(valid3, training=False)

        critic_gen_valid = critic(gen_valid, training=False)

        # Losses
        # identity mapping loss
        # loss_id = (mae(remix1,gen_remix1)+mae(remix2,gen_remix2)+mae(remix3,gen_remix3))/3.                         #loss_id = 0. IF THE IDENTITY LOSS TERM IS NOT NEEDED
        loss_id = 0
        # travel loss
        loss_m = loss_travel(siam_orig1, siam_gen_orig1, siam_orig3, siam_gen_orig3) + loss_siamese(siam_orig1, siam_orig3)

        # generator and critic losses
        loss_g = L_g_adv(critic_gen_orig)
        loss_dr = loss_d_target(critic_remix)
        loss_df = loss_d_g_src(critic_gen_orig)
        loss_d = (loss_dr + loss_df) / 2.
        # generator+siamese total loss
        lossgtot = loss_g + 10. * loss_m + 0.5 * loss_id  # CHANGE LOSS WEIGHTS HERE  (COMMENT OUT +w*loss_id IF THE IDENTITY LOSS TERM IS NOT NEEDED)

        loss_g_valid = L_g_adv(critic_gen_valid) + 10. * loss_travel(siam_valid1, siam_gen_valid1, siam_valid3, siam_gen_valid3) + 0.5 * loss_siamese(siam_valid1, siam_valid3)

    # computing and applying gradients
    grad_gen = tape_gen.gradient(lossgtot, gen.trainable_variables + siam.trainable_variables)
    opt_gen.apply_gradients(zip(grad_gen, gen.trainable_variables + siam.trainable_variables))

    grad_disc = tape_disc.gradient(loss_d, critic.trainable_variables)
    opt_disc.apply_gradients(zip(grad_disc, critic.trainable_variables))

    return loss_dr, loss_df, loss_g, loss_id, lossgtot, loss_g_valid

@tf.function
def train_all_new(orig, remix, validation, validation_remix, alpha: float = 0, beta: float = 10, gamma: float = 10, delta: float = 3):
    orig1, orig2, orig3 = extract_image(orig)
    remix1, remix2, remix3 = extract_image(remix)
    valid1, valid2, valid3 = extract_image(validation)
    valr1, valr2, valr3 = extract_image(validation_remix)

    with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:# tf.GradientTape as tape_siam:
        # translating A to B
        gen_orig1 = gen(orig1, training=True)
        gen_orig2 = gen(orig2, training=True)
        gen_orig3 = gen(orig3, training=True)
        # identity mapping B to B                                                        COMMENT THESE 3 LINES IF THE IDENTITY LOSS TERM IS NOT NEEDED
        gen_remix1 = gen(remix1, training=True)
        gen_remix2 = gen(remix2, training=True)
        gen_remix3 = gen(remix3, training=True)
        # concatenate/assemble converted spectrograms
        gen_orig = assemble_image([gen_orig1, gen_orig2, gen_orig3])
        gen_remix = assemble_image([gen_remix1, gen_remix2, gen_remix3])

        # feed concatenated spectrograms to critic
        critic_gen_orig = critic(gen_orig, training=True)
        critic_remix = critic(remix, training=True)
        # feed 2 pairs (A,G(A)) extracted spectrograms to Siamese
        siam_gen_orig1 = siam(gen_orig1, training=True)
        siam_gen_orig3 = siam(gen_orig3, training=True)
        siam_orig1 = siam(orig1, training=True)
        siam_orig3 = siam(orig3, training=True)

        gen_valid1 = gen(valid1, training=False)
        gen_valid2 = gen(valid2, training=False)
        gen_valid3 = gen(valid3, training=False)
        gen_valid = assemble_image([gen_valid1, gen_valid2, gen_valid3])

        gen_valr1 = gen(valr1, training=False)
        gen_valr2 = gen(valr2, training=False)
        gen_valr3 = gen(valr3, training=False)
        gen_valr = assemble_image([gen_valr1, gen_valr2, gen_valr3])

        siam_gen_valid1 = siam(gen_valid1, training=False)
        siam_gen_valid3 = siam(gen_valid3, training=False)
        siam_valid1 = siam(valid1, training=False)
        siam_valid3 = siam(valid3, training=False)

        critic_gen_valid = critic(gen_valid, training=False)


        loss_d = L_d(critic_remix, critic_gen_orig)

        loss_s_margin = L_s_margin(delta, orig1, orig3, siam_orig1, siam_orig3)
        l_travel = L_travel(siam_orig1, siam_orig3, siam_gen_orig1, siam_gen_orig3)
        #lossgtot = L_g(0.5, 10, critic_gen_orig, remix, gen_remix, l_travel)
        loss_g_train = L_g_noID(beta, critic_gen_orig, l_travel)
        loss_s = L_s(beta, gamma, l_travel, loss_s_margin)

        l_travel_val = L_travel(siam_valid1, siam_valid3, siam_gen_valid1, siam_gen_valid3)
        loss_g_valid = L_g_noID(beta, critic_gen_valid, l_travel_val)
        #loss_g_valid = L_g_full(0.5, 10, critic_gen_valid, validation_remix, gen_valr, gen_valid1, gen_valid2, siam_gen_valid1, siam_gen_valid3)

    grad_gen = tape_gen.gradient(loss_g_train, gen.trainable_variables + siam.trainable_variables)
    opt_gen.apply_gradients(zip(grad_gen, gen.trainable_variables + siam.trainable_variables))

    grad_disc = tape_disc.gradient(loss_d, critic.trainable_variables)
    opt_disc.apply_gradients(zip(grad_disc, critic.trainable_variables))

    #grad_siam = tape_siam.gradient(loss_s, siam.trainable_variables)
    #opt_siam.apply_gradients(zip(grad_siam, siam.trainable_variables))

    return loss_d, loss_g_train, loss_g_valid, loss_s

""" Exklusives Diskriminator Training """


@tf.function
def train_d(a, b):
    aa, aa2, aa3 = extract_image(a)
    with tf.GradientTape() as tape_disc:
        fab = gen(aa, training=True)
        fab2 = gen(aa2, training=True)
        fab3 = gen(aa3, training=True)
        fabtot = assemble_image([fab, fab2, fab3])

        cab = critic(fabtot, training=True)
        cb = critic(b, training=True)

        loss_dr = loss_d_target(cb)
        loss_df = loss_d_g_src(cab)

        loss_d = (loss_dr + loss_df) / 2.

    grad_disc = tape_disc.gradient(loss_d, critic.trainable_variables)
    opt_disc.apply_gradients(zip(grad_disc, critic.trainable_variables))

    return loss_dr, loss_df


@tf.function
def train_d_new(orig, remix):
    orig1, orig2, orig3 = extract_image(orig)
    with tf.GradientTape() as tape_disc:
        g_orig1 = gen(orig1, training=True)
        g_orig2 = gen(orig2, training=True)
        g_orig3 = gen(orig3, training=True)
        g_orig = assemble_image([g_orig1, g_orig2, g_orig3])

        d_g_orig = critic(g_orig, training=True)
        d_remix = critic(remix, training=True)

        loss_d = L_d(d_remix, d_g_orig)
        #loss_d = (loss_dr + loss_df) / 2.

    grad_disc = tape_disc.gradient(loss_d, critic.trainable_variables)
    opt_disc.apply_gradients(zip(grad_disc, critic.trainable_variables))

    return loss_d

""" Trainings Schleife zum Laufenlassen des Trainings """


def train(epochs, aspec, dstrain_o, dstrain_r, dsval_o, batch_size=16, lr=0.0001, n_save=6, gupt=5, startep=0):
    # open logfile
    file = open(f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}_losses.txt", "w")

    # set lr
    update_lr(lr)
    df_list = []
    dr_list = []
    g_list = []
    id_list = []
    val_list = []
    c = 0
    g = 0

    # epochs
    for epoch in range(startep, epochs):
        bef = time.time()
        # batches
        for batchi, (a, b) in enumerate(zip(dstrain_o, dstrain_r)):

            # get validation iterator
            validation = dsval_o.as_numpy_iterator().next()

            # cond generator training
            if (batchi % gupt) == 0:
                dloss_t, dloss_f, _, idloss, gloss, valid_loss = train_all(a, b, validation)
            else:
                dloss_t, dloss_f = train_d(a, b)

            df_list.append(dloss_f)
            dr_list.append(dloss_t)
            g_list.append(gloss)
            id_list.append(idloss)
            val_list.append(valid_loss)
            c += 1
            g += 1

            if batchi % 250 == 0:
                log(time.strftime("%H:%M:%S ", time.localtime()), end='')
                log(f'[Epoch {epoch}/{epochs}] [Batch {batchi}] [D loss f: {np.mean(df_list[-g:], axis=0)} ', end='')
                log(f'r: {np.mean(dr_list[-g:], axis=0)}] ', end='')
                log(f'[G loss: {np.mean(g_list[-g:], axis=0)}] ', end='')
                log(f'[ID loss: {np.mean(id_list[-g:])}] ', end='')
                log(f'[LR: {lr}]')
                g = 0

            nbatch = batchi
        # end for batch

        log(f'Time for epoch {epoch}: {int(time.time() - bef)}')
        log(f'Time/Batch {(time.time() - bef) / nbatch}')

        save_end(epoch, np.mean(g_list[-n_save * c:], axis=0), np.mean(df_list[-n_save * c:], axis=0),
                 np.mean(id_list[-n_save * c:], axis=0), gen, critic, siam, aspec, n_save=n_save, save_path=gl_savepath)
        log(f'Mean D loss: {np.mean(df_list[-c:], axis=0)} Mean G loss: {np.mean(g_list[-c:], axis=0)} Mean ID loss: {np.mean(id_list[-c:], axis=0)}')
        file.write(
            f'{epoch},{np.mean(df_list[-c:], axis=0)},{np.mean(dr_list[-g:], axis=0)},{np.mean(g_list[-c:], axis=0)},{np.mean(id_list[-c:], axis=0)},{lr},{np.mean(val_list[-c:])}\n')
        c = 0
        logfile.flush()
        file.flush()

    # end for epochs

    save_end(epochs - 1, np.mean(g_list[-n_save * c:], axis=0), np.mean(df_list[-n_save * c:], axis=0),
             np.mean(id_list[-n_save * c:], axis=0), gen, critic, siam, aspec, n_save=1, save_path=gl_savepath)
    log(f'Mean D loss: {np.mean(df_list[-c:], axis=0)} Mean G loss: {np.mean(g_list[-c:], axis=0)} Mean ID loss: {np.mean(id_list[-c:], axis=0)} Mean Valid loss: {np.mean(val_list[-c:], axis=0)}')
    file.write(
        f'{epochs - 1},{np.mean(df_list[-c:], axis=0)},{np.mean(dr_list[-g:], axis=0)},{np.mean(g_list[-c:], axis=0)},{np.mean(id_list[-c:], axis=0)},{lr},{np.mean(val_list[-c:])}\n')
    file.flush()
    file.close()


def train_new(epochs, dstrain_o, dstrain_r, dsval_o, dsval_r, batch_size=16, lr=0.0001, n_save=6, gupt=5, startep=0):
    file = open(f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}_losses.txt", "w")

    update_lr(lr)
    d_list = []
    g_list = []
    s_list = []
    id_list = []
    val_list = []
    e_count = 0
    temp_count = 0

    for epoch in range(startep, epochs):
        bef = time.time()

        for batchi, (a, b) in enumerate(zip(dstrain_o, dstrain_r)):

            validation = dsval_o.as_numpy_iterator().next()
            val_r = dsval_r.as_numpy_iterator().next()


            if (batchi % gupt) == 0:
                loss_d, loss_g_train, loss_g_valid, loss_s = train_all_new(a, b, validation, val_r)
            else:
                loss_d = train_d_new(a, b)

            d_list.append(loss_d)
            g_list.append(loss_g_train)
            s_list.append(loss_s)
            val_list.append(loss_g_valid)
            e_count += 1
            temp_count += 1

            if batchi % 250 == 0:
                log(time.strftime("%H:%M:%S ", time.localtime()), end='')
                log(f'[Epoch {epoch}/{epochs}] [Batch {batchi}] [D loss: {np.mean(d_list[-temp_count:], axis=0)} ', end='')
                log(f'S loss: {np.mean(s_list[-temp_count:], axis=0)}] ', end='')
                log(f'[G loss: {np.mean(g_list[-temp_count:], axis=0)}] ', end='')
                log(f'[Val loss: {np.mean(val_list[-temp_count:])}] ', end='')
                log(f'[LR: {lr}]')
                temp_count = 0

            nbatch = batchi
        # end for batch

        log(f'Time for epoch {epoch}: {int(time.time() - bef)}')
        log(f'Time/Batch {(time.time() - bef) / nbatch}')

        save_end(epoch, np.mean(g_list[-n_save * e_count:], axis=0), np.mean(d_list[-n_save * e_count:], axis=0),
                 np.mean(s_list[-n_save * e_count:], axis=0), gen, critic, siam, dstrain_o, n_save=1, save_path=gl_savepath)
        log(f'Mean D loss: {np.mean(d_list[-e_count:], axis=0)} Mean G loss: {np.mean(g_list[-e_count:], axis=0)} Mean S loss: {np.mean(s_list[-e_count:], axis=0)} Mean Valid loss: {np.mean(val_list[-e_count:], axis=0)}')
        file.write(
            f'{np.mean(d_list[-e_count:], axis=0)},{np.mean(g_list[-e_count:], axis=0)},{np.mean(s_list[-e_count:], axis=0)},{lr},{np.mean(val_list[-e_count:], axis=0)}\n')
        logfile.flush()
        file.flush()

    # end for epochs

    save_end(epochs - 1, np.mean(g_list[-n_save * e_count:], axis=0), np.mean(d_list[-n_save * e_count:], axis=0),
             np.mean(s_list[-n_save * e_count:], axis=0), gen, critic, siam, dstrain_o, n_save=1, save_path=gl_savepath)
    log(f'Mean D loss: {np.mean(d_list[-e_count:], axis=0)} Mean G loss: {np.mean(g_list[-e_count:], axis=0)} Mean S loss: {np.mean(s_list[-e_count:], axis=0)} Mean Valid loss: {np.mean(val_list[-e_count:], axis=0)}')
    file.write(f'{np.mean(d_list[-e_count:], axis=0)},{np.mean(g_list[-e_count:], axis=0)},{np.mean(s_list[-e_count:], axis=0)},{lr},{np.mean(val_list[-e_count:], axis=0)}\n')
    file.flush()
    file.close()

""" ######################### START OF TRAINING CODE ########################## """


# ids, names, train_o = load_spec_array('../spec_train_o')
# train_o_split = split_equal_size(train_o)  # hier werden die daten in gleiche laenge aufgesteilt
#
# _, _, train_r = load_spec_array('../spec_train_r')
# train_r_split = split_equal_size(train_r)
# del train_r
#
# _, _, val_o = load_spec_array('../spec_val_o')
# val_o_split = split_equal_size(val_o)
#
# _, _, val_r = load_spec_array('../spec_val_r')
# val_r_split = split_equal_size(val_r)
# del val_r
#
#
# #dsa = tf.data.Dataset.from_tensor_slices(train_o_split).map(proc,num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True)
# # dsa = tf.data.Dataset.from_tensor_slices(adata).repeat(50).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True)
# # dsb = tf.data.Dataset.from_tensor_slices(bdata).repeat(50).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True)
#
# # dsa = tf.data.Dataset.from_tensor_slices(adata).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True)
# # dsb = tf.data.Dataset.from_tensor_slices(bdata).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True)
#
# dstrain_o = tf.data.Dataset.from_tensor_slices(train_o_split).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
# dstrain_r = tf.data.Dataset.from_tensor_slices(train_r_split).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
#
# dsval_o = tf.data.Dataset.from_tensor_slices(val_o_split).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
# dsval_r = tf.data.Dataset.from_tensor_slices(val_r_split).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
#
# #del train_o_split
# #del train_r_split
#
# gl_savepath = '../Ergebnisse/Versuch03_2_0_LossPaperNoID/'
# #gl_loadpath = '../Ergebnisse/Versuch01_1_0_ohneValidierung/2023-07-27-10-31_294_0.4249099_0.6567595'
#
# # continue training
# # gen, critic, siam, [opt_gen, opt_disc] = get_networks(shape, load_model=True, path=gl_loadpath)
# # train(500, dstrain_o, dstrain_r, dsval_o, batch_size=bs, lr=0.0002, n_save=6, gupt=3, startep=295)
#
# # begin training
# gen, critic, siam, [opt_gen, opt_disc, opt_siam] = get_networks2(shape, load_model=False)
# train_new(500, dstrain_o, dstrain_r, dsval_o, dsval_r, batch_size=bs, lr=0.0002, n_save=6, gupt=3)