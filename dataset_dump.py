import datetime
import os
import signal

import tensorflow as tf
import numpy as np

from constants import *

from preprocessing import convert_audio_array_to_spec_array, load_audio_array, split_equal_size, load_spec_array
from losses import loss_travel, loss_siamese, L_g_adv, loss_d_gen_orig, loss_d_remix
from architecture import build, load, extract_image, assemble_image, get_networks
from testing_network import save_end, use_generator



def handler(signum, frame):
    print("Aborting process, ctrl-C")
    logfile.flush()
    logfile.close()


signal.signal(signal.SIGINT, handler)

"""
zufaellige ausschnitte der groesse 3 * shape ausstanzen """


@tf.function
def proc(x):
    return tf.image.random_crop(x, size=[hop, 3 * shape, 1])


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

        # identity mapping loss
        # loss_id = (mae(remix1,gen_remix1)+mae(remix2,gen_remix2)+mae(remix3,gen_remix3))/3.                         #loss_id = 0. IF THE IDENTITY LOSS TERM IS NOT NEEDED
        loss_id = 0
        # travel loss
        loss_m = loss_travel(siam_orig1, siam_gen_orig1, siam_orig3, siam_gen_orig3) + loss_siamese(siam_orig1,
                                                                                                    siam_orig3)
        # generator and critic losses
        loss_g = L_g_adv(critic_gen_orig)
        loss_dr = loss_d_remix(critic_remix)
        loss_df = loss_d_gen_orig(critic_gen_orig)
        loss_d = (loss_dr + loss_df) / 2.
        # generator+siamese total loss
        lossgtot = loss_g + 10. * loss_m + 0.5 * loss_id  # CHANGE LOSS WEIGHTS HERE  (COMMENT OUT +w*loss_id IF THE IDENTITY LOSS TERM IS NOT NEEDED)

        gen_valid1 = gen(valid1, training=False)
        gen_valid2 = gen(valid2, training=False)
        gen_valid3 = gen(valid3, training=False)
        gen_valid = assemble_image([gen_valid1, gen_valid2, gen_valid3])
        siam_gen_valid1 = siam(gen_valid1, training=False)
        siam_gen_valid3 = siam(gen_valid3, training=False)
        siam_valid1 = siam(valid1, training=False)
        siam_valid3 = siam(valid3, training=False)

        critic_gen_valid = critic(gen_valid, training=False)
        loss_g_valid = L_g_adv(critic_gen_valid) + 10. * loss_travel(siam_valid1, siam_gen_valid1, siam_valid3,
                                                                     siam_gen_valid3) + 0.5 * loss_siamese(siam_valid1,
                                                                                                           siam_valid3)

    # computing and applying gradients
    grad_gen = tape_gen.gradient(lossgtot, gen.trainable_variables + siam.trainable_variables)
    opt_gen.apply_gradients(zip(grad_gen, gen.trainable_variables + siam.trainable_variables))

    grad_disc = tape_disc.gradient(loss_d, critic.trainable_variables)
    opt_disc.apply_gradients(zip(grad_disc, critic.trainable_variables))

    return loss_dr, loss_df, loss_g, loss_id, lossgtot, loss_g_valid


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

        loss_dr = loss_d_remix(cb)
        loss_df = loss_d_gen_orig(cab)

        loss_d = (loss_dr + loss_df) / 2.

    grad_disc = tape_disc.gradient(loss_d, critic.trainable_variables)
    opt_disc.apply_gradients(zip(grad_disc, critic.trainable_variables))

    return loss_dr, loss_df


""" Trainings Schleife zum Laufenlassen des Trainings """


def train(epochs, aspec, batch_size=16, lr=0.0001, n_save=6, gupt=5, startep=0):
    file = open(f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}_losses.txt", "w")

    update_lr(lr)
    df_list = []
    dr_list = []
    g_list = []
    id_list = []
    val_list = []
    c = 0
    g = 0

    for epoch in range(startep, epochs):
        bef = time.time()

        for batchi, (a, b) in enumerate(zip(dstrain_o, dstrain_r)):

            validation = dsval_o.as_numpy_iterator().next()

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


""" ######################### START OF TRAINING CODE ########################## """

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

train_o = load_spec_array('../spec_train_o')
train_o_split = split_equal_size(train_o)  # hier werden die daten in gleiche laenge aufgesteilt

train_r = load_spec_array('../spec_train_r')
train_r_split = split_equal_size(train_r)
del train_r

val_o = load_spec_array('../spec_val_o')
val_o_split = split_equal_size(val_o)

val_r = load_spec_array('../spec_val_r')
val_r_split = split_equal_size(val_r)

# dsa = tf.data.Dataset.from_tensor_slices(adata).repeat(50).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True)
# dsb = tf.data.Dataset.from_tensor_slices(bdata).repeat(50).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True)

# dsa = tf.data.Dataset.from_tensor_slices(adata).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True)
# dsb = tf.data.Dataset.from_tensor_slices(bdata).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True)

#dstrain_o = tf.data.Dataset.from_tensor_slices(train_o_split).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
#dstrain_r = tf.data.Dataset.from_tensor_slices(train_r_split).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

#dsval_o = tf.data.Dataset.from_tensor_slices(val_o_split).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
#dsval_r = tf.data.Dataset.from_tensor_slices(val_r_split).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

dstrain = tf.data.Dataset.from_tensor_slices((train_o_split, train_r_split))
dsval = tf.data.Dataset.from_tensor_slices((val_o_split, val_r_split))

trainwriter = tf.io.TFRecordWriter("./TrainDS.tfrecords")
valwriter = tf.io.TFRecordWriter("./ValDS.tfrecords")

trainwriter.write(dstrain)
valwriter.write(dsval)
