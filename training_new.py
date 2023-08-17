import datetime
from LPrint import LPrint

import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Model
from tensorflow.python.data import AUTOTUNE

from architecture import extract_image, assemble_image
from losses import L_d, L_s_margin, L_g_noID, L_s, L_travel, L_g
from testing_network import save_end

from constants import *
from dataset_processing import load_dsparts, save_spec_to_wv
from architecture_v2 import load, build

def get_networks(shape, load_model=False, path=None):
    if not load_model:
        gen, critic, siam = build()
    else:
        gen, critic, siam = load(path)

    print('Built networks')
    opt_gen = Adam(0.0001, 0.5)
    opt_disc = Adam(0.0001, 0.5)
    opt_siam = Adam(0.0001, 0.5)

    return gen, critic, siam, [opt_gen, opt_disc, opt_siam]

def train_all(train_src, train_trgt, val_src, val_trgt, alpha = 0, beta = 10, gamma = 10, delta = 3):
    src1, src2, src3 = extract_image(train_src)
    trgt1, trgt2, trgt3 = extract_image(train_trgt)
    vals1, vals2, vals3 = extract_image(val_src)
    valr1, valr2, valr3 = extract_image(val_trgt)

    with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:  # tf.GradientTape as tape_siam:
        # translating A to B
        gen_src1 = gen(src1, training=True)
        gen_src2 = gen(src2, training=True)
        gen_src3 = gen(src3, training=True)

        # identity mapping B to B                                                        COMMENT THESE 3 LINES IF THE IDENTITY LOSS TERM IS NOT NEEDED
        gen_trgt1 = gen(trgt1, training=True)
        gen_trgt2 = gen(trgt2, training=True)
        gen_trgt3 = gen(trgt3, training=True)

        # concatenate/assemble converted spectrograms
        gen_src = assemble_image([gen_src1, gen_src2, gen_src3])
        gen_trgt = assemble_image([gen_trgt1, gen_trgt2, gen_trgt3])

        # feed concatenated spectrograms to critic
        d_gen_src = critic(gen_src, training=True)
        d_trgt = critic(train_trgt, training=True)

        # feed 2 pairs (A,G(A)) extracted spectrograms to Siamese
        siam_gen_src1 = siam(gen_src1, training=True)
        siam_gen_src2 = siam(gen_src3, training=True)
        siam_src1 = siam(src1, training=True)
        siam_src3 = siam(src3, training=True)

        gen_vals1 = gen(vals1, training=False)
        gen_vals2 = gen(vals2, training=False)
        gen_vals3 = gen(vals3, training=False)
        gen_vals = assemble_image([gen_vals1, gen_vals2, gen_vals3])

        gen_valr1 = gen(valr1, training=False)
        gen_valr2 = gen(valr2, training=False)
        gen_valr3 = gen(valr3, training=False)
        gen_valr = assemble_image([gen_valr1, gen_valr2, gen_valr3])

        siam_gen_vals1 = siam(gen_vals1, training=False)
        siam_gen_vals3 = siam(gen_vals3, training=False)
        siam_vals1 = siam(vals1, training=False)
        siam_vals3 = siam(vals3, training=False)

        d_gen_vals = critic(gen_vals, training=False)

        loss_d = L_d(d_trgt, d_gen_src)

        loss_s_margin = L_s_margin(delta, src1, src3, siam_src1, siam_src3)
        l_travel = L_travel(siam_src1, siam_src3, siam_gen_src1, siam_gen_src2)
        lossgtot = L_g(alpha, beta, d_gen_src, train_trgt, gen_trgt, l_travel)
        loss_g_train = L_g_noID(beta, d_gen_src, l_travel)
        loss_s = L_s(beta, gamma, l_travel, loss_s_margin)

        l_travel_val = L_travel(siam_vals1, siam_vals3, siam_gen_vals1, siam_gen_vals3)
        loss_g_valid = L_g_noID(beta, d_gen_vals, l_travel_val)
         #loss_g_valid = L_g_full(0.5, 10, d_gen_vals, validation_remix, gen_valr, gen_vals1, gen_vals2, siam_gen_vals1, siam_gen_vals3)

    grad_gen = tape_gen.gradient(loss_g_train, gen.trainable_variables + siam.trainable_variables)
    gl_opt_gen.apply_gradients(zip(grad_gen, gen.trainable_variables + siam.trainable_variables))

    grad_disc = tape_disc.gradient(loss_d, critic.trainable_variables)
    gl_opt_disc.apply_gradients(zip(grad_disc, critic.trainable_variables))

    # grad_siam = tape_siam.gradient(loss_s, siam.trainable_variables)
    # opt_siam.apply_gradients(zip(grad_siam, siam.trainable_variables))

    return loss_d, loss_g_train, loss_g_valid, loss_s

def train_d(sample_src, sample_trgt):
    src1, src2, src3 = extract_image(sample_src)

    with tf.GradientTape() as tape_disc:
        g_orig1 = gen(src1, training=True)
        g_orig2 = gen(src2, training=True)
        g_orig3 = gen(src3, training=True)
        g_src = assemble_image([g_orig1, g_orig2, g_orig3])

        d_g_src = critic(g_src, training=True)
        d_trgt = critic(sample_trgt, training=True)

        loss_d = L_d(d_trgt, d_g_src)
        #loss_d = (loss_dr + loss_df) / 2.

    grad_disc = tape_disc.gradient(loss_d, critic.trainable_variables)
    gl_opt_disc.apply_gradients(zip(grad_disc, critic.trainable_variables))

    return loss_d

def train(ds_train: tf.data.Dataset, ds_val: tf.data.Dataset, epochs: int = 300, batch_size=16, lr=0.0001, n_save=6,
          gen_update=5, startep=0):
    # LOGGING
    csvfile = open(f"{GL_SAVE}/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}_losses.txt", "w")

    # UPDATE learning rate
    gl_opt_gen.learning_rate = lr
    gl_opt_disc.learning_rate = lr

    # GET validation iterator
    val_pool = dsval.as_numpy_iterator()

    lossprint = LPrint()

    # epoch count for integrated loss
    ep_count = 0
    temp_count = 0

    # Training Loop epochs
    for epoch in range(startep, epochs):
        before = time.time()

        nbatch = 0
        # Training Loop batches
        for batch_nr, elem in enumerate(ds_train):
            #id = elem[0]
            sample_src = elem[1]
            sample_trgt = elem[2]

            # Train generator ever gen_update batches
            val = val_pool.next()
            if (batch_nr % gen_update) == 0:
                losses = train_all(sample_src, sample_trgt, val[1], val[2], GL_ALPHA, GL_BETA, GL_GAMMA, GL_DELTA)
            else:
                losses = train_d(sample_src, sample_trgt)

            # update losses and counters
            temp_count += 1
            ep_count += 1
            lossprint.append(losses)

            # Print status every 100 batches
            if batch_nr % 100 == 0:
                log(time.strftime("%H:%M:%S ", time.localtime()), end='')
                # log epoch/epochs batch_nr d_loss d_loss_r d_loss_f g_loss s_loss id_loss lr
                msgstr = f'[Epoch {epoch}/{epochs}] [Batch {batch_nr}] '
                msgstr += lossprint.to_str(-temp_count)
                msgstr += f'[LR: {lr}]\n'
                log(msgstr)
                temp_count = 0

            nbatch = batch_nr

        # END FOR BATCH

        # print time for epoch and time for batch
        log(f'Time for epoch {epoch}: {int(time.time() - before)}')
        log(f'Time/Batch {(time.time() - before) / nbatch}')

        # save weights every n_save epochs
        ll_dict = lossprint.get_mean(-n_save * ep_count)
        save_end(epoch, ll_dict['gloss'], ll_dict['dfloss'],
             ll_dict['idloss'], gen, critic, siam, ds_train.shuffle(5).take(1)[1], n_save=n_save, save_path=GL_SAVE)

        # print losses and write to loss_file
        ll_dict = lossprint.get_mean(-ep_count)
        losses_csv = f'{epoch},{lossprint.to_csv_part(-ep_count)},{lr}\n'
        log(f'Mean D loss: {ll_dict["dloss"]} Mean G loss: {ll_dict["gloss"]} Mean ID loss: {ll_dict["idloss"]}, Mean S loss: {ll_dict["sloss"]}')
        csvfile.write(losses_csv)
        #csvfile.write(f'{epoch},{np.mean(df_list[-ep_count:], axis=0)},{np.mean(dr_list[-ep_count:], axis=0)},{np.mean(g_list[-ep_count:], axis=0)},{np.mean(id_list[-ep_count:], axis=0)},{lr},{np.mean(val_list[-ep_count:])}\n')
        ep_count = 0
        logfile.flush()
        csvfile.flush()

    # save weights every n_save epochs
    ll_dict = lossprint.get_mean_all()
    save_end(epochs - 1, ll_dict['gloss'], ll_dict['dfloss'],
            ll_dict['idloss'], gen, critic, siam, ds_train.shuffle(5).take(1)[1], n_save=n_save, save_path=GL_SAVE)

    # print losses and write to loss_file
    ll_dict = lossprint.get_mean(-ep_count)
    losses_csv = f'{epochs - 1},{lossprint.to_csv_part(-ep_count)},{lr}\n'
    log(f'Mean D loss: {ll_dict["dloss"]} Mean G loss: {ll_dict["gloss"]} Mean ID loss: {ll_dict["idloss"]}, Mean S loss: {ll_dict["sloss"]}')
    csvfile.write(losses_csv)
    #csvfile.write(f'{epoch},{np.mean(df_list[-ep_count:], axis=0)},{np.mean(dr_list[-ep_count:], axis=0)},{np.mean(g_list[-ep_count:], axis=0)},{np.mean(id_list[-ep_count:], axis=0)},{lr},{np.mean(val_list[-ep_count:])}\n')
    logfile.flush()
    csvfile.flush()

GL_SAVE = '../Ergebnisse/Versuch04_2_0_LossPaperNoID/'
GL_LOAD = '../Ergebnisse/Versuch01_1_0_ohneValidierung/2023-07-27-10-31_294_0.4249099_0.6567595'

if __name__ == "__main__":
    dsval = load_dsparts("dsvalQuick")
    dstrain = load_dsparts('dstrainQuick')

    dsval = dsval.shuffle(10000).batch(GL_BS, drop_remainder=True).prefetch(AUTOTUNE)
    dstrain = dstrain.shuffle(10000).batch(GL_BS, drop_remainder=True).prefetch(AUTOTUNE)

    #dsval = dsval.shuffle(10000).prefetch(AUTOTUNE)
    #dstrain = dstrain.shuffle(10000).prefetch(AUTOTUNE)

    # do things: get networks with proper size (shape should be changed)
    gen, critic, siam, [gl_opt_gen, gl_opt_disc, gl_opt_siam] = get_networks(GL_SHAPE, load_model=False)

    print(gen.summary())
    print(critic.summary())
    print(siam.summary())

    # start training
    train(dstrain, dsval, 500, batch_size=GL_BS, lr=0.0001, n_save=6, gen_update=5, startep=0)