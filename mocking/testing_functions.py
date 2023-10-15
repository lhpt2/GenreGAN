"""
Attempt to create dummyfunctions for debugging
errors in the code
"""
import random
import numpy as np

from constants import log
from training import make_losses_string, make_csv_string

def dummy_losses_d():
    loss_d = random.uniform(0.3, 1000.4578)
    loss_df = random.uniform(0.3, 1000.4578)
    loss_dr = random.uniform(0.3, 1000.4578)
    return loss_d, loss_df, loss_dr

def dummy_losses_all():
    loss_g = random.uniform(0.3, 1000.4578)
    loss_d = random.uniform(0.3, 1000.4578)
    loss_df = random.uniform(0.3, 1000.4578)
    loss_dr = random.uniform(0.3, 1000.4578)
    loss_s = random.uniform(0.3, 1000.4578)
    l_id = random.uniform(0.3, 1000.4578)
    loss_g_val = random.uniform(0.3, 1000.4578)

    return loss_g, loss_d, loss_df, loss_dr, loss_s, l_id, loss_g_val

def testfunc(epochs, batchsize, num_batches, gen_update=5, n_save=6):

    status_every_nbatch = 100
    lr = 0.0001
    d_list = []
    df_list = []
    dr_list = []
    g_list = []
    s_list = []
    id_list = []
    val_list = []

    batch_count = 0
    final_batch_c = 0

    loss_g = 0
    loss_d = 0
    loss_df = 0
    loss_dr = 0
    loss_s = 0
    l_id = 0
    loss_g_val = 0

    for epoch in range(epochs):

        for batch_nr in range(num_batches):

            if (batch_nr % gen_update) == 0:
                loss_g, loss_d, loss_df, loss_dr, loss_s, l_id, loss_g_val = dummy_losses_all()
            else:
                loss_d, loss_df, loss_dr = dummy_losses_d()

            d_list.append(loss_d)
            df_list.append(loss_df)
            dr_list.append(loss_dr)
            g_list.append(loss_g)
            s_list.append(loss_s)
            id_list.append(l_id)
            val_list.append(loss_g_val)
            batch_count += 1

            if batch_nr % status_every_nbatch == 0:
                msgstr = f'[Epoch {epoch}/{epochs}] [Batch {batch_nr}] '
                msgstr += make_losses_string(d_list, dr_list, df_list, g_list, s_list, id_list, val_list, -status_every_nbatch, 4)
                print(msgstr)


        if (epoch % n_save) == 0:
            gloss = np.mean(g_list[-n_save * batch_count:], axis=0),
            dloss = np.mean(d_list[-n_save * batch_count:], axis=0),
            sloss = np.mean(s_list[-n_save * batch_count:], axis=0),

        print(f'Mean D loss: {np.mean(d_list[-batch_count:], axis=0)} Mean G loss: {np.mean(g_list[-batch_count:], axis=0)} Mean Val loss: {np.mean(val_list[-batch_count:], axis=0)} Mean ID loss: {np.mean(id_list[-batch_count:], axis=0)}, Mean S loss: {np.mean(s_list[-batch_count:], axis=0)}')


        losses_csv = f'{epoch},{make_csv_string(d_list, dr_list, df_list, g_list, s_list, id_list, val_list, -batch_count, 9)},{lr}\n'
        print(f"CSV: {losses_csv}")

        final_batch_c = batch_count
        batch_count = 0

    last_nsave = (epochs - 1) % n_save * final_batch_c
    log(f'Mean D loss: {np.mean(d_list[-last_nsave:], axis=0)} Mean G loss: {np.mean(g_list[-last_nsave:], axis=0)} Mean Val loss: {np.mean(val_list[-last_nsave:], axis=0)} Mean ID loss: {np.mean(id_list[-last_nsave:], axis=0)}, Mean S loss: {np.mean(s_list[-last_nsave:], axis=0)}')
    losses_csv = f'{epochs - 1},{make_csv_string(d_list, dr_list, df_list, g_list, s_list, id_list, val_list, -last_nsave, 6)},{lr}\n'
    print(f"CSV: {losses_csv}")