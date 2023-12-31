"""
File containing all loss functions
used for training the networks
"""
import tensorflow as tf
import tensorflow_probability as tfp
from MAINCONFIG import GL_DELTA


#### discriminator loss
def L_d(d_target, d_g_source):
    dr_loss = L_d_real(d_target)
    df_loss = L_d_fake(d_g_source)
    return dr_loss + df_loss, df_loss, dr_loss

def L_d_fake(d_g_source):
    return tf.reduce_mean(tf.maximum(0., 1 + d_g_source))
    #return -tf.reduce_mean(tf.minimum(0., -1 - d_g_source))

def L_d_real(d_target):
    return tf.reduce_mean(tf.maximum(0., 1 - d_target))
    #return -tf.reduce_mean(tf.minimum(0., -1 + d_target))


#### generator loss
def L_g(d_g_src, l_travel, l_id):
    return L_g_adv(d_g_src) + l_id + l_travel

def L_g_adv(d_g_src):
    return -tf.reduce_mean(d_g_src)

def L_g_id(alpha: float, trgt, g_trgt):
    return alpha * tf.reduce_mean(l2_squared(g_trgt - trgt))

### loss forcing network to match a frequency distribution
def L_g_freqprio(g_src, freqmask_spec):
    #freqmask_spec already reduced mean
    g_src_edit, _ = tf.linalg.normalize(g_src, ord=1, axis=1)
    g_src_edit = tf.reduce_mean(g_src_edit, axis=2)
    freqmask_spec = tf.expand_dims(freqmask_spec, 0)
    freqmask_spec = tf.repeat(freqmask_spec, g_src.shape[0], 0)
    return tf.reduce_sum(tf.abs(freqmask_spec - g_src_edit))

#### siamese loss
def L_s(l_travel, l_s_margin):
    return l_travel + l_s_margin

#### losses used for multiple networks
def L_travel(beta: float, s_src1, s_src2, s_g_src1, s_g_src2):
    t12 = s_src1 - s_src2
    t_12 = s_g_src1 - s_g_src2
    return beta * tf.reduce_mean(cos_sim(t12, t_12) + l2_squared(t12 - t_12))

def L_s_margin(gamma: float, delta: float, src1, src2, s_src1, s_src2):
    t12 = s_src1 - s_src2
    return gamma * tf.reduce_mean(tf.maximum(0.0, delta - l2_norm(t12)))



###### sorted out
def L_g_noID(d_g_src, l_travel: float):
    return L_g_adv(d_g_src) + l_travel

'''
Get average frequency distribution of target dataset (not normalized)
'''
def get_target_avg(dstrain: tf.data.Dataset):
    result = None
    for _, _, spec in dstrain.take(1):
        result = tf.zeros(spec.shape[0], dtype=spec.dtype)
        trgt_norm = tf.zeros(spec.shape[0], dtype=spec.dtype)

    for _, _, target in dstrain:
        trgt_norm, _ = tf.linalg.normalize(target, ord=1, axis=0)
        result += tf.reduce_mean(trgt_norm, axis=1)


    return result / len(dstrain)


def map_handle_nan(x):
    if tf.math.is_nan(x):
        return 0.
    else:
        return x

"""
Measure if src and trgt are parallel enough for calculating a loss
"""
def is_parallel(src, trgt):
    res = tfp.stats.correlation(src, trgt, sample_axis=1, event_axis=None)
    res = tf.map_fn(lambda x: map_handle_nan(x), res)
    res = tf.reduce_mean(res)
    if res >= 0.5:
        return 1
    return 0

"""
Loss for comparison of parallel data 
"""
def L_g_parallel_comparison(g_src_batch, trgt_batch, src_batch):
    res = 0.
    count = 0
    for i in range(g_src_batch.shape[0]):
        if is_parallel(src_batch[i], trgt_batch[i]) == 1:
            count += 1
            res += l2_norm(g_src_batch[i] - trgt_batch[i])


    if count == 0:
        return 0.0

    return res / float(count)

##################
# HELPER FUNCTIONS
##################

def mag(x):
    return l2_norm(x)

def l2_norm(x):
    return tf.sqrt(tf.reduce_sum(tf.multiply(x, x)))

def l2_squared(x):
    return tf.reduce_sum(tf.multiply(x, x))
def dot(x, y):
    return tf.reduce_sum(tf.multiply(x, y))
def cos_sim(A, B):
    return dot(A, B) / (mag(A) * mag(B))

# Die l/2 spektrogramme werden im Code zu l/3 spektrogrammen und in den Fehlerfunktionen werden Spektrogram 1 und 3 jeweils verwertet

##################
# OLD LOSSES
##################

def mae(x,y):
    return tf.reduce_mean(tf.abs(x-y))

def mse(x,y):
    return tf.reduce_mean((x-y)**2)

def loss_travel(siam_orig1, siam_gen_orig1, siam_orig3, siam_gen_orig3):
    l1 = tf.reduce_mean(((siam_orig1 - siam_orig3) - (siam_gen_orig1 - siam_gen_orig3)) ** 2)
    l2 = tf.reduce_mean(
        tf.reduce_sum(
            -(
                    tf.nn.l2_normalize(siam_orig1 - siam_orig3, axis=[-1]) * tf.nn.l2_normalize(siam_gen_orig1 - siam_gen_orig3, axis=[-1])
            ), axis=-1
        )
    )
    return l1+l2

def loss_siamese(siam_orig1, siam_orig3):
    logits = tf.sqrt(tf.reduce_sum((siam_orig1 - siam_orig3) ** 2, axis=-1, keepdims=True))
    return tf.reduce_mean(tf.square(tf.maximum((GL_DELTA - logits), 0.0)))

def loss_d_g_src(d_g_src):
    return tf.reduce_mean(tf.maximum(1 + d_g_src, 0))

def loss_d_target(d_target):
    return tf.reduce_mean(tf.maximum(1 - d_target, 0))