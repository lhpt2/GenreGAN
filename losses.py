import numpy
import tensorflow as tf
from constants import GL_DELTA

#Losses

# Die l/2 spektrogramme werden im Code zu l/3 spektrogrammen und in den Fehlerfunktionen werden Spektrogram 1 und 3 jeweils verwertet

def mae(x,y):
    return tf.reduce_mean(tf.abs(x-y))

def mse(x,y):
    return tf.reduce_mean((x-y)**2)

def L_g_id(alpha: float, trgt, g_trgt):
   return alpha * tf.reduce_mean(l2_squared(g_trgt - trgt))
# finished

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

def L_travel(beta: float, s_src1, s_src2, s_g_src1, s_g_src2):
    t12 = s_src1 - s_src2
    t_12 = s_g_src1 - s_g_src2
    return beta * tf.reduce_mean(cos_sim(t12, t_12) + l2_squared(t12 - t_12))
#finished

def loss_siamese(siam_orig1, siam_orig3):
    logits = tf.sqrt(tf.reduce_sum((siam_orig1 - siam_orig3) ** 2, axis=-1, keepdims=True))
    return tf.reduce_mean(tf.square(tf.maximum((GL_DELTA - logits), 0.0)))

def L_s_margin(gamma: float, delta: float, src1, src2, s_src1, s_src2):
    t12 = s_src1 - s_src2
    return gamma * tf.reduce_mean(tf.maximum(0.0, delta - l2_norm(t12)))
# finished



def loss_d_g_src(d_g_src):
    return tf.reduce_mean(tf.maximum(1 + d_g_src, 0))

def loss_d_target(d_target):
    return tf.reduce_mean(tf.maximum(1 - d_target, 0))

def L_g_adv(d_g_src):
    return -tf.reduce_mean(d_g_src)

# all param losses

def L_s_full(beta: float, gamma: float, delta: float, src1, src2, s_src1, s_src2, s_g_src1, s_g_src2):
    return L_travel(beta, s_src1, s_src2, s_g_src1, s_g_src2) + L_s_margin(gamma, delta, src1, src2, s_src1, s_src2)

def L_g_full(alpha: float, beta: float, d_g_src, trgt, g_trgt, s_src1, s_src2, s_g_src1, s_g_src2):
    return L_g_adv(d_g_src) + L_g_id(alpha, trgt, g_trgt) + L_travel(beta, s_src1, s_src2, s_g_src1, s_g_src2)

# final losses

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

def L_s(l_travel, l_s_margin: float):
    return l_travel + l_s_margin

#def L_g(alpha: float, d_g_src, trgt, g_trgt, l_travel: float):
#    return L_g_adv(d_g_src) + L_g_id(alpha, trgt, g_trgt) + l_travel

def L_g(d_g_src, l_travel: float, l_id: float):
    return L_g_adv(d_g_src) + l_id + l_travel

def L_g_noID(d_g_src, l_travel: float):
    return L_g_adv(d_g_src) + l_travel

""" 
Losses code

loss_g -> L_g_adv 
loss_m -> loss_travel(siam_orig1, siam_gen_orig1, siam_orig3, siam_gen_orig3) + loss_siamese(siam_orig1, siam_orig3) 
loss_id -> (mae(remix1, gen_remix1) + mae(remix2, gen_remix2) + mae(remix3, gen_remix3)) / 3.

GEN_loss = loss_g (OK) + 10. * loss_m[travel+siamese] + 0.5 * loss_id
DISC_loss = (loss_dr + loss_df) / 2. ==> OK

"""

"""

gen_adv_l = -mean_a(D(G(a)))
gen_id_l = mean_b(l2_squared(G(b) - b))

travel_l = mean_a(cos_sim(t12, t_12) + l2_squared(t12-t_12))
t12 = S(a__i) - S(a__j)
t_12 = S(G(a__i)) - S(G(a__j))

siam_margin_l = mean_a(max(0, (delta - l2_norm(t12))))

DISC_loss = -mean_b(min((0, -1+D(b))) - mean_a(min(0, -1-D(G(a))))
GEN_loss = gen_adv_l + alpha * gen_id_l + beta * travel_l
SIAM_loss = beta * travel_l + gamma * siam_margin_l 

"""