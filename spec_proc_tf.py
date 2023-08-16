import librosa
import tensorflow as tf
import tensorflow_io as tfio
from keras.losses import MeanAbsoluteError
from keras.optimizers import Adam

from constants import *

def normalize(S):
    return np.clip((((S - GL_MIN_LEVEL_DB) / -GL_MIN_LEVEL_DB) * 2.) - 1., -1, 1)

def denormalize(S):
    return (((np.clip(S, -1, 1)+1.)/2.) * -GL_MIN_LEVEL_DB) + GL_MIN_LEVEL_DB

def melspecfunc2(waveform, nfft=6 * GL_HOP, win_len=6 * GL_HOP, sr=22050):
   specgram = tfio.audio.spectrogram(waveform, nfft, win_len, GL_HOP)
   mel_specgram = tfio.audio.melscale(specgram, sr, GL_HOP, 0, sr // 2)
   return mel_specgram

def wave_to_db_spec(wv, hop=192):
    S = tf.convert_to_tensor(wv, dtype=tf.float32)
    S = tf.reshape(S, (1, -1))
    S = melspecfunc(S)
    S = np.array(np.squeeze(S))
    S = librosa.power_to_db(S) - GL_REF_LEVEL_DB
    S = normalize(S)
    return tf.transpose(S, (1, 0))

def spectral_convergence(input, target):
    return 20 * (tf.math.log(tf.norm(input - target)) - tf.math.log(tf.norm(target)))

def melspecfunc(waveform, hop=GL_HOP, sr=GL_SR):
    n_fft = 6 * hop
    win_length = 6 * hop
    hop_length = hop
    pad = 0
    power = 2

    # Compute the spectrogram
    specgram = tf.signal.stft(waveform, frame_length=win_length, frame_step=hop_length, fft_length=n_fft, pad_end=False)
    specgram = tf.abs(specgram) ** power

    # Compute the Mel spectrogram
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=hop,
                                                                        num_spectrogram_bins=n_fft // 2 + 1,
                                                                        sample_rate=sr)
    mel_specgram = tf.matmul(specgram, linear_to_mel_weight_matrix)

    return mel_specgram

def GRAD(spec, transform_fn, samples=None, init_x0=None, maxiter=1000, tol=1e-6, verbose=1, evaiter=10, lr=0.003):
    #samples = (spec.shape[-1] * hop)
    samples = (spec.GL_SHAPE[-1] - 1) * 192 + 6 * 192

    spec = np.transpose(spec, (0, 2, 1))
    if init_x0 is None:
        init_x0 = tf.random.normal((1, samples), stddev=1e-6)
    x = tf.Variable(init_x0, dtype=tf.float32)
    T = spec

    criterion = MeanAbsoluteError()
    optimizer = Adam(learning_rate=lr)

    bar_dict = {}
    metric_func = spectral_convergence  # You need to define this function
    bar_dict['spectral_convergence'] = 0
    metric = 'spectral_convergence'

    init_loss = None
    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            with tf.GradientTape() as tape:
                V = transform_fn(x)
                loss = criterion(V, T)
            gradients = tape.gradient(loss, [x])
            optimizer.apply_gradients(zip(gradients, [x]))
            lr *= 0.9999
            optimizer.learning_rate.assign(lr)

            if i % evaiter == evaiter - 1:
                V = transform_fn(x)
                bar_dict[metric] = metric_func(V, spec).numpy().item()
                l2_loss = criterion(V, spec).numpy().item()
                pbar.set_postfix(**bar_dict, loss=l2_loss)
                pbar.update(evaiter)

    return x.numpy().flatten()
