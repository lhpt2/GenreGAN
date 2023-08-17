import os

from pathlib import Path
import numpy as np
import librosa
import torch
import torch.nn as nn
import tensorflow as tf
from tqdm import tqdm
from torchaudio.transforms import Spectrogram, MelScale
from glob import glob

from constants import *
from constants import secs_to_bins

torch.set_default_tensor_type('torch.cuda.FloatTensor')

specobj = Spectrogram(n_fft=6 * GL_HOP, win_length=6 * GL_HOP, hop_length=GL_HOP, pad=0, power=2, normalized=True)
specfunc = specobj.forward
melobj = MelScale(n_stft=(6 * GL_HOP) // 2 + 1, n_mels=GL_HOP, sample_rate=GL_SR, f_min=0.)
melfunc = melobj.forward

def wave_to_db_spec(wv, hop=192):
    S = torch.Tensor(wv).view(1, -1)
    S = melspecfunc(S)
    S = np.array(torch.squeeze(S).detach().cpu())
    S = librosa.power_to_db(S) - GL_REF_LEVEL_DB
    return normalize(S)

def db_spec_to_wave(S):
    S = denormalize(S) + GL_REF_LEVEL_DB
    S = librosa.db_to_power(S)
    wv = GRAD(np.expand_dims(S,0), melspecfunc, maxiter=2000, evaiter=10, tol=1e-8)
    return np.array(np.squeeze(wv))

def load_single_audio(filepath):
    x, _ = tf.audio.decode_wav(tf.io.read_file(filepath), 1)
    x = np.array(x, dtype=np.float32)
    return x

""" Waveform array from path of folder containing wav files """
def load_audio_array(path):
    ls = glob(f'{path}/*.wav')
    adata = []
    for i in range(len(ls)):
        x, _ = tf.audio.decode_wav(tf.io.read_file(ls[i]), 1)
        x = np.array(x, dtype=np.float32)
        adata.append(x)

    if len(adata) > 1:
        return np.array(adata, dtype=object)
    else:
        return np.array(adata, dtype=np.float32)

""" Read spectrograms from hard drive remix and original combined"""
def load_spec_o_r_arrays(spec_o_path, spec_r_path):
    filenames = glob(f'{spec_o_path}/*.npy')
    o_specs = np.empty(len(filenames), dtype=object)
    r_specs = np.empty(len(filenames), dtype=object)
    ids: [int] = [None] * len(filenames)
    names: [str] = [None] * len(filenames)

    for i in range(len(filenames)):
        name = os.path.splitext(os.path.basename(filenames[i]))[0]
        name = name.split('_')
        idstr = name[0]
        id = int(idstr)
        name = '_'.join(name[2:])

        rname = os.path.basename(filenames[i]).split('_')
        rname = idstr + '_r_' + '_'.join(rname[2:])

        o_spec = np.load(filenames[i])
        r_spec = np.load(spec_r_path + '/' + rname)
        o_specs[i] = np.expand_dims(o_spec, -1)
        r_specs[i] = np.expand_dims(r_spec, -1)
        ids[i] = id
        name = name.encode()
        names[i] = name

    return np.array(ids), np.array(names), o_specs, r_specs


""" Read spectrograms from hard drive """
def load_spec_array(path):
    ls = glob(f'{path}/*.npy')
    specs=np.empty(len(ls), dtype=object)
    ids: [int] = []
    names: [str] = []
    for i in range(len(ls)):
        name = os.path.splitext(os.path.basename(ls[i]))[0]
        id = int(name.split('_')[0])
        name = '_'.join(name.split('_')[2:])
        x = np.load(ls[i])
        x = np.array(x, dtype=np.float32)
        specs[i] = np.expand_dims(x, -1)
        ids.append(id)

    return ids, names, specs

""" Generate spectrograms from waveform array """
def convert_audio_array_to_spec_array(data):
    specs=np.empty(data.shape[0], dtype=object)
    for i in range(data.shape[0]):
        x = data[i]
        S = wave_to_db_spec(x)
        S = np.array(S, dtype=np.float32)
        specs[i]=np.expand_dims(S, -1)
    print(specs.shape)
    return specs

""" Generate multiple spectrograms with a determined length from single wav file """
def load_audiofile_to_multispec(path, length=4 * 16000):
    x, sr = librosa.load(path,sr=16000)
    x,_ = librosa.effects.trim(x)
    loudls = librosa.effects.split(x, top_db=50)
    xls = np.array([])
    for interv in loudls:
        xls = np.concatenate((xls,x[interv[0]:interv[1]]))
    x = xls
    num = x.shape[0]//length
    specs=np.empty(num, dtype=object)
    for i in range(num-1):
        a = x[i*length:(i+1)*length]
        S = wave_to_db_spec(a)
        S = np.array(S, dtype=np.float32)
        try:
            sh = S.shape
            specs[i]=S
        except AttributeError:
            print('spectrogram failed')
    print(specs.shape)
    return specs

""" Split spectrograms in chunks with equal size 
hier bekommen alle stuecke die gleiche laenge """
def split_equal_size(specarray: np.ndarray):
    ls = []
    local_minimum = 0

    maxspeclen = 10 * GL_SHAPE                                                              #max spectrogram length

    # fuer jedes element in data
    for i in range(specarray.shape[0] - 1):

        if specarray[i].shape[1]<=specarray[i + 1].shape[1]:
            local_minimum = specarray[i].shape[1]
        else:
            local_minimum = specarray[i + 1].shape[1]
        # ermittle kuerzeste spektrumlaenge

        if 3*GL_SHAPE <= local_minimum < maxspeclen:
            maxspeclen = local_minimum
        # setze kuerzeste spektrumlaenge auf maximallänge

    for spec_nr in range(specarray.shape[0]):
        spec = specarray[spec_nr]
        if spec.shape[1]>=3*GL_SHAPE:
            for j in range(spec.shape[1] // maxspeclen):
                ls.append(spec[:,j*maxspeclen:j*maxspeclen+maxspeclen,:])

            ls.append(spec[:,-maxspeclen:,:])

    print(bins_to_secs(maxspeclen))

    # teile spektrogramme in stuecke der laenge maxspeclen und fuege alle in liste ein

    return np.array(ls)

def shorten_to_shortest_spec(specarray: np.ndarray):
    specarray = np.squeeze(specarray)
    min_t_len = specarray[0].shape[1]
    finalres = []

    # dimensionen: nr_spec, n_mels, time

    # calc shortest spectrogram
    for spec_nr in range(specarray.shape[0]):
        spec = specarray[spec_nr]
        if spec.shape[1] < min_t_len:
            min_t_len = spec.shape[1]

    for spec_nr in range(specarray.shape[0]):
        spec = specarray[spec_nr]
        spec = spec[:, :min_t_len]
        finalres.append(spec)

    return np.squeeze(np.array(finalres))

def enlarge_to_longest_spec(specarray: np.ndarray):
    specarray = np.squeeze(specarray)
    max_t_len = specarray[0].shape[1]

    # calc longest spectrogram
    for spec_nr in range(specarray.shape[0]):
        spec = specarray[spec_nr]
        if spec.shape[1] > max_t_len:
            max_t_len = spec.shape[1]

    # create proto spec
    proto_spec = np.zeros((specarray.shape[0], specarray[0].shape[0], max_t_len))

    for spec_nr in range(specarray.shape[0]):
        spec = np.squeeze(specarray[spec_nr])
        proto_spec[spec_nr, :spec.shape[0], :spec.shape[1]] = spec

    return proto_spec

def split_size_of_secs(specarray: np.ndarray, secs: int = 5, hop = GL_HOP):
    ls = []
    specarray = np.squeeze(specarray)
    maxlen = specarray[0].shape[1]

    # calc spectral time from given spec duration
    bin_nr = secs_to_bins(secs)

    # calc longest spectrogram
    for spec_nr in range(specarray.shape[0]):
        spec = specarray[spec_nr]
        if spec.shape[1] > maxlen:
            maxlen = spec.shape[1]

    # calc new dimension from time axis
    nr_subspecs = np.ceil(maxlen / bin_nr)

    for spec_nr in range(specarray.shape[0]):
        spec = specarray[spec_nr]
        protospec = np.zeros((hop ,maxlen))
        print(spec.shape, protospec.shape)
        protospec[:,:spec.shape[1]] = spec
        ls.append(protospec)

    # teile spektrogramme in stuecke der laenge maxspeclen und fuege alle in liste ein

    return np.array(ls)

""" Concatenate spectrograms in array along the time axis """
def concat_specarray(specarray: np.ndarray):
    first = True

    merged_spec = np.array([])
    nr_specs = specarray.shape[0]

    for i in range(nr_specs):
        spec = specarray[i]
        spec = np.squeeze(spec)
        if first:
            merged_spec = spec
            first = False
        else:
            merged_spec = np.concatenate((merged_spec,spec), axis=1)

    return np.squeeze(merged_spec)

def melspecfunc(waveform):
  specgram = specfunc(waveform)
  #print(specgram.shape)
  mel_specgram = melfunc(specgram)
  #print(mel_specgram.shape)
  return mel_specgram

def spectral_convergence(input, target):
    return 20 * ((input - target).norm().log10() - target.norm().log10())

def GRAD(spec, transform_fn, samples=None, init_x0=None, maxiter=1000, tol=1e-6, verbose=1, evaiter=10, lr=0.003):

    spec = torch.Tensor(spec)
    samples = (spec.shape[-1] * GL_HOP) - GL_HOP

    if init_x0 is None:
        init_x0 = spec.new_empty((1,samples)).normal_(std=1e-6)
    x = nn.Parameter(init_x0)
    T = spec

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam([x], lr=lr)

    bar_dict = {}
    metric_func = spectral_convergence
    bar_dict['spectral_convergence'] = 0
    metric = 'spectral_convergence'

    init_loss = None
    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            optimizer.zero_grad()
            V = transform_fn(x)
            loss = criterion(V, T)
            loss.backward()
            optimizer.step()
            lr = lr*0.9999
            for param_group in optimizer.param_groups:
              param_group['lr'] = lr

            if i % evaiter == evaiter - 1:
                with torch.no_grad():
                    V = transform_fn(x)
                    bar_dict[metric] = metric_func(V, spec).item()
                    l2_loss = criterion(V, spec).item()
                    pbar.set_postfix(**bar_dict, loss=l2_loss)
                    pbar.update(evaiter)

    return x.detach().view(-1).cpu()

def normalize(S):
  return np.clip((((S - GL_MIN_LEVEL_DB) / -GL_MIN_LEVEL_DB) * 2.) - 1., -1, 1)

def denormalize(S):
  return (((np.clip(S, -1, 1)+1.)/2.) * -GL_MIN_LEVEL_DB) + GL_MIN_LEVEL_DB
