"""
"""
from glob import glob
from tqdm import tqdm
from torchaudio.transforms import MelScale, Spectrogram

import tensorflow as tf
import os
import librosa
import soundfile as sf
import torch
import torch.nn as nn

from MAINCONFIG import *

specobj = Spectrogram(n_fft=6 * GL_HOP, win_length=6 * GL_HOP, hop_length=GL_HOP, pad=0, power=2, normalized=True)
specfunc = specobj.forward
melobj = MelScale(n_stft=(6 * GL_HOP) // 2 + 1, n_mels=GL_HOP, sample_rate=GL_SR, f_min=0.)
melfunc = melobj.forward

def melspecfunc(waveform):
    specgram = specfunc(waveform)
    mel_specgram = melfunc(specgram)
    return mel_specgram

def load_single_audio(filepath):
    x, _ = tf.audio.decode_wav(tf.io.read_file(filepath), 1)
    x = np.array(x, dtype=np.float32)
    return x

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

def wave_to_db_spec(wv, hop=192):
    S = torch.Tensor(wv).view(1, -1)
    S = melspecfunc(S)
    S = np.array(torch.squeeze(S).detach().cpu())
    S = librosa.power_to_db(S) - GL_REF_LEVEL_DB
    return normalize(S)

def db_spec_to_wave(S):
    S = denormalize(S) + GL_REF_LEVEL_DB
    S = librosa.db_to_power(S)
    wv = GRAD(np.expand_dims(S, 0), melspecfunc, maxiter=2000, evaiter=10, tol=1e-8)
    return np.array(np.squeeze(wv))

def save_spec_to_wv(spec, filepath='./test.wav'):
    wv = db_spec_to_wave(spec)
    sf.write(filepath, wv, GL_SR)

def load_spec_array_splitseconds(o_path, min=0, max=0, sec: int = 5):
    listing_orig = glob(f'{o_path}/*.npy')
    r_path = o_path.replace('_o', '_r')

    ids = []
    olist = []
    rlist = []

    if max == 0 or max > len(listing_orig):
        max = len(listing_orig)

    for idx in range(min, max):
        print(f"Loading spectrogram nr. {idx}...")
        o_name = os.path.basename(listing_orig[idx])
        id = int(o_name.split('_')[0])
        splitname = o_name.split('_')
        r_name = splitname[0] + '_r_' + '_'.join(splitname[2:])

        # load songs as specs
        orig = np.load(listing_orig[idx])
        remix = np.load(r_path + '/' + r_name)

        id, orig, remix = split_xsec_size(id, orig, remix, bins=(3 * GL_SHAPE))

        # append to final list
        ids.append(id)
        olist.append(orig)
        rlist.append(remix)

    ids = np.array(ids)
    olist = np.array(olist)
    rlist = np.array(rlist)
    print("finished")
    return ids, olist, rlist

def load_wv_array_splitseconds_checked(o_path, min=0, max=0):
    listing_orig = glob(f'{o_path}_o/*.wav')
    r_path = f'{o_path}_r'

    ids = []
    olist = []
    rlist = []

    # set max into appropriate range
    if max == 0 or max > len(listing_orig):
        max = len(listing_orig)

    # set max into appropriate range
    for idx in range(min, max):
        print(f"Loading spectrogram nr. {idx}...")
        # get filenames and ids
        o_name = os.path.basename(listing_orig[idx])
        id = int(o_name.split('_')[0])
        splitname = o_name.split('_')
        r_name = splitname[0] + '_r_' + '_'.join(splitname[2:])

        orig = load_single_audio(listing_orig[idx])
        orig = wave_to_db_spec(orig, GL_HOP)
        remix = load_single_audio(r_path + '/' + r_name)
        remix = wave_to_db_spec(remix, GL_HOP)

        id, orig, remix = split_xsec_size(id, orig, remix, bins=3 * GL_SHAPE)

        # test for same length of remix and original
        # take length of original as length for original and remix
        # append to lists that will be returned
        if orig.shape[0] != remix.shape[0]:
            print("Unbalanced")
            for i in range(orig.shape[0]):
                ids.append(id[i])
                olist.append(orig[i])
                rlist.append(remix[i])
        else:
            # append to final list
            for i, o, r in zip(id, orig, remix):
                ids.append(i)
                olist.append(o)
                rlist.append(r)

    # test for same amount of track slices in source and target
    if len(olist) != len(rlist):
        print("ERROR lists of diff size")
        exit(0)

    #print(f'Creating tensors from id({len(ids)}) olist({len(olist)}), rlist({len(rlist)})')
    print("finished")
    # return all final lists as tuple
    return np.array(ids, dtype=np.float32), np.array(olist, dtype=np.float32), np.array(rlist, dtype=np.float32)

def load_spec_array_splitseconds_checked(o_path, min=0, max=0, sec: int = 5):
    '''

    This function loads .npy waveform data from the given directory path,
    and returns a tuple of arrays. It checks the source and target arrays for the same size.

    :param o_path: Path must point to a directory ending with _o and another one with the same name, but ending with _r
                   The one with _o at the end contains source files (originals) the other one contains target files (remixes)
                   The naming of the files must
    :param min: Start at file nr min
    :param max: End at file nr max - 1
    :param sec:
    :return: tuple of numpy arrays (identifiers, sourcedata, targetdata)
    '''

    # get originals file list
    listing_orig = glob(f'{o_path}_o/*.npy')
    # construct remix path
    r_path = f'{o_path}_r'

    ids = []
    olist = []
    rlist = []

    # set max into appropriate range
    if max == 0 or max > len(listing_orig):
        max = len(listing_orig)

    # cycle through all individual tracks
    for idx in range(min, max):
        print(f"Loading spectrogram nr. {idx}...")
        # get filenames and ids
        o_name = os.path.basename(listing_orig[idx])
        id = int(o_name.split('_')[0])
        splitname = o_name.split('_')
        r_name = splitname[0] + '_r_' + '_'.join(splitname[2:])

        # load songs as specs
        orig = np.load(listing_orig[idx])
        remix = np.load(r_path + '/' + r_name)

        # split track into specs of length bins
        id, orig, remix = split_xsec_size(id, orig, remix, bins=3 * GL_SHAPE)

        # test for same length of remix and original
        # take length of original as length for original and remix
        # append to lists that will be returned
        if orig.shape[0] != remix.shape[0]:
            print("Unbalanced")
            for i in range(orig.shape[0]):
                ids.append(id[i])
                olist.append(orig[i])
                rlist.append(remix[i])
        else:
            # append to final list
            for i, o, r in zip(id, orig, remix):
                ids.append(i)
                olist.append(o)
                rlist.append(r)

    # test for same amount of track slices in source and target
    if len(olist) != len(rlist):
        print("ERROR lists of diff size")
        exit(0)

    #print(f'Creating tensors from id({len(ids)}) olist({len(olist)}), rlist({len(rlist)})')
    print("finished")
    # return all final lists as tuple
    return np.array(ids, dtype=np.float32), np.array(olist, dtype=np.float32), np.array(rlist, dtype=np.float32)

def load_single_spec_splitsecond(o_file, secs: int = 5):
    # get path of remix
    id = os.path.basename(o_file).split('_')[0]
    r_path = os.path.dirname(o_file).replace('_o', '_r')
    r_name = os.path.basename(o_file).split('_')
    r_name = r_name[0] + '_r_' + '_'.join(r_name[2:])

    orig = np.load(o_file)
    remix = np.load(r_path + '/' + r_name)

    id, orig, remix = split_xsec_size(int(id), orig, remix, secs)

    return id, orig, remix

# split arrays of data into slices of either bins fft-bins or secs (default: secs = 5) if bins is 0
def split_xsec_size(id: int, orig: np.ndarray, remix: np.ndarray, secs: int = 5, bins: int = 0):
    # shorten to shorter sample
    if bins != 0:
        binlen = bins
    else:
        binlen = secs_to_bins(secs)

    print("Binsize / Speclength:", binlen)

    maxdur = orig.shape[1]
    if maxdur > remix.shape[1]:
        maxdur = remix.shape[1]

    maxdur = (maxdur // binlen) * binlen
    orig = orig[:, :maxdur]
    remix = remix[:, :maxdur]

    num_slices = int(maxdur / binlen)

    ids = []
    olist = []
    rlist = []
    for i in range(num_slices):
        ids.append((id, i))
        olist.append(orig[:, i * binlen: (i + 1) * binlen])
        rlist.append(remix[:, i * binlen: (i + 1) * binlen])

    return np.array(ids), np.array(olist), np.array(rlist)

# construct a dataset from arrays and save to path
def construct_save_ds(ids, origs, remixes, path):
    ds = tf.data.Dataset.from_tensor_slices((ids, origs, remixes))
    print(f"Dataset constructed, saving now...")
    ds.save(path)
    print(f"Data saved to {path}")

#
def save_npys_to_Dataset(origpath, savepath, min=0, max=0):
    ids, train_o, train_r = load_spec_array_splitseconds(origpath, min=min, max=max)
    print("IDs", ids.shape)
    print("Set Orig", train_o.shape[0], train_o[0].shape[0], train_o[1].shape[0])
    print("Set Remix", train_r.shape[0], train_r[0].shape[0], train_r[1].shape[0])
    construct_save_ds(ids, train_o, train_r, savepath)


# load saved datasets that may be saved in parts
def load_dsparts(name: str):
    ds_names = glob(f"./{name}_part*")

    if len(ds_names) == 0:
        ds_names = glob(f'./{name}')
        if len(ds_names) == 0:
            raise NotADirectoryError

    print(f"Loading Dataset Part 1")
    ds = tf.data.Dataset.load(ds_names[0])

    for i in range(1, len(ds_names)):
        print(f"Loading Dataset Part {i + 1}")
        temp = tf.data.Dataset.load(ds_names[i])
        ds.concatenate(temp)

    print(f"Dataset {name} loaded from {len(ds_names)} parts")
    return ds

def main():
    ids, olist, rlist = load_wv_array_splitseconds_checked("dsSimple_train_o")
    construct_save_ds(ids, olist, rlist, "dsSimple_train")
    ids, val_o, val_r = load_wv_array_splitseconds_checked("dsSimple_val_o")
    construct_save_ds(ids, val_o, val_r, "dsSimple_val")
    exit(0)

if __name__ == "__main__":
    main()