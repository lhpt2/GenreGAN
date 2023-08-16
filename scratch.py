import datetime
from glob import glob
from os.path import basename
import os

import numpy
import numpy as np
import scipy.signal
import soundfile
import tensorflow as tf
from numpy import float32
import librosa
import matplotlib.pyplot as plt

from preprocessing import convert_audio_array_to_spec_array, load_audio_array
from preprocessing import db_spec_to_wave, wave_to_db_spec
from testing_network import  use_generator
from constants import GL_SR
import scipy as sc

"""
Dateien mit zugehoerigem Dateinamen einlesen
- Datenstruktur erstellen, die Dateinamen an Daten koppelt
"""

# def tospec(data: np.ndarray):
#
#     S = power_spec_to_db_spec(data[0])
#     S = np.array(S, dtype=np.float32)
#     return S

def convert_folder(srcpath, destpath):
   ls = glob(f'{srcpath}/*.wav')

   if not os.path.exists(destpath):
      os.mkdir(destpath)

   for file in ls:
      filename = os.path.splitext(basename(file))[0]
      print(filename)
      wv, sr = tf.audio.decode_wav(tf.io.read_file(file))
      wv = np.array(wv, dtype=np.float32)
      spec = wave_to_db_spec(wv)
      np.save(f"{destpath}/{filename}.npy", spec, allow_pickle=False)

def convert_file(srcfile, destpath):
   filename = os.path.splitext(basename(srcfile))[0]
   print(filename)
   wv, sr = tf.audio.decode_wav(tf.io.read_file(srcfile))
   wv = np.array(wv, dtype=np.float32)
   spec = wave_to_db_spec(wv)
   np.save(f"{destpath}/{filename}.npy", spec, allow_pickle=False)

   # adata = []
   # for i in range(len(ls)):
   #    x, sr = tf.audio.decode_wav(tf.io.read_file(ls[i]), 1)
   #    x = np.array(x, dtype=np.float32)
   #    adata.append(x)
   #
   # if len(adata) > 1:
   #    return np.array(adata, dtype=object)
   # else:
   #    return np.array(adata, dtype=np.float32)


#print(f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}")

#convert_file(f"{path}/traindata_r/125_r_Maverick_City_Jireh.wav", "spec_train_r")
#convert_file(f"{path}/traindata_r/126_r_Saint_JHN_Roses.wav", "spec_train_r")

#wv, sr = librosa.load("jazz2/jazz.00000.wav", sr=22050)  #Load waveform

#file="./testjazz1.npy"
#a = audio_array(path)
#print(a)

# b = tospec(a)
# c = numpy.squeeze(b[0])
# np.save(file, c, allow_pickle=False)

# def save_spec_to_file(path, spec):
#    filename = "00"
#    spec = np.squeeze(spec)
#    np.save(f'{path}/{filename}.npy', spec, allow_pickle=False)

#spec0 = numpy.load(file)

#spec1 = wave_to_db_spec(wv)                                                    #Waveform to Spectrogram

#print(spec0)
#print(spec1)

#print(spec1.shape)

#wv = db_spec_to_wave(spec1)
#wv0 = db_spec_to_wave(spec0)
#soundfile.write("testjazz1.wav", wv, sr)

#print(wv.shape)
#wv2 = db_spec_to_wave(speca)

#plt.figure(figsize=(50,1))                                          #Show Spectrogram
#plt.imshow(np.flip(speca, axis=0), cmap=None)
#plt.axis('off')
#plt.show()

#abwv = use_generator(speca, name='FILENAME1', path='../content/')           #Convert and save wav
#path="../backup_datensatz_22khz/mono/"

srcpath="../../datensatz_22khz/mono"
destpath="../../datensatz_22khz/specs"

#convert_folder(f"{srcpath}/traindata_o", f"{destpath}/spectrain_o")
#convert_folder(f"{srcpath}/traindata_r", f"{destpath}/spectrain_r")
convert_folder(f"{srcpath}/valdata_o", f"{destpath}/specval_o")
# convert_folder(f"{srcpath}/valdata_r", f"{destpath}/specval_r")
# convert_folder(f"{srcpath}/testdata_o", f"{destpath}/spectest_o")
# convert_folder(f"{srcpath}/testdata_r", f"{destpath}/spectest_r")

