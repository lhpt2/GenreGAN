import math

import soundfile as sf
import os, datetime
import numpy as np
import matplotlib.pyplot as plt

from constants import sr, shape, log
from preprocessing import db_spec_to_wave, concat_specarray
from architecture import get_networks

""" Converting from source Spectrogram to target Spectrogram """
def use_generator(spec, gen, path='./', show=False):
   specarr = chopspec(spec)
   print(specarr.shape)
   a = specarr
   print('Generating...')
   ab = gen(a, training=False)
   print('Assembling and Converting...')
   ab = specass(ab,spec)
   print('Saving...')
   abwv = db_spec_to_wave(ab)
   sf.write(path+'/AB.wav', abwv, sr)
   #print('Assembling and Converting...')
   #a = specass(a,spec)
   #print('Saving...')
   #awv = db_spec_to_wave(a)
   #sf.write(pathfin+'/A.wav', awv, sr)
   print('Saved WAV!')
   #IPython.display.display(IPython.display.Audio(np.squeeze(abwv), rate=sr))
   #IPython.display.display(IPython.display.Audio(np.squeeze(awv), rate=sr))
   if show:
      fig, axs = plt.subplots(ncols=2)
      axs[0].imshow(np.flip(a, -2), cmap=None)
      axs[0].axis('off')
      axs[0].set_title('Source')
      axs[1].imshow(np.flip(ab, -2), cmap=None)
      axs[1].axis('off')
      axs[1].set_title('Generated')
      plt.show()
   return abwv

""" ################# internal functions (not for direct use) ###################"""

""" Generate a random batch to display current training results """
def testgena(aspec):

   sw = True

   while sw:
      a = np.random.choice(aspec)

      if (a.shape[1]//shape) != 1:
         sw=False

   dsa = []
   if a.shape[1]//shape>6:
      num=6
   else:
      num=a.shape[1]//shape

   rn = np.random.randint(a.shape[1]-(num*shape))

   for i in range(num):
      im = a[:,rn+(i*shape):rn+(i*shape)+shape]
      im = np.reshape(im, (im.shape[0],im.shape[1],1))
      dsa.append(im)

   return np.array(dsa, dtype=np.float32)

""" Show results mid-training """
def save_test_image_full(path, gen, aspec):
   a = testgena(aspec)
   print(a.shape)
   ab = gen(a, training=False)
   ab = concat_specarray(ab)
   a = concat_specarray(a)
   abwv = db_spec_to_wave(ab)
   awv = db_spec_to_wave(a)
   sf.write(path+'/orig.wav', awv, sr)
   sf.write(path+'/new_file.wav', abwv, sr)
   #IPython.display.display(IPython.display.Audio(np.squeeze(abwv), rate=sr))
   #IPython.display.display(IPython.display.Audio(np.squeeze(awv), rate=sr))
   #fig, axs = plt.subplots(ncols=2)
   #axs[0].imshow(np.flip(a, -2), cmap=None)
   #axs[0].axis('off')
   #axs[0].set_title('Source')
   #axs[1].imshow(np.flip(ab, -2), cmap=None)
   #axs[1].axis('off')
   #axs[1].set_title('Generated')
   #plt.show()

""" Save in training loop """
def save_end(epoch,gloss,closs,mloss, gen, critic, siam, aspec, n_save=3,save_path='./'):                 #use custom save_path (i.e. Drive '../content/drive/My Drive/')
   if epoch % n_save == 0:
      log(f'Saving epoch {epoch}...')
      #path = f'{save_path}/MELGANVC-{str(gloss)[:9]}-{str(closs)[:9]}-{str(mloss)[:9]}'
      path = f'{save_path}/{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}_{str(epoch)}_{str(gloss)[:9]}_{str(closs)[:9]}'
      os.mkdir(path)
      gen.save_weights(path+'/gen.h5')
      critic.save_weights(path+'/critic.h5')
      siam.save_weights(path+'/siam.h5')
      save_test_image_full(path, gen, aspec)

""" Assembling generated Spectrogram chunks into final Spectrogram """
def specass(a,spec):
   first_handled=False
   con = np.array([])
   nim = a.shape[0]
   for i in range(nim-1):
      im = a[i]
      im = np.squeeze(im)
      if not first_handled:
         con=im
         first_handled=True
      else:
         con = np.concatenate((con,im), axis=1)
   diff = spec.shape[1]-(nim*shape)
   a = np.squeeze(a)
   con = np.concatenate((con,a[-1,:,-diff:]), axis=1)
   return np.squeeze(con)

""" Splitting input spectrogram into different chunks to feed to the generator """
def chopspec(spec):
   dsa=[]
   for i in range(spec.shape[1]//shape):
      im = spec[:,i*shape:i*shape+shape]
      im = np.reshape(im, (im.shape[0],im.shape[1],1))
      dsa.append(im)
   imlast = spec[:,-shape:]
   imlast = np.reshape(imlast, (imlast.shape[0],imlast.shape[1],1))
   dsa.append(imlast)
   return np.array(dsa, dtype=np.float32)

def bins_to_secs(bins, sr=22050, hop=192):
   return bins * hop // sr

def secs_to_bins(secs, sr=22050, hop=192):
   return math.ceil(secs * sr / hop)

if __name__ == '__main__':
   gen, critic, siam, [opt_gen, opt_disc] = get_networks(shape, load_model=True, path='../Ergebnisse/Versuch02_1_0_Validierung/2023-08-08-01-59_499_-9.440582_0.6141752')
   spec = np.load('../spec_val_o/155_o_Karol_G_Ocean.npy')
   spec = spec[:, secs_to_bins(100):secs_to_bins(110)]
   use_generator(spec, gen)