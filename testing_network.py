"""
Functions to test a trained network
"""
import soundfile as sf
import os, datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.optimizers import Adam

from tools.architecture_v2 import build, load
from MAINCONFIG import GL_SR, GL_SHAPE, log
from tools.dataset_processing import db_spec_to_wave

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

def save_spec_to_wv(spec, filepath='./test.wav'):
   wv = db_spec_to_wave(spec)
   sf.write(filepath, wv, GL_SR)

""" Converting from source Spectrogram to target Spectrogram """
def use_generator(spec, gen, geninfo, path='./', name="sample",  show=False):
   orig_spec = chopspec(spec)

   ver = geninfo["version"]
   testnr = geninfo["test"]
   ep = geninfo["epoch"]

   print('Generating...')
   gen_specarr = gen(orig_spec, training=False)

   print('Assembling and Converting...')
   final_genspec = specass(gen_specarr,spec)

   print('Saving...')
   save_spec_to_wv(final_genspec, filepath=path + f'/t{testnr}_e{ep}_v{ver}_gen_{name}.wav')
   print('Saved WAV!')

   print('Saving original...')
   save_spec_to_wv(spec, filepath=path + f'/t{testnr}_e{ep}_v{ver}_{name}.wav')
   print('Saved WAV!')
   #IPython.display.display(IPython.display.Audio(np.squeeze(abwv), rate=sr))
   #IPython.display.display(IPython.display.Audio(np.squeeze(awv), rate=sr))
   if show:
      fig, axs = plt.subplots(ncols=2)
      axs[0].imshow(np.flip(spec, -2), cmap=None)
      axs[0].axis('off')
      axs[0].set_title('Source')
      axs[1].imshow(np.flip(final_genspec, -2), cmap=None)
      axs[1].axis('off')
      axs[1].set_title('Generated')
      plt.show()
   return final_genspec

""" ################# internal functions (not for direct use) ###################"""

""" Generate a random batch to display current training results """
def testgena(aspec):

   sw = True

   while sw:
      a = aspec

      if (a.shape[1] // GL_SHAPE) != 1:
         sw=False

   dsa = []
   if a.shape[1] // GL_SHAPE > 6:
      num=6
   else:
      num= a.shape[1] // GL_SHAPE

   rn = np.random.randint(a.shape[1] - (num * GL_SHAPE))

   for i in range(num):
      im = a[:,rn+(i * GL_SHAPE):rn + (i * GL_SHAPE) + GL_SHAPE]
      im = np.reshape(im, (im.shape[0], im.shape[1], 1))
      dsa.append(im)

   return np.array(dsa, dtype=np.float32)

def cut4gen(spec):
   return tf.reshape(spec, [3, 192, GL_SHAPE, 1])

def uncut4gen(spec):
   return tf.squeeze(tf.reshape(spec, [1, 192, 3 * GL_SHAPE, 1]))

""" Show results mid-training """
def save_test_image_full(path, gen, aspec):
   #a = testgena(aspec)

   # get right sample from dataset and add alibi dim for generator
   id = int(aspec[0][0][0])
   aspec = aspec[1][0]
   aspec = np.expand_dims(aspec, -1)

   a = cut4gen(aspec)
   ab = gen(a, training=False)

   ab = uncut4gen(ab)
   a = uncut4gen(a)

   abwv = db_spec_to_wave(ab)
   awv = db_spec_to_wave(a)

   save_spec_img(a, f'Original #{id}', filename=path + f'/{id}_orig.png')
   save_spec_img(ab, f'Generated #{id}', filename=path + f'/{id}_generated.png')
   sf.write(path + f'/{id}_orig.wav', awv, GL_SR)
   sf.write(path + f'/{id}_generated.wav', abwv, GL_SR)
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

def save_spec_img(x, title, filename='Testfig.png'):
   fig, axs = plt.subplots()
   axs.imshow(np.flip(x, -2), cmap=None)
   axs.axis('off')
   axs.set_title(title)
   # fig, axs = plt.subplots(ncols=2)
   # axs[0].imshow(np.flip(a, -2), cmap=None)
   # axs[0].axis('off')
   # axs[0].set_title('Source')
   # axs[1].imshow(np.flip(ab, -2), cmap=None)
   # axs[1].axis('off')
   # axs[1].set_title('Generated')
   fig.savefig(filename)
   #plt.show(block=True)

""" Save in training loop """
def save_end(epoch, gloss, closs, mloss, gen, critic, siam, aspec, n_save=3, save_path='./'):                 #use custom save_path (i.e. Drive '../content/drive/My Drive/')
      log(f'Saving epoch {epoch}...')
      #path = f'{save_path}/MELGANVC-{str(gloss)[:9]}-{str(closs)[:9]}-{str(mloss)[:9]}'
      path = f'{save_path}/{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}_{str(epoch)}'
      os.mkdir(path)
      gen.save_weights(path+'/gen.h5')
      critic.save_weights(path+'/critic.h5')
      siam.save_weights(path+'/siam.h5')
      save_test_image_full(path, gen, aspec)

""" Assembling generated Spectrogram chunks into final Spectrogram """
def specass(a, spec):
   first_handled = False
   con = np.array([])
   nim = a.shape[0]
   for i in range(nim-1):
      im = a[i]
      im = np.squeeze(im)
      if not first_handled:
         con=im
         first_handled = True
      else:
         con = np.concatenate((con,im), axis=1)
   diff = spec.shape[1] - (nim * GL_SHAPE)
   a = np.squeeze(a)
   con = np.concatenate((con,a[-1,:,-diff:]), axis=1)
   return np.squeeze(con)

""" Splitting input spectrogram into different chunks to feed to the generator """
def chopspec(spec):
   dsa=[]
   for i in range(spec.shape[1] // GL_SHAPE):
      im = spec[:, i * GL_SHAPE:i * GL_SHAPE + GL_SHAPE]
      im = np.reshape(im, (im.shape[0], im.shape[1], 1))
      dsa.append(im)
   imlast = spec[:, -GL_SHAPE:]
   imlast = np.reshape(imlast, (imlast.shape[0], imlast.shape[1], 1))
   dsa.append(imlast)
   return np.array(dsa, dtype=np.float32)


if __name__ == '__main__':
   #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
   #GL_SHAPE = 24
   geninfo = { "test": 7,
               "version": "3.0",
               "epoch": 500,
            }

   LOAD = "../Ergebnisse/Versuch06_3_0_LossPaper_3_10_10_0.7/2023-08-19-23-23_500_1767_0.01"
   #starttime=0
   #snippetlen=10
   gen, _, _, _ = get_networks(GL_SHAPE, load_model=True, path=LOAD)
   #spec = np.load('../spec_val_o/155_o_Karol_G_Ocean.npy')
   #start: int = secs_to_bins(starttime)
   #end: int = secs_to_bins(starttime + snippetlen)
   #spec = spec[:, start:end]
   spec = tf.random.uniform(shape=[192, 576])
   use_generator(spec, gen, geninfo)