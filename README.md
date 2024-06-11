# GenreGAN
GAN for Music Genre transfer based upon M. Pasinis Work "MelGan-VC" created in the context of a bachelor thesis.
This work was created for a bachelor thesis about Music Genre Transfer. It deals with a GAN working on mel-spectrograms
and is based on the following work: https://github.com/marcoppasini/MelGAN-VC, Paper from M. Pasini: https://arxiv.org/abs/1910.03713

# Main dependencies
The main dependencies are:

- ``tensorflow`` - main library
- ``torch`` - for some audio/spectrogram related stuff that did not work properly with tensorflow
- ``librosa``- for conversion to spectrogram data
- ``soundfile`` - for reading and writing soundfiles

There is a ``requirements.txt`` provided in the root directory of the project.

# SHORT Summary

The ``tools/`` directory and the ``testing.py`` contains all of the needed logic to setup, train and test the network.
Some code is not refactored for general use, there are still "hardcoded" parts like paths, so there have to be
adjustments made for custom use. I did not find the time to refactor.

- ``MAINCONFIG.py`` holds global constants used globally in all scripts.
- ``convert_dataset.py`` is an example script for how a dataset would be prepared for training.
- ``validate_ds.py`` is an example script that loads two datasets ``dstrain`` and ``dsval`` and extracts a sample from source and
target space from it, in order to write them to wavefiles. This was used for testing proper construction and integrity of the datasets.
- ``training.py`` includes functions used for training and the process of how to train the network (at the bottom ``__main__`` section of the file)
It shows how a folder of wave files is turned into a dataset ready to be trained.
- ``testing_network.py``is an example script that loads a trained network and feeds it a spectrogram for conversion.
The function ``use_generator`` passes the data through the loaded network and saves it as a wavefile.
- ``encode_spec.py`` and ``decode_spec.py`` are scripts that take either a path to a wavefile and a samplerate (22050 kHz recommended)
or a path to a spectogram saved as ``.npy`` spectrogram file and convert it to either one of the formats.
- ``plotloss.py`` gives information about loss development extracted from the csv file that losses are logged into. There might be some lines commented.

The main part of the code is inside of the ``tools/``  directory:

- ``architecture_v2.py`` contains all architecture related functions
- ``dataset_processing.py`` contains all code for converting between spectrogram and wave data
- ``losses.py`` contains the loss functions and related code
- ``specnorm.py`` contains code for spectral normalization
