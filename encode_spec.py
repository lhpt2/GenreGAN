import sys
import numpy
import os
import numpy as np

import soundfile

from preprocessing import db_spec_to_wave, wave_to_db_spec, load_single_audio

if len(sys.argv) < 3:
    raise Exception("Give filename and samplerate as arguments")

name = sys.argv[1]
sr = int(sys.argv[2])

wv = load_single_audio(name)
spec = wave_to_db_spec(wv)

filename = os.path.splitext(os.path.basename(sys.argv[1]))[0]
#soundfile.write(f"{filename}.wav", wv, sr)
np.save(f"{filename}.npy", spec)
print(f"Spectrogram written to {filename}.npy")


