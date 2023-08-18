import sys
import numpy
import os

import soundfile

from dataset_processing import db_spec_to_wave

if len(sys.argv) < 3:
    raise Exception("Give filename and samplerate as arguments")

name = sys.argv[1]
sr = int(sys.argv[2])

spec = numpy.load(sys.argv[1])
spec = numpy.squeeze(spec)
wv = db_spec_to_wave(spec)
filename = os.path.splitext(os.path.basename(sys.argv[1]))[0]
soundfile.write(f"{filename}.wav", wv, sr)
print(f"Audio written to {filename}.wav")


