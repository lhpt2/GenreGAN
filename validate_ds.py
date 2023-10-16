from MAINCONFIG import *
from tools.dataset_processing import load_dsparts
from tools.dataset_processing import db_spec_to_wave
import soundfile

dsval = load_dsparts("dsSimple_val")
dstrain = load_dsparts('dsSimple_train')

dsval = dsval.shuffle(10000).as_numpy_iterator()
dstrain = dstrain.shuffle(10000).as_numpy_iterator()

_, v_orig, v_remix = dsval.next()
_, t_orig, t_remix = dstrain.next()

v_orig = db_spec_to_wave(v_orig)
v_remix = db_spec_to_wave(v_remix)
t_orig = db_spec_to_wave(t_orig)
t_remix = db_spec_to_wave(t_remix)

soundfile.write(f"t_orig.wav", t_orig, GL_SR)
soundfile.write(f"v_orig.wav", v_orig, GL_SR)
soundfile.write(f"v_remix.wav", v_remix, GL_SR)
soundfile.write(f"t_remix.wav", t_remix, GL_SR)