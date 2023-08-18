from dataset_processing import load_dsparts, db_spec_to_wave
import pyaudio
import numpy as np

def playaudio(spec: np.ndarray):
    wv = db_spec_to_wave(spec)
    del spec
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                   rate=22050,
                    frames_per_buffer=1024,
                    output=True,
                    output_device_index=1
                    )
    stream.write(wv.astype(np.float32).tobytes())
    stream.close()