import scipy.io.wavfile
import numpy as np
import pydub
import os
from pydub import AudioSegment

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)
dir_raw_data="./data/Sound_Files/"
dir_normalized_data="./data/Normalized_Sound_Files/"
os.makedirs(os.path.dirname(dir_normalized_data), exist_ok=True)
for (root_d, dirs_d, files_d) in os.walk(dir_raw_data):
    for d in files_d:
        a, b = d.split(".")
        sound = AudioSegment.from_file(root_d+d)
        normalized_sound = match_target_amplitude(sound, -20.0)
        normalized_sound.export(dir_normalized_data+"Normalized_"+a+".wav", format="wav")
