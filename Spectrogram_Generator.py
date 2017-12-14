from scipy import signal
import numpy as np
import pickle
import os
import Load_Audio_Data

#compute spectomgram
def _compute_spectrungram(data,fs):
    sg = []
    for i in range(data.shape[0]):
        f, t, Sxx = signal.spectrogram(data[i], fs)
        sg.append(Sxx)
    return np.array(sg)

#precompute spectrumgram for the training data set
#the demision of returned matrix is (num_of_chuncks,height, width,channels)
def convert_to_spectrogram(dir_source,dir_out,fs,output_channel,chunk_time=0.1, down_sample=False,down_sample_rate=4):
    os.makedirs(os.path.dirname(dir_out), exist_ok=True)
    for (root_d, dirs_d, files_d) in os.walk(dir_source):
        for d in files_d:
            a,b=d.split(".")
            (data_1, data_2) = Load_Audio_Data.get_data(root_d + d,chunk_time=chunk_time, down_sample=down_sample,down_sample_rate=down_sample_rate)
            spectrogram_1 = _compute_spectrungram(data_1, fs)
            spectrogram_2 = _compute_spectrungram(data_2, fs)
            if output_channel==1:
                pickle.dump(np.concatenate((spectrogram_1,spectrogram_2),axis=0), open(dir_out + a + ".p", "wb"))
            else:
                spectrogram_mix=np.array((spectrogram_1,spectrogram_2))
                #now the dimision of spectrogram_mix is
                #(channels, num_of_chuncks,height,width)
                spectrogram_mix=np.swapaxes(spectrogram_mix,0,3)
                # now the dimision of spectrogram_mix is
                # (width, num_of_chuncks,height,channels)
                spectrogram_mix = np.swapaxes(spectrogram_mix, 0, 2)
                # now the dimision of spectrogram_mix is
                # (height, num_of_chuncks,width,channels)
                spectrogram_mix = np.swapaxes(spectrogram_mix, 0, 1)
                # now the dimision of spectrogram_mix is
                # (num_of_chuncks,height, width,channels), this is desired
                pickle.dump(spectrogram_mix, open(dir_out+a+".p", "wb"))

