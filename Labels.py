import pandas as pd
import numpy as np

def get_labels(filename,chunk_time=0.1):

    # Step 1: Read CSV file
    # Read file in utf-8 format
    data = pd.read_csv(filename, encoding ='utf-8')

    dataHealthy1 = data[data['tier'].str.contains(u"1") & data['text'].str.contains(u'S')]
    dataHealthy2 = data[data['tier'].str.contains(u"2") & data['text'].str.contains(u'S')]

    healthy1Time = dataHealthy1[['tmin','tmax']]
    healthy2Time = dataHealthy2[['tmin','tmax']]

    # We slice the float to and convert it to 0.1sec label
    healthy1Time = healthy1Time.astype(float) / chunk_time
    healthy1Time = healthy1Time.astype(int)

    healthy2Time = healthy2Time / chunk_time
    healthy2Time = healthy2Time.astype(int)

    maxFrame = (data['tmax'].max()/chunk_time).astype(int)

    label_1 = np.zeros((maxFrame), dtype=np.int)
    label_2 = np.zeros((maxFrame), dtype=np.int)

    # Assign values to each frame for each time duration of speech
    for index, row in healthy1Time.iterrows():
        label_1[row['tmin']: row['tmax']] = 1

    for index, row in healthy2Time.iterrows():
        label_2[row['tmin']: row['tmax']] = 1

    #return the labels for each channel
    return (label_1.reshape(-1,1),label_2.reshape(-1,1))

