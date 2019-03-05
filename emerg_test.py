"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).
"""

import pyaudio
import wave
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


if os.path.exists("results.csv"):
    os.remove("result.csv")

CHUNK = 1024
WIDTH = 2
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 20


class NullDevice():
    def write(self, s):
        pass

if len(sys.argv) < 2:
    print("Live test, need subject name.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)

SUBJECT_FREQ_LIST = []
subject = sys.argv[1]
freq_file = "output/"+ subject + "_freqs.csv"
print(freq_file)
with open(freq_file) as reader:
    csv_reader = csv.reader(reader, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in csv_reader:
        for freq in row:
            rounded = int(float(freq))
            if rounded not in SUBJECT_FREQ_LIST:
                SUBJECT_FREQ_LIST.append( rounded )

print(SUBJECT_FREQ_LIST)

def clip_signal(signal, clipping_thresh=1000, clipped_value=215):
    """
    Remove high frequency harmonics/noises   
    @params   
    signal - Input signal    
    clipping_thresh - Threshold frequency above which to clip   
    clipped_values - Value to which the signal is clipped   
    """
    while np.argmax(signal) >= clipping_thresh:
        signal[np.argmax(signal)] = 0
    return signal


def process_batch(batch, chunk_no, write_csv = False):
    index_factor = RATE / CHUNK
    data_fft = np.fft.fft(data_np)
    data_freq = np.abs(data_fft)/len(data_fft) # Dividing by length to normalize the amplitude as per https://www.mathworks.com/matlabcentral/answers/162846-amplitude-of-signal-after-fft-operation
    date_freq = clip_signal(data_freq,clipping_thresh=1000, clipped_value=215)

    avg_mag = np.average(data_freq)
    max_freq = index_factor* np.argmax(data_freq)
    stdev = np.std(data_freq, ddof=1)
    print("Chunk: {} max_freq: {} avg: {} stdev: {}".format(chunk_no, max_freq, avg_mag, stdev)  )
    if write_csv == True:
        with open("result.csv", mode='a') as writer_file:
            csv_writer = csv.writer(writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([chunk_no, max_freq, avg_mag, stdev])
    return max_freq, avg_mag, stdev

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

print("* recording")

i = 0
batch_sequence = np.array([])
total_sequence = np.array([])

max_frequency_list = np.array([])
average_mag_list = np.array([])
stdev_list = np.array([])

global_max_len = 0
global_avg = 0
global_std = 0


with open("result.csv", mode='a') as writer_file:
    csv_writer = csv.writer(writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["MAX_FREQ_SIZE","GLOBAL_AVG", "GLOBAL_STD"])
i = 0
while True:
    data = stream.read(CHUNK)
    data_unpacked = struct.unpack('{n}h'.format(n= len(data)/2 ), data) 
    data_np = np.array(data_unpacked)
    batch_sequence = np.concatenate((batch_sequence,data_np))   # Collect signals for short duration and process as batches
    total_sequence = np.concatenate((total_sequence,data_np))   # Collect total signal so far
    maximum = average = stdev = 0
    if i % 5 == 0:     #Approximately 100ms = 1 batch
        maximum, average, stdev = process_batch(batch_sequence, i, write_csv=False)   # Process the signals collected so far
        batch_sequence=np.array([])     # And clear away that buffer
        
        if maximum not in max_frequency_list:
            max_frequency_list = np.append(max_frequency_list, maximum)
        average_mag_list = np.append(average_mag_list,average)
        stdev_list = np.append(stdev_list, stdev)

    if i % 25 == 0 and i > 0:    # Approx 500ms, evaluate 5 batches
        global_max_len = len(max_frequency_list)
        global_avg= np.average(average_mag_list)
        global_std = np.std(stdev_list, ddof=1)

        print("Global max len: {} avg: {} std: {}".format(global_max_len, global_avg, global_std))
        print("-----")

        decision = ""
        for freq in SUBJECT_FREQ_LIST:
            if np.abs(freq-maximum) < 5 and global_std <= 5:
                decision = "hum"
                break            
        if global_std >= 5:
            decision = "talk"

        with open("result.csv", mode='a') as writer_file:
            csv_writer = csv.writer(writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([max_frequency_list, global_avg, global_std, decision ] )
            csv_writer.writerow(["---","---", "---"])

    if i % 50 == 0: # Approx 1s, purge lists
        print("-------------------PURGE-----------------------")
        max_frequency_list = np.array([])
        average_mag_list = np.array([])
        stdev_list = np.array([])

        global_max_len = 0
        global_avg = 0
        global_std = 0

        with open("result.csv", mode='a') as writer_file:
            csv_writer = csv.writer(writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["PURGE"] )
            csv_writer.writerow(["---","---", "---"])



    stream.write(data, CHUNK)
    i += 1

stream.stop_stream()
stream.close()

p.terminate()