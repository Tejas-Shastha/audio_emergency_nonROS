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

CHUNK = 1024
WIDTH = 2
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 20


class NullDevice():
    def write(self, s):
        pass

os.remove("result.csv")

def clip_signal(signal, rate, batch_size, clipping_thresh=1000, clipped_value=215):
    """
    Remove high frequency harmonics/noises   
    @params   
    signal - Input signal    
    clipping_thresh - Threshold frequency above which to clip   
    clipped_values - Value to which the signal is clipped   
    """
    index_factor = (rate / CHUNK) / batch_size
    while index_factor * np.argmax(signal) >= clipping_thresh:
        signal[np.argmax(signal)] = 0
    return signal

def process_batch(batch, batch_size, chunk_no, write_csv = False):
    index_factor = (RATE / CHUNK) / batch_size
    data_fft = np.fft.fft(batch)
    data_freq = np.abs(data_fft)/len(data_fft) # Dividing by length to normalize the amplitude as per https://www.mathworks.com/matlabcentral/answers/162846-amplitude-of-signal-after-fft-operation
    date_freq = clip_signal(data_freq, RATE, batch_size, clipping_thresh=1000, clipped_value=215)

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
# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
while True:
    data = stream.read(CHUNK)
    data_unpacked = struct.unpack('{n}h'.format(n= len(data)/2 ), data) 
    data_np = np.array(data_unpacked)
    batch_sequence = np.concatenate((batch_sequence,data_np))   # Collect signals for short duration and process as batches
    total_sequence = np.concatenate((total_sequence,data_np))   # Collect total signal so far
    maximum = average = stdev = 0
    batch_size = 5

    if i % batch_size == 0:     #Approximately 100ms = 1 batch
        maximum, average, stdev = process_batch(batch_sequence, batch_size, i, write_csv=False)   # Process the signals collected so far
        batch_sequence=np.array([])     # And clear away that buffer
        
        if maximum not in max_frequency_list and i > batch_size-1:
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
        for freq in [137, 275, 267]:
            if np.abs(freq-maximum) < 5 and global_std <= 5:
                decision = "hum"
                break            
        if global_std >= 5:
            decision = "talk"

        with open("result.csv", mode='a') as writer_file:
            csv_writer = csv.writer(writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # csv_writer.writerow(["MAX_FREQ_SIZE","GLOBAL_AVG", "GLOBAL_STD"])
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
            # csv_writer.writerow(["MAX_FREQ_SIZE","GLOBAL_AVG", "GLOBAL_STD"])
            csv_writer.writerow(["PURGE"] )
            csv_writer.writerow(["---","---", "---"])



    stream.write(data, CHUNK)
    i += 1

print("* done")


data_fft = np.fft.fft(total_sequence)
data_freq = np.abs(data_fft)
data_freq = data_freq/len(data_fft) # Dividing by length to normalize the amplitude as per https://www.mathworks.com/matlabcentral/answers/162846-amplitude-of-signal-after-fft-operation
average_mag = np.average(data_freq)

index_factor = RATE / (CHUNK*i)

max_freq = index_factor * np.argmax(data_freq)
print("Average mag norm : {}".format(average_mag))
print("Max freq : {}".format(max_freq))

x = np.array(range(len(data_freq)))
x = index_factor * x
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x, data_freq)
ax.set_xscale('log')
plt.show()


stream.stop_stream()
stream.close()

p.terminate()