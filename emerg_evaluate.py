"""PyAudio Example: Play a WAVE file."""

# Sample duration * sampling frequency = total number of samples = 220500
# Total number of samples / chunk size = number of chunks = 215.33
# Sample duration / number of chunks = chunk duration = 23.22 ms

import pyaudio
import wave
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
import csv

CHUNK = 1024



def clip_signal(signal, clipping_thresh=1000, clipped_value=215):
    """
    Remove high frequency harmonics/noises   
    @params   
    signal - Input signal    
    clipping_thresh - Threshold frequency above which to clip   
    clipped_values - Value to which the signal is clipped   
    """
    index_factor = rate / CHUNK
    while index_factor * np.argmax(signal) >= clipping_thresh:
        signal[np.argmax(signal)] = 0
    return signal


def process_batch(batch, chunk_no, write_csv = False):
    index_factor = rate / CHUNK
    data_fft = np.fft.fft(data_np)
    data_freq = np.abs(data_fft)/len(data_fft) # Dividing by length to normalize the amplitude as per https://www.mathworks.com/matlabcentral/answers/162846-amplitude-of-signal-after-fft-operation
    date_freq = clip_signal(data_freq,clipping_thresh=1000, clipped_value=215)

    avg_mag = np.average(data_freq)
    max_freq = index_factor* np.argmax(data_freq)
    stdev = np.std(data_freq, ddof=1)
    print("Chunk: {} max_freq: {} avg: {} stdev: {}".format(chunk_no, max_freq, avg_mag, stdev)  )
    if write_csv == True:
        with open("output.csv", mode='a') as writer_file:
            csv_writer = csv.writer(writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([chunk_no, max_freq, avg_mag, stdev])

    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # x = np.array(range(len(data_freq)))    
    # ax.plot( rate/CHUNK * x, data_freq)
    # ax.set_xscale('log')
    # plt.show()

    return max_freq, avg_mag, stdev

if len(sys.argv) < 2:
    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)

with open("output.csv", mode='a') as writer_file:
    csv_writer = csv.writer(writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow([sys.argv[1]])
    csv_writer.writerow(['CHUNK','MAX FREQ', 'AVG', 'STDEV'])

wf = wave.open(sys.argv[1], 'rb')

p = pyaudio.PyAudio()
rate = float(wf.getframerate())

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

data = wf.readframes(CHUNK)

i = 0
batch_sequence = np.array([])
total_sequence = np.array([])

max_frequency_list = np.array([])
average_mag_list = np.array([])
stdev_list = np.array([])

global_max_len = 0
global_avg = 0
global_std = 0


print("")
print("")
print("")
while data != '':
    data_unpacked = struct.unpack('{n}h'.format(n= len(data)/2 ), data) 
    data_np = np.array(data_unpacked)
    batch_sequence = np.concatenate((batch_sequence,data_np))   # Collect signals for short duration and process as batches
    total_sequence = np.concatenate((total_sequence,data_np))   # Collect total signal so far
    if i % 5 == 0:     #Approximately 100ms
        maximum, average, stdev = process_batch(batch_sequence, i, write_csv=False)   # Process the signals collected so far
        batch_sequence=np.array([])     # And clear away that buffer
        
        if maximum not in max_frequency_list:
            max_frequency_list = np.append(max_frequency_list, maximum)
        average_mag_list = np.append(average_mag_list,average)
        stdev_list = np.append(stdev_list, stdev)

    if i % 22 == 0:
        global_max_len = len(max_frequency_list)
        global_avg= np.average(average_mag_list)
        global_std = np.std(stdev_list, ddof=1)

        print("Global max len: {} avg: {} std: {}".format(global_max_len, global_avg, global_std))
        print("-----")

        with open("output.csv", mode='a') as writer_file:
            csv_writer = csv.writer(writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["MAX_FREQ_SIZE","GLOBAL_AVG", "GLOBAL_STD"])
            csv_writer.writerow([len(max_frequency_list), np.average(average_mag_list), np.std(stdev_list) ] )
            csv_writer.writerow(["---","---", "---"])


    stream.write(data)
    data = wf.readframes(CHUNK)
    i += 1

print("")
print("")
print("Global max feq list: {} size: {}".format(max_frequency_list, global_max_len))
print("Global average:{}".format( global_avg ))
print("Global stdev:{}".format( global_std ))

# with open("output.csv", mode='a') as writer_file:
#     csv_writer = csv.writer(writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     csv_writer.writerow(["MAX_FREQ_SIZE","GLOBAL_AVG", "GLOBAL_STD"])
#     csv_writer.writerow([len(max_frequency_list), np.average(average_mag_list), np.std(stdev_list) ] )
#     csv_writer.writerow(["---","---", "---"])

# data_fft = np.fft.fft(total_sequence)
# data_freq = np.abs(data_fft)
# data_freq = data_freq/len(data_fft) # Dividing by length to normalize the amplitude as per https://www.mathworks.com/matlabcentral/answers/162846-amplitude-of-signal-after-fft-operation
# date_freq = clip_signal(data_freq,clipping_thresh=len(data_fft)*0.8, clipped_value=215)
# average_mag = np.average(data_freq)

# index_factor = rate / (CHUNK*i)

# max_freq = index_factor * np.argmax(data_freq)
# print("Average mag norm : {}".format(average_mag))
# print("Max freq : {}".format(max_freq))

# x = np.array(range(len(data_freq)))
# x = index_factor * x
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(x, data_freq)
# ax.set_xscale('log')
# plt.show()

stream.stop_stream()
stream.close()

p.terminate()