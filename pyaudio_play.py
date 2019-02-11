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

CHUNK = 1024



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


def process_batch(batch):
    pass


if len(sys.argv) < 2:
    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)

wf = wave.open(sys.argv[1], 'rb')

p = pyaudio.PyAudio()
rate = float(wf.getframerate())

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

data = wf.readframes(CHUNK)

i = 0
max_freq_list = []
time_signal = np.array([])
while data != '':
    i += 1
    data_unpacked = struct.unpack('{n}h'.format(n= len(data)/2 ), data) 
    data_np = np.array(data_unpacked)
    time_signal = np.concatenate((time_signal,data_np))  
    index_factor = rate / (CHUNK)
    data_fft = np.fft.fft(data_np)
    data_freq = np.abs(data_fft)/len(data_fft) # Dividing by length to normalize the amplitude as per https://www.mathworks.com/matlabcentral/answers/162846-amplitude-of-signal-after-fft-operation
    date_freq = clip_signal(data_freq,clipping_thresh=1000, clipped_value=215)
    max_freq = index_factor* np.argmax(data_freq)
    print("Chunk: {} max_freq: {}".format(i, rate/ CHUNK * np.argmax(data_freq)))

    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # x = np.array(range(len(data_freq)))    
    # ax.plot( rate/CHUNK * x, data_freq)
    # ax.set_xscale('log')
    # plt.show()

    max_freq_list.append(np.argmax(data_fft))
    stream.write(data)
    data = wf.readframes(CHUNK)


data_fft = np.fft.fft(time_signal)
data_freq = np.abs(data_fft)
data_freq = data_freq/len(data_freq) # Dividing by length to normalize the amplitude as per https://www.mathworks.com/matlabcentral/answers/162846-amplitude-of-signal-after-fft-operation
average_mag = np.average(data_freq)

index_factor = rate / (CHUNK*i)

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