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

def evaluate_wav(wavefile):
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


    def process_batch(batch, batch_size, chunk_no, rate, write_csv = False):
        index_factor = (rate / CHUNK) / batch_size
        data_fft = np.fft.fft(batch)
        data_freq = np.abs(data_fft)/len(data_fft) # Dividing by length to normalize the amplitude as per https://www.mathworks.com/matlabcentral/answers/162846-amplitude-of-signal-after-fft-operation
        date_freq = clip_signal(data_freq,rate,batch_size, clipping_thresh=1000, clipped_value=215)

        avg_mag = np.average(data_freq)
        max_freq = index_factor * np.argmax(data_freq)
        max_amp = data_freq[np.argmax(data_freq)]
        stdev = np.std(data_freq, ddof=1)
        print("Chunk: {} max_freq: {} max_amp: {} stdev: {}".format(chunk_no, max_freq, max_amp, stdev)  )
        if write_csv == True:
            with open("report.csv", mode='a') as writer_file:
                csv_writer = csv.writer(writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([chunk_no, max_freq, avg_mag, stdev])
        return max_freq, max_amp, avg_mag, stdev

    wf = wave.open(wavefile, 'rb')

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

        batch_size = 5
        if i % batch_size == 0:     #Approximately 100ms
            maximum, amp_max, average, stdev = process_batch(batch_sequence, batch_size, i,rate, write_csv=False)   # Process the signals collected so far
            batch_sequence=np.array([])     # And clear away that buffer

            if maximum not in max_frequency_list and i > batch_size-1:
                max_frequency_list = np.append(max_frequency_list, maximum)
            average_mag_list = np.append(average_mag_list,average)
            stdev_list = np.append(stdev_list, stdev)

        if i % 22 == 0:
            global_max_len = len(max_frequency_list)
            global_avg= np.average(average_mag_list)
            global_std = np.std(stdev_list, ddof=1)

            print("Global max len: {} avg: {} std: {}".format(global_max_len, global_avg, global_std))
            print("-----")

            # with open("output.csv", mode='a') as writer_file:
            #     csv_writer = csv.writer(writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     csv_writer.writerow(["MAX_FREQ_SIZE","GLOBAL_AVG", "GLOBAL_STD"])
            #     csv_writer.writerow([len(max_frequency_list), np.average(average_mag_list), np.std(stdev_list) ] )
            #     csv_writer.writerow(["---","---", "---"])


        stream.write(data)
        data = wf.readframes(CHUNK)
        i += 1

    print("")
    print("")
    print("Global max feq list: {} size: {}".format(max_frequency_list, global_max_len))
    print("Global average:{}".format( global_avg ))
    print("Global stdev:{}".format( global_std ))
    print("Robot noise freq list: [344.53125     86.1328125  258.3984375  215.33203125 129.19921875 172.265625    43.06640625 861.328125  ]")

    # with open("output.csv", mode='a') as writer_file:
    #     csv_writer = csv.writer(writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     csv_writer.writerow(["MAX_FREQ_SIZE","GLOBAL_AVG", "GLOBAL_STD"])
    #     csv_writer.writerow([len(max_frequency_list), np.average(average_mag_list), np.std(stdev_list) ] )
    #     csv_writer.writerow(["---","---", "---"])

    data_fft = np.fft.fft(total_sequence)
    data_freq = np.abs(data_fft)
    data_freq = data_freq/len(data_fft) # Dividing by length to normalize the amplitude as per https://www.mathworks.com/matlabcentral/answers/162846-amplitude-of-signal-after-fft-operation
    date_freq = clip_signal(data_freq, rate,1, clipping_thresh=len(data_fft)*0.8, clipped_value=215)
    average_mag = np.average(data_freq)

    index_factor = rate / (CHUNK*i)

    max_freq = index_factor * np.argmax(data_freq)
    max_amp = data_freq[np.argmax(data_freq)]
    print("Average mag norm : {}".format(average_mag))
    print("Max freq : {}".format(max_freq))
    print("Max freq's amp : {}".format(max_amp))

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
        sys.exit(-1)
    evaluate_wav(sys.argv[1])
