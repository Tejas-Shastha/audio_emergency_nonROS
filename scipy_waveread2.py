# Source : https://stackoverflow.com/questions/54592194/fft-of-data-received-from-pyaudio-gives-wrong-frequency
import sys
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

# Read the file (rate and data):
rate, data = wavfile.read(sys.argv[1]) # See source

# Compute PSD:
f, Pxx_den = signal.periodogram(data, rate)

# Display PSD:
fig, axe = plt.subplots()
print(np.shape(f), np.shape(Pxx_den))
axe.semilogy(f, Pxx_den)
axe.set_xlim([0,500])
axe.set_ylim([1e-8, 1e10])
axe.set_xlabel(r'Frequency, $\nu$ $[\mathrm{Hz}]$')
axe.set_ylabel(r'PSD, $P$ $[V^2\mathrm{Hz}^{-1}]$')
axe.set_title('Periodogram')
axe.grid(which='both')