from scipy import signal
from scipy.stats import mode
from numpy import linalg as LA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba
import time
from scipy.sparse.linalg import eigs
from numpy.linalg import inv
import sys

fs = 1000
Ns = 1000000000
# X_julia_$(fs)_$(Ns)_class1_part1.npz
X_class1_part1 = np.load(r'E:\phm08-project\X_julia_%i_%i_class1_part1.npz'%(fs, Ns))
X_class1_part2 = np.load(r'E:\phm08-project\X_julia_%i_%i_class1_part2.npz'%(fs, Ns))
X_class1_part3 = np.load(r'E:\phm08-project\X_julia_%i_%i_class1_part3.npz'%(fs, Ns))
X_class1_part4 = np.load(r'E:\phm08-project\X_julia_%i_%i_class1_part4.npz'%(fs, Ns))

X_class1 = np.concatenate((X_class1_part1, X_class1_part2, X_class1_part3, X_class1_part4), axis=None)
X_class1 = X_class1[1:]

segment_size = np.int32(0.5*Ns)
freqs, psd = signal.welch(X_class1[0:int(Ns)],fs = fs,  nperseg=segment_size/10000)
#psd = psd / psd.sum()
freq_lim = 40
x = freqs[freqs <= freq_lim]
x_len = x.size
y = psd[:x_len]
#y = y / y.sum()
fig, ax = plt.subplots(1, 1, figsize=(7.2, 6))
plt.semilogy(x, y, color='k')
plt.vlines([0.2, 1, 2.4, 3.6, 7,9,14,16,19,21], 0, 1, transform=ax.get_xaxis_transform(), linewidth=0.3, color="k")

t1 = ("Thermal hydraulic fluctuations")
plt.text(0.9, 9e-8, t1, ha='left', rotation=90, wrap=True, fontsize=8)

t2 = ("First beam mode of the fuel \nassembly")
plt.text(4.3, 9e-8, t2, ha='left', rotation=90, wrap=True, fontsize=8)

t3 = ("Core barrel free swinging peak")
plt.text(8.4, 9e-8, t3, ha='left', rotation=90, wrap=True, fontsize=8)

t4 = ("Vessel swinging")
plt.text(15.5, 9e-8, t4, ha='left', rotation=90, wrap=True, fontsize=8)

t5 = ("Shell mode n = 2 \n of the core barrel")
plt.text(20.7, 9e-8, t5, ha='left', rotation=90, wrap=True, fontsize=8)


plt.title('In the absence of anomaly: power spectral density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.xlim([0,40])
plt.ylim([8e-8,1e-5])
plt.tight_layout()
plt.savefig('normal_PSD.png', dpi= 300)
