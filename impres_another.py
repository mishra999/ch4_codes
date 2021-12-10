# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:12:50 2020

@author: mmishra
"""


#do correlation in time , take average in time, then dive in frequency domain
M1 = 30000
from scipy import signal
corr1 = np.zeros((M1,1999))
corr2 = np.zeros((M1,1999))
corr3 = np.zeros((M1,1999)) #autocorrelation between output
h1 = np.zeros(1999,dtype=np.complex)
h2 = np.zeros(1999,dtype=np.complex)
h = np.zeros((1999),dtype=np.complex)
for i in range(M1):
    corr1[i] = signal.correlate(res_50[i,0]/np.max(res_50[i,0,100:800]), res_50[i,1]/np.max(res_50[i,1,100:800]), mode='full') / len(res_50[i,1])
#    corr1[i] = signal.correlate(np.lib.pad(a0[i], (500,500), 'constant', constant_values=(0., 0.)), np.lib.pad(a2[i], (500,500), 'constant', constant_values=(0., 0.)), mode='same') #/ len(a0[0])
#    corr1[i] = np.lib.pad(cor1, (500,500), 'constant', constant_values=(0., 0.))
    corr2[i] = signal.correlate(res_50[i,1]/np.max(res_50[i,1,100:800]), res_50[i,1]/np.max(res_50[i,1,100:800]), mode='full') / len(res_50[i,1])
#    corr2[i] = signal.correlate(np.lib.pad(a2[i], (500,500), 'constant', constant_values=(0., 0.)), np.lib.pad(a2[i], (500,500), 'constant', constant_values=(0., 0.)), mode='same')
#    corr2[i] = np.lib.pad(cor2, (500,500), 'constant', constant_values=(0., 0.))
#    corr3[i] = signal.correlate(a0[i], a0[i], mode='same') #/ len(a0[0])
#    plt.plot(corr2)
#    plt.xlabel('sample number',fontsize=16)
#    plt.ylabel('autocorrelation',fontsize=16)
##    h1[i] = np.fft.fft(corr1[i])
#    plt.plot(f0[0:50],np.abs(h1)[0:50])
##    h2[i] = np.fft.fft(corr2[i])
#    plt.plot(f0[0:1000],np.abs(h2)[0:1000])
#    plt.xlabel('frequency (MHz)', fontsize=16)
#    plt.ylabel('power\n(arb. units)', fontsize=16)

##    h[i] = h1[i]/h2[i]
#    plt.plot(f0[0:50],np.abs(h)[0:50])
#    plt.xlabel('frequency (MHz)', fontsize=16)
#    plt.ylabel('power\n(arb. units)', fontsize=16)
#corr11 = np.zeros((2000))
#corr22 = np.zeros(2000)
corrr1 =  np.zeros(1999)
corrr2 =  np.zeros(1999)
#corrr3 =  np.zeros(2000)
for i in range(1999):
    for j in range(M1):
        corrr1[i] = corrr1[i] + corr1[j,i]
        corrr2[i] = corrr2[i] + corr2[j,i]
#        (corrr1[i]) = Decimal(corrr1[i]) + Decimal(corr1[j,i]/M1)
#        (corrr2[i]) = Decimal(corrr2[i]) + Decimal(corr2[j,i]/M1)

#        corrr3[i] = corrr3[i] + corr3[j,i]
#        corr11[i] = corr11[i] + corr1[j,i]
#        corr22[i] = corr22[i] + corr2[j,i]
#corr11 = corr11/10000
#corr22 = corr22/10000
corrr1 = corrr1/M1
#corrr11 = np.lib.pad(corrr1, (0,2000), 'constant', constant_values=(0., 0.))
corrr2 = corrr2/M1
#corrr22 = np.lib.pad(corrr2, (0,2000), 'constant', constant_values=(0., 0.))
#for i in range(2000):
#    h1[i] = np.fft.fft(Decimal(corrr1[i]))
#    h2[i] = np.fft.fft(Decimal(corrr2[i]))
h1 = np.fft.fft(corrr1)
h2 = np.fft.fft(corrr2)
h65 = h1 / h2


N1 = 1999
fs = 1000
f0 = np.arange(N1)
f0 = (fs * 1. /N1) * f0
plt.plot(f0[0:1000],np.abs(h65)[0:1000])

plt.plot(np.fft.ifft(h65))
h_data = np.fft.ifft(h65)[0:1000]
plt.plot(corrr1)
#to get autocorellation for output y
corrr3 = corrr3/M1
corrr33 = np.lib.pad(corrr3, (0,500), 'constant', constant_values=(0., 0.))
# if no zero padding
#h1 = np.fft.fft(corrr1)
#h2 = np.fft.fft(corrr2)
#h = h1 / h2
N1 = 1000
fs = 1000
f0 = np.arange(N1)
f0 = (fs * 1. /N1) * f0
plt.figure()
plt.plot(f0[0:500],np.abs(np.fft.fft(h_data[0:1000]))[0:500])
#plt.plot(f0[55:75],np.abs(h65[55:75]))
plt.xlabel('frequency (MHz)')
plt.ylabel('amplitude (arb. units)')
#plt.ylim(0,2.5)
plt.tight_layout()