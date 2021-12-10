# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:49:15 2020

@author: mmishra
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from matplotlib import rcParams
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11
rcParams['legend.fontsize'] = 11
rcParams['font.family'] = 'sans-serif'#sans-serif
#rcParams['font.sans-serif'] = ['Verdana']
#rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = False
rcParams['figure.figsize'] = 6,5#5, 3


# I make my own newfig and savefig functions
def newfig(width):
#    plt.clf()
    fig = plt.figure(figsize=figsize(width))
#    ax = fig.add_subplot(111)
    return fig

def savefig(filename):
    plt.savefig('{}.pdf'.format(filename), bbox_inches='tight',dpi=500)
    plt.savefig('{}.png'.format(filename), bbox_inches='tight',dpi=500)
#    plt.savefig('{}.svg'.format(filename), bbox_inches='tight')



ch1 = [
            (   'c1'        ,   np.dtype('S1') ),
            (   'c2'        ,   np.dtype('S3') )]


bh = [
            (   'c1'        ,   np.dtype('S2') ),
            (   'c2'        ,   np.ushort )]

tch = [
            (   'c1'        ,   np.dtype('S2') ),
            (   'tc'        ,   np.ushort )]

eh = [
            (   'c1'        ,   np.dtype('S4') ),
            (   'serial'        ,   np.int32),
            (   'year'        ,   np.ushort),
            (   'month'        ,   np.ushort ),
            (   'day'        ,   np.ushort ),
            (   'hour'        ,   np.ushort ),
            (   'minute'        ,   np.ushort ),
            (   'sec'        ,   np.ushort ),
            (   'millisec'        ,   np.ushort ),
            (   'range'        ,   np.ushort )]

time_bins = np.zeros((2,4,1024)) #for two channels
#time = np.zeros((1,2,1024))
#wave = np.zeros((1,2,1024))
nb = 0 #number of boards

#count the negative timebins
cct = 0
for i in range(1024):
    if time_bins[0,0,i]<=0:
        cct +=1
print(cct)


vch1 = np.zeros((2,4,200000,1024))
tcell = np.zeros((2,200000))
#read board header
#with open(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\data_100000', 'rb') as f:
with open(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\data_200000.dat', 'rb') as f:
    b = np.fromfile(f, dtype=np.dtype('S4'), count=1)
#    print('file header=',b.astype(str))
    b = np.fromfile(f, dtype=np.dtype('S4'), count=1)
#    print('time header=',b.astype(str))
    c1 =0
    while(1):
        b = np.fromfile(f, dtype=bh, count=1)
        bb = b['c1'].astype(str)
#        print(b['c2'])
        if bb!='B#':
            #event header found
            f.seek(-4,1)
            break
#        print('board serial number',b['c2'])
        for i in range(5):#keep looping for time bins for all channels
            b = np.fromfile(f, dtype=ch1, count=1)
            bb = b['c1'].astype(str)
#            print(bb)
            if bb != 'C':
                f.seek(-4,1)
                break
            i11 = int(b['c2'])
#            print('found time calibration of channel', i11)
            b = np.fromfile(f, dtype=np.float32, count=1024)
            time_bins[c1,i] = b
#            print(b)
        c1 +=1
    nb = c1
#    print('number of boards', c1)
    
    cvc = 0 #counter for number of events to read
    while(1): #loop over events
        be = np.fromfile(f, dtype=eh, count=1)
        if not be:
            break
#        print('found event', int(be['serial']), int(be['sec']), int(be['millisec']))
        for i1 in range(nb):#number of boards
            b1 = np.fromfile(f, dtype=bh, count=1)
            bbb = b1['c1'].astype(str)
            if bbb != 'B#':
                print('invalid board header....exiting....')
                sys.exit()
                
            bt = np.fromfile(f, dtype=tch, count=1)
            bb = bt['c1'].astype(str)
            if bb != 'T#':
                print('invalid trigger cell....exiting....')
                sys.exit()            
            if nb > 1:
                bserial = b1['c2'].astype(str)
#                print('board serial is ' ,bserial)
                
#            plt.figure()
            tcell[i1,cvc] = bt['tc'] #get trigger cell
            for ch in range(4):#get channels data
#                print('we are hre')
                b = np.fromfile(f, dtype=ch1, count=1)
                bb = b['c1'].astype(str)
                if bb != 'C':
                    f.seek(-4,1)
                    break
#                print(b['c2'])
                ch_ind = int(b['c2'])-1
                s = np.fromfile(f, dtype=np.int32, count=1)#get scaler
                v = list(np.fromfile(f, dtype=np.ushort, count=1024))#get sample value
#                v[:] = [x - np.mean(v[15:1015]) for x in v]
                vch1[i1,ch_ind,cvc] = v
#                plt.plot(v)
#                print(vch1[ch_ind,cvc])
                
                 

#                plt.plot(v)
        
#                for i4 in range(1024):#convert data to volts
#                    wave[i1,ch_ind,i4] = (v[i4] / 65536. + be['range']/1000.0 - 0.5)
#                    #calculate time for each cell of present channel
#                    for j2 in range(i4):
#                        time[i1,ch_ind,i4] += time_bins[i1,ch_ind,((j2+bt['tc'])%1024)] 
#                vch1[cvc] = wave[i1,ch_ind] #saving data
#                tch1[cvc] = time[i1,ch_ind] #saving data
#                print('channel ch',ch)
#                print(tch1[cvc])
#            #allign cell 0 of all channels
#            t1 = time[i1,0,(1024-bt['tc']) % 1024]
#            for chn in range(1,2):
#                t2 = time[i1,chn,(1024-bt['tc']) % 1024]
#                dt = t1 - t2
#                for i5 in range(1024):
#                    time[i1,chn,i5] += dt
#            t1 = 0
#            t2 = 0
#            thres = 0.3
        cvc +=1
        if cvc >199998: #number of events to read (n-1)
            break
        

#plt.plot(vch1[2])        

#get time arrays for all trigger cells
time_samples = np.zeros((4,1024,1024))#channel,trigger cell, timebins
for i in range(4):#channles
    for j in range(1024): #trigger cell
        temptime = np.zeros(1024)
        for k in range(1024): #timebins
            q, r = divmod(j+k,1024)
            if q:
                temptime[k] = np.sum(time_bins[0,i,j:(j+k)]) + np.sum(time_bins[0,i,0:r])
            else:
                temptime[k] = np.sum(time_bins[0,i,j:(j+k)])
        
        time_samples[i,j] = np.copy(temptime)

#time alignment
for j in range(1024):#trigger cells
    t1 = 0
    t2 = 0
    time1 = time_samples[0,j]
    t1 = time1[(1024-j) % 1024]
    for ii in range(1,4):
        time2 = time_samples[ii,j]
        t2 = time2[(1024-j) % 1024]
    
        dt = t1 - t2
        for j1 in range(1024):
            time_samples[ii,j,j1] += dt
        
#get time arrays for all trigger cells of second board
time_samples1 = np.zeros((1,1024,1024))#channel,trigger cell, timebins
for i in range(1):#channles
    for j in range(1024): #trigger cell
        temptime = np.zeros(1024)
        for k in range(1024): #timebins
            q, r = divmod(j+k,1024)
            if q:
                temptime[k] = np.sum(time_bins[1,i,j:(j+k)]) + np.sum(time_bins[1,i,0:r])
            else:
                temptime[k] = np.sum(time_bins[1,i,j:(j+k)])
        
        time_samples1[i,j] = np.copy(temptime)

#time alignment
for j in range(1024):#trigger cells
    t1 = 0
    t2 = 0
    time1 = time_samples1[0,j]
    t1 = time1[(1024-j) % 1024]
    
    
    
    

#from here
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
a00 = np.zeros((4,199998,1000))
#a2 = np.zeros((10000,10000))
chchch = [1,3] 
for i1 in range(4):
    cct =0
    for i in range(199998):
        y1 = np.longdouble(vch1[0,i1,i]) - np.mean(np.longdouble(vch1[0,i1,i,5:105])) #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
        x1 = time_samples[i1,int(tcell[0,i])] #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
        
        #linear interp
        xs = np.arange(15,1015,1.)#160/4000 = 0.04
        f2 = interp1d(x1,y1) #,kind='previous'
        a00[i1,i] = f2(xs)#f2(xs)#- np.mean(f2(xs)),y1[15:1015]


aa00 = np.zeros((1,199998,1000))
for i1 in range(1):
    cct =0
    for i in range(199998):
        y1 = np.longdouble(vch1[1,i1,i]) - np.mean(np.longdouble(vch1[1,i1,i,5:105])) #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
        x1 = time_samples[i1,int(tcell[1,i])] #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
        
        #linear interp
        xs = np.arange(15,1015,1.)#160/4000 = 0.04
        f2 = interp1d(x1,y1) 
        aa00[i1,i] = y1[15:1015]#f2(xs)#- np.mean(f2(xs)),y1[15:1015]


#
#p1_ind = np.zeros(3)
#for i1 in range(3):
#    for i in range(9999):
#        if 500. < np.max(a00[i1,i]) <1000.:#3500,4000
#            p1_ind[i1] = i
#            plt.plot(a00[i1,i])
#            break
#
#p1_ind1 = np.zeros(1)
#for i1 in range(1):
#    for i in range(999):
#        if 1500. < np.max(aa00[i1,i]) <6000.:
#            p1_ind1[i1] = i
#            plt.plot(aa00[i1,i])
#            break
#
#xx_inp_ring = np.fft.fft(a00[3,int(p1_ind[2])])
#xx_inp_ring = np.fft.fft(a00[3,int(p1_ind1[0])])
#
##xx_inp_ring = np.fft.fft(a00[nn,a0_start[nn]:(a0_start[nn]+1500)])
##out_x = xx_inp_ring[0:1500]/hd1[0:1500]
#out_x = xx_inp_ring/h65
#out_xn = np.real(np.fft.ifft(out_x))
#plt.figure()
#from matplotlib.legend_handler import HandlerLine2D
##mm, = plt.plot(out_xn[300+500:400+500],'r',label='recovered') #recoverd anode pulse
#mm, = plt.plot(out_xn,'r',label='recovered') #recoverd anode pulse
#nn, = plt.plot(a00[2,int(p1_ind[2])],'g',label='original')#original anode pulse
##nn, = plt.plot(aa00[0,int(p1_ind1[0])],'g',label='original')#original anode pulse
#
#plt.xlabel('sample number',fontsize=16)
#plt.ylabel('sample value',fontsize=16)
#plt.legend(loc=4)
#
#plt.figure()
#plt.plot(f0[0:500],np.abs(np.fft.fft(out_xn))[0:500])
#plt.plot(f0[0:500],np.abs(np.fft.fft(a00[2,int(p1_ind[2])]))[0:500])
#
#plt.xlabel('frequency (MHz)', fontsize=16)
#plt.ylabel('power\n(arb. units)', fontsize=16)
#plt.plot(out_xn)
#plt.plot(a22[7])
#
#output_signal = scipy.signal.filtfilt(b, a, out_xn)
##output_signal1 = scipy.signal.filtfilt(b1, a1, out_xn)
#plt.plot(f0[0:500],np.abs(np.fft.fft(output_signal))[0:500])
#
#plt.figure()
#from matplotlib.legend_handler import HandlerLine2D
##mm, = plt.plot(output_signal[800:900],'r',label='recovered') #recoverd anode pulse
#mm, = plt.plot(output_signal[100:240]/2**16*1000,'b',label='recovered',linewidth=2.0,alpha=0.7) #recoverd anode pulse
##mm, = plt.plot(output_signal[110:210]/2**16*1000,'b',label='recovered') #recoverd anode pulse
##mm1, = plt.plot(output_signal1[300:400],'r',label='recovered1') #recoverd anode pulse
#nn, = plt.plot(a00[2,int(p1_ind[2]),100:240]/2**16*1000,'g',label='original',linewidth=2.0,alpha=0.7)#original anode pulse
##nn, = plt.plot(aa00[0,int(p1_ind1[0]),110-4:210-4]/2**16*1000,'g',label='original')#original anode pulse
#
#plt.title('65 MHz',fontsize=20)
#plt.xlabel('time (ns)',fontsize=16)
#plt.ylabel('sample value (mV)',fontsize=16)
#plt.legend(loc=1)
#plt.tight_layout()
#
#plt.plot(a00[2,123,120:240],'g')
#plt.xlabel('sample number',fontsize=16)
#plt.ylabel('sample value (ADC units)',fontsize=16)
#plt.legend(loc=1)
#plt.tight_layout()
#plt.plot(a00[3,123,60:600],'g')
#plt.xlabel('sample number',fontsize=16)
#plt.ylabel('sample value (ADC units)',fontsize=16)
#plt.legend(loc=1)
#plt.tight_layout()


res_50 = []
res_55 = []
res_60 = []
res_65 = []
r_extra = []
for i in range(199998):
    res_temp50 = []
    res_temp55 = []
    res_temp60 = []
    res_temp65 = []
    cnt = 0
    cnt50 = 0
    cnt55 = 0
    cnt60 = 0
    cnt65 = 0
    d1 = np.abs(np.fft.fft(a00[3,i]))
    if 10 + np.argmax(d1[10:500])== 49 or 10 + np.argmax(d1[10:500])== 50 or 10 + np.argmax(d1[10:500])== 51:
#    if 48 + np.argmax(d1[48:52]) == 49 or 48 + np.argmax(d1[48:52]) == 50:
#        if d1[48 + np.argmax(d1[48:52])] - d1[48 + np.argmax(d1[48:52]) - 1] > 0. and d1[48 + np.argmax(d1[48:52])] - d1[48 + np.argmax(d1[48:52]) - 2] > 5000.:
#            if d1[48 + np.argmax(d1[48:52])] - d1[48 + np.argmax(d1[48:52]) + 1] > 0. and d1[48 + np.argmax(d1[48:52])] - d1[48 + np.argmax(d1[48:52]) + 2] > 5000.:
        res_temp50.append(a00[3,i])
        res_temp50.append(a00[0,i])
        res_50.append(res_temp50)

        
#                cnt += 1
#                cnt50 += 1
#    elif 10 + np.argmax(d1[10:500])== 54 or 10 + np.argmax(d1[10:500])== 55 or 10 + np.argmax(d1[10:500])== 56:            
##    if 53 + np.argmax(d1[53:57]) == 54 or 53 + np.argmax(d1[53:57]) == 55:
##        if d1[53 + np.argmax(d1[53:57])] - d1[53 + np.argmax(d1[53:57]) - 1] > 0. and d1[53 + np.argmax(d1[53:57])] - d1[53 + np.argmax(d1[53:57]) - 2] > 5000.:
##            if d1[53 + np.argmax(d1[53:57])] - d1[53 + np.argmax(d1[53:57]) + 1] > 0. and d1[53 + np.argmax(d1[53:57])] - d1[53 + np.argmax(d1[53:57]) + 2] > 5000.:
#        res_temp55.append(a00[3,i])
#        res_temp55.append(aa00[0,i])#SiPM siganla connected to second board
#        res_55.append(res_temp55)


#                cnt += 1
#                cnt55 += 1
    elif 10 + np.argmax(d1[10:500])== 59 or 10 + np.argmax(d1[10:500])== 60 or 10 + np.argmax(d1[10:500])== 61:
#    if 58 + np.argmax(d1[58:62]) == 59 or 58 + np.argmax(d1[58:62]) == 60:
#        if d1[58 + np.argmax(d1[58:62])] - d1[58 + np.argmax(d1[58:62]) - 1] > 0. and d1[58 + np.argmax(d1[58:62])] - d1[58 + np.argmax(d1[58:62]) - 2] > 5000.:
#            if d1[58 + np.argmax(d1[58:62])] - d1[58 + np.argmax(d1[58:62]) + 1] > 0. and d1[58 + np.argmax(d1[58:62])] - d1[58 + np.argmax(d1[58:62]) + 2] > 5000.:
        res_temp60.append(a00[3,i])
        res_temp60.append(a00[1,i])
        res_60.append(res_temp60)


#                cnt += 1 
#                cnt60 += 1
    elif 10 + np.argmax(d1[10:500])== 64 or 10 + np.argmax(d1[10:500])== 65 or 10 + np.argmax(d1[10:500])== 66:            
#    if 63 + np.argmax(d1[63:67]) == 64 or 63 + np.argmax(d1[63:67]) == 65:
#        if d1[63 + np.argmax(d1[63:67])] - d1[63 + np.argmax(d1[63:67]) - 1] > 0. and d1[63 + np.argmax(d1[63:67])] - d1[63 + np.argmax(d1[63:67]) - 2] > 5000.:
#            if d1[63 + np.argmax(d1[63:67])] - d1[63 + np.argmax(d1[63:67]) + 1] > 0. and d1[63 + np.argmax(d1[63:67])] - d1[63 + np.argmax(d1[63:67]) + 2] > 5000.:
        res_temp65.append(a00[3,i])
        res_temp65.append(a00[2,i])#SiPM siganla connected to second board
        res_65.append(res_temp65)
    else:
        r_extra.append(a00[3,i])

res_50 = np.asarray(res_50)
#res_55 = np.asarray(res_55)
res_60 = np.asarray(res_60)
res_65 = np.asarray(res_65)

from signal import signal 
plt.plot(np.convolve(h50_t[0:1000], res_50[100,1], mode='full')[0:1000])   
plt.plot(res_50[100,0]) 
plt.show()    

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
# Always scale the input. The most convenient way is to use a pipeline.
reg = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=1000, tol=1e-3))
reg.fit(X, y)
reg.get_params()
plt.plot(X,y)
reg.score(X, y)
reg.coef_


reg = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, fit_intercept=False, max_iter=1000, tol=None, shuffle=True, verbose=0, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False, n_iter_no_change=50, warm_start=False, average=False)
reg.fit(X, y)


import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.interpolate import UnivariateSpline
N, Wn = signal.buttord((130/500),(160/500) , 3, 10) #(240/500),(290/500) , 3, 10, charge
b, a = signal.butter(N, Wn, 'low')

#h50_f = np.fft.fft(h50_t1[0:1050])
#rec_50 = np.zeros((len(res_50),1050))
#for i in range(len(res_50)):
#    out_x =  np.fft.fft(np.lib.pad(res_50[i,0], (0,50), 'constant', constant_values=(0., 0.)))/h50_f
#    out_xn = np.real(np.fft.ifft(out_x))
#    rec_50[i] = scipy.signal.filtfilt(b, a, out_xn)
    
h50_f = np.fft.fft(h50_t[0:1000])
rec_50 = np.zeros((len(res_50),1000))
for i in range(len(res_50)):
    out_x =  np.fft.fft(res_50[i,0])/h50_f
    out_xn = np.real(np.fft.ifft(out_x))
    rec_50[i] = scipy.signal.filtfilt(b, a, out_xn)

from collections import deque 
h55_f = np.fft.fft(h55_t[0:1000])
rec_55 = np.zeros((len(res_55),1000))
for i in range(len(res_55)):
    out_x =  np.fft.fft(res_55[i,0])/h55_f#np.lib.pad(res_55[i,0], (0,100), 'constant', constant_values=(0., 0.))
    out_xn = np.real(np.fft.ifft(out_x))
    rec_55[i] = scipy.signal.filtfilt(b, a, out_xn)
    dd= deque(rec_55[i])
    dd.rotate(31)
    rec_55[i] = dd

h60_f = np.fft.fft(h60_t[0:1000])
rec_60 = np.zeros((len(res_60),1000))
for i in range(len(res_60)):
    out_x =  np.fft.fft(res_60[i,0])/h60_f
    out_xn = np.real(np.fft.ifft(out_x))
    rec_60[i] = scipy.signal.filtfilt(b, a, out_xn)

h65_f = np.fft.fft(h65_t[0:1000])
rec_65 = np.zeros((len(res_65),1000))
for i in range(len(res_65)):
    out_x =  np.fft.fft(res_65[i,0])/h65_f
    out_xn = np.real(np.fft.ifft(out_x))
    rec_65[i] = scipy.signal.filtfilt(b, a, out_xn)

for i in range(199,200):
    plt.plot(rec_65[i]/2**16*1000)
    plt.plot(res_65[i,1]/2**16*1000)




#50 MHz

      
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.interpolate import UnivariateSpline
#N1, Wn1 = signal.buttord((140/500),(168/500) , 0.05, 10) #(240/500),(290/500) , 3, 10, charge        (130/500),(170/500) , 3, 10, timing  (290/500),(325/500) , 3, 10 (last), (140/500),(168/500) , 0.05, 10
#b1, a1 = signal.butter(N1, Wn1, 'low')

N, Wn = signal.buttord((240/500),(275/500) , 3, 10) #(240/500),(290/500) , 3, 10, charge        (130/500),(170/500) , 3, 10, timing  (290/500),(325/500) , 3, 10 (last), (140/500),(168/500) , 0.05, 10
b, a = signal.butter(N, Wn, 'low')

#h50_f = np.fft.fft(h50_t1[0:1050])
#rec_50 = np.zeros((len(res_50),1050))
#for i in range(len(res_50)):
#    out_x =  np.fft.fft(np.lib.pad(res_50[i,0], (0,50), 'constant', constant_values=(0., 0.)))/h50_f
#    out_xn = np.real(np.fft.ifft(out_x))
#    rec_50[i] = scipy.signal.filtfilt(b, a, out_xn)


h50_f = np.fft.fft(h50_t[0:1000])
for i in range(108,len(res_50)):#len(res_50)
    out_x =  np.fft.fft(res_50[i,0])/h50_f
    out_xn = np.real(np.fft.ifft(out_x))
    resh = scipy.signal.filtfilt(b, a, out_xn)# - np.average(scipy.signal.filtfilt(b, a, out_xn)[20:60])
    if 220<np.max(res_50[i,1,100:900])/2**16*1000<232: 
#        plt.figure()
#        plt.plot(res_50[i,1])
        plt.plot(resh)
#        plt.plot(f0[0:500],np.abs(np.fft.fft(res_50[i,0]))[0:500])
#        plt.plot(f0[0:500],np.abs(np.fft.fft(resh))[0:500])
#        plt.plot(f0[0:500],np.abs(np.fft.fft(out_xn))[0:500])
#        plt.plot(f0[0:500],np.abs(np.fft.fft(res_50[i,1]))[0:500])
        break
    

#h50_f = np.fft.fft(h50_t[0:1000])
#rec_50 = np.zeros((35000,1000))#len(res_50)
#for i in range(35000):#len(res_50)
#    out_x =  np.fft.fft(res_50[i,0])/h50_f
#    out_xn = np.real(np.fft.ifft(out_x))
#    rec_501 = scipy.signal.filtfilt(b1, a1, out_xn)
#    if np.max(rec_501[100:800])/2**16*1000 > 140.:
#        rec_50[i] = scipy.signal.filtfilt(b, a, out_xn)
#    else:
#        rec_50[i] = scipy.signal.filtfilt(b1, a1, out_xn)

   
h50_f = np.fft.fft(h50_t[0:1000])
rec_50 = np.zeros((35000,1000))#len(res_50)
for i in range(35000):#len(res_50)
    out_x =  np.fft.fft(res_50[i,0])/h50_f
    out_xn = np.real(np.fft.ifft(out_x))
    rec_50[i] = scipy.signal.filtfilt(b, a, out_xn)




import copy
g_res50=[]
rres_50 = copy.deepcopy(res_50)
rrec_50 = copy.deepcopy(rec_50)
t_ind = []
t_res50 = []
from scipy.interpolate import UnivariateSpline
max_res50 = np.zeros((len(rres_50),2))
for i in range(len(rec_50)):#len(rec_50)
    max_arg1 = 100 + np.argmax(rres_50[i,1,100:900])
    abcissa1 = np.asarray([ind for ind in range(max_arg1-4, max_arg1+3)])
    ordinate1 = rres_50[i,1,abcissa1]
    spl1 = UnivariateSpline(abcissa1, ordinate1,k=3)
    xs1 = np.linspace(max_arg1-4, max_arg1+3,100)
    max_res50[i,0] = np.max(spl1(xs1))
#    max_res50[i,0] = np.max(rres_50[i,1,100:900])

    max_arg2 = 100 + np.argmax(rec_50[i, 100:900])
    abcissa2 = np.asarray([ind for ind in range(max_arg2-4, max_arg2+3)])
    ordinate2 = rrec_50[i,abcissa2]
    spl2 = UnivariateSpline(abcissa2, ordinate2,k=3)
    xs2 = np.linspace(max_arg2-4, max_arg2+3,100)
    max_res50[i,1] = np.max(spl2(xs2))
#    max_res50[i,1] = np.max(rec_50[i, 100:900])
    
    
    if 2480.89<np.max(spl1(xs1)) < 17000. and -30. < (max_res50[i,0] - max_res50[i,1])/2**16*1000 < 20.:
        t_ind.append(i)
        g_temp =[]
        g_temp.append(max_res50[i,0])
        g_temp.append(max_res50[i,1])
        g_res50.append(g_temp)
        
#        t_temp = []
#        for ii in range(len(xs1)):
#            if spl1(xs1)[ii] <= max_res50[i,0]/2 < spl1(xs1)[ii+1]:
#                yy1 = spl1(xs1)[ii]
#                xx1 = xs1[ii]
#                yy2 = spl1(xs1)[ii+1]
#                xx2 = xs1[ii+1]
#                break
#        slope = (yy1 - yy2) / (xx1 - xx2)
#        intrcept = yy1 - (slope*xx1) 
#        t_temp.append(((max_res50[i,0]/2) - intrcept)/slope )
#    
#        for ii in range(len(xs2)-1):
#            if spl2(xs2)[ii] <= max_res50[i,1]/2 < spl2(xs2)[ii+1]:
#                yy1 = spl2(xs2)[ii]
#                xx1 = xs2[ii]
#                yy2 = spl2(xs2)[ii+1]
#                xx2 = xs2[ii+1]
#                break
#        if ii==99:
#            t_temp.append(2. )
#            t_res50.append(t_temp)
#            continue
#                
#        slope = (yy1 - yy2) / (xx1 - xx2)
#        intrcept = yy1 - (slope*xx1) 
#        t_temp.append(((max_res50[i,1]/2) - intrcept)/slope )
#            
#        t_res50.append(t_temp)
##        print (i, max_res50[i,0], max_res50[i,1])
##        break
#    


        abc = np.asarray([ind for ind in range(max_arg1-4, max_arg1+1)])
        ordi = rres_50[i,1,abc]
        t_temp = []
        for ii in range(len(abc)-1):
            if ordi[ii] < max_res50[i,0]/2 < ordi[ii+1]:#np.max(rres_50[i,1,100:900])
                yy1 = ordi[ii]
                xx1 = abc[ii]
                yy2 = ordi[ii+1]
                xx2 = abc[ii+1]
                break
            
        slope = (yy1 - yy2) / (xx1 - xx2)
        intrcept = yy1 - (slope*xx1) 
        t_temp.append(((max_res50[i,0]/2) - intrcept)/slope )
    
    
        abc = np.asarray([ind for ind in range(max_arg2-4, max_arg2+1)])
        ordi = rrec_50[i,abc]
        for ii in range(len(abc)-1):
            if ordi[ii] < max_res50[i,1]/2 < ordi[ii+1]:#np.max(rec_50[i, 100:900])max_res50[i,1]
                yy1 = ordi[ii]
                xx1 = abc[ii]
                yy2 = ordi[ii+1]
                xx2 = abc[ii+1]
                break
            
        slope = (yy1 - yy2) / (xx1 - xx2)
        intrcept = yy1 - (slope*xx1) 
        t_temp.append(((max_res50[i,1]/2) - intrcept)/slope )
        
    
        t_res50.append(t_temp)
#        
        
        

#    if t_res50[i,0] - t_res50[i,1] < -2500:
#        plt.plot(rres_50[i,1]/2**16*1000)
#        plt.plot(rrec_50[i]/2**16*1000)    
#        break
#                
                

g_res50 = np.asarray(g_res50)#28653
t_res50 = np.asarray(t_res50)


for i in range(40,len(res_50)):
    if 1000<max_res50[i,0]<1500:
        
        plt.figure()
        plt.plot(rres_50[i,1]/2**16*1000)
        plt.plot(rrec_50[i]/2**16*1000)
        break
#        plt.plot(rres_50[i,0]/2**16*1000)

for i in range(len(res_50)):
    if -0.39> t_res50[i,0] - t_res50[i,1] >-0.4:
        
        plt.figure()
        plt.plot(rres_50[i,1]/2**16*1000)
        plt.plot(rrec_50[i]/2**16*1000)
        break       
    
plt.plot(res_50[54,0]/2**16*1000)
plt.xlabel('sample number (ns)')
plt.ylabel('sample value (mV)')
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\sinusoid_ex')
plt.plot(f0[20:60],np.abs(np.fft.fft(res_50[54,0]/2**16*1000))[20:60])
plt.xlabel('frequency (MHz)')
plt.ylabel('amplitude\n (arb. units)')

plt.hist((g_res50[:,0] - g_res50[:,1])/2**16*1000,800, color='gray')
plt.xlim(-12,6)
plt.ylabel('counts')
plt.xlabel('original peak - recovered peak \n (mV)')
plt.title('50 MHz', fontsize=18)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch50')

dif_ch = []
for i in range(len(g_res50)):
    if -12. < (g_res50[i,0] - g_res50[i,1])/2**16*1000 < 6:
        dif_ch.append((g_res50[i,0] - g_res50[i,1])/2**16*1000)

print(np.mean(dif_ch))#-4.035392607394933
print(np.sqrt(np.var(dif_ch)))#2.170848722031384

#plot 2d hist
plt.figure()
plt.hist2d((g_res50[:,0]-g_res50[:,1]), g_res50[:,1], bins=100)
plt.xlabel('original peak - recovered peak \n (mV)')
plt.ylabel('recovered peak \n (mV)')
plt.tight_layout()


plt.scatter((g_res50[:,1])/2**16*1000, g_res50[:,0]/2**16*1000)

plt.scatter(g_res50[0:3000,1]/2**16*1000, g_res50[0:3000,0]/2**16*1000,s=0.1, c='k', alpha=0.2)
plt.xlabel('recovered peak (mV)')#r'\textbf{time}
plt.ylabel('original peak (mV)')
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch50_line')

plt.hist2d((t_res50[:,0] - t_res50[:,1])*1000, g_res50[:,1], bins=100)


plt.hist((t_res50[:,0] - t_res50[:,1])*1000,200, color='gray')
plt.xlim(-600,200)
plt.ylabel('counts')
plt.xlabel('original timing - recovered timing \n (ps)')
plt.title('50 MHz', fontsize=18)
print(np.mean((t_res50[:,0] - t_res50[:,1])))#-4.252257682191539
print(np.sqrt(np.var((t_res50[:,0] - t_res50[:,1]))))#2.4252711802330493
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\t50')

dif_t = []
for i in range(len(t_res50)):
    if -550. < (t_res50[i,0] - t_res50[i,1])*1000 < 200.:
        dif_t.append(t_res50[i,0] - t_res50[i,1])

print(np.mean(dif_t)*1000)#-162.35651025639297
print(np.sqrt(np.var(dif_t))*1000)#99.21231141724061

plt.scatter(t_res50[0:9000,1], t_res50[0:9000,0],s=0.1, c='k', alpha=0.2)
plt.xlabel('recovered timing (ns)')#r'\textbf{time}
plt.ylabel('original timing (ns)')
plt.xlim(125,129)
plt.ylim(125,129)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\t50_line')


plt.figure()    
plt.hist((max_res50[:,0] - max_res50[:,1]),1800,alpha=0.6)#/2**16*1000
plt.xlim(-6,15)

plt.figure()
plt.hist(g_res50[:,0]/2**16*1000,200,alpha=0.7)#
plt.hist(g_res50[:,1],200,alpha=0.6)#

nbins = 150
fig7 = plt.figure()
hq, bnedgess  = np.histogram(g_res50[:,0],bins=nbins)
#plt.xlim([0, 25000])
#plt.ylim([0, 160])
plt.hist(g_res50[:,0], bins=nbins)
yxq = 0.8*np.max(hq[90:150])*np.ones(150)
yxq1 = 0.8*250.8*np.ones(150)
bne11=(bnedgess[1:]+bnedgess[:-1])/2
plt.plot(bne11,yxq)
plt.plot(bne11,yxq1)#13157
plt.xlabel('charge collected \n(arb. units)',fontsize=16)
plt.ylabel('frequency',fontsize=16)
plt.tight_layout()
plt.show()


# (0.8*467.5 at 13229.1)
keV = 477.3/13157 * g_res50[:,0]

import matplotlib.pyplot as plt
# Estimate the histogram
nbins = 150
fig7 = plt.figure()
hq, bnedgess  = np.histogram(keV,bins=nbins)
plt.xlim([0.*477.3/13157, 16500*477.3/13157])
#plt.ylim([0, 160])
plt.hist(keV, bins=nbins)
yxq = 0.8*np.max(hq[90:150])*np.ones(150)
bne11=(bnedgess[1:]+bnedgess[:-1])/2
plt.plot(bne11,yxq)
plt.xlabel('charge collected \n(keVee)',fontsize=16)
plt.ylabel('frequency',fontsize=16)
plt.tight_layout()
plt.show()
 

plt.hist((g_res50[:,0] - g_res50[:,1])*477.3/13157,100, color='gray')
plt.xlim(-30,15)
plt.ylabel('counts')
plt.xlabel('original peak - recovered peak \n (keV)')
plt.title('50 MHz', fontsize=18)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch50keV')

dif_ch = []
for i in range(len(g_res50)):
    if -30. < (g_res50[i,0] - g_res50[i,1])*477.3/13157 < 20:
        dif_ch.append((g_res50[i,0] - g_res50[i,1])*477.3/13157)

print(np.mean(dif_ch))#-7.549846419373711
print(np.sqrt(np.var(dif_ch)))#5.182315682372439

#plot 2d hist
plt.hist2d((g_res50[:,0]-g_res50[:,1])*477.3/13157, g_res50[:,1]*477.3/13157, bins=200)
plt.xlabel('original peak - recovered peak \n (keV)')
plt.ylabel('recovered peak \n (keV)')
plt.tight_layout()
plt.xlim(-30,20)

savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch50_2d')

plt.scatter((g_res50[:,1])*477.3/13157, g_res50[:,0]*477.3/13157)

plt.scatter(g_res50[0:4000,1]*477.3/13157, g_res50[0:4000,0]*477.3/13157,s=0.1, c='k', alpha=0.2)
plt.xlabel('recovered peak (keV)')#r'\textbf{time}
plt.ylabel('original peak (keV)')
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch50_linekeV')

    
e_range = [90., 120.,150., 200., 300., 400.,600.]
diff_amp50 = [[] for i in range(6)]
diff_t50 = [[] for i in range(6)]

for i in range(len(e_range)-1):
    for j in range(len(g_res50[:,1])):
        if e_range[i] < g_res50[j,1]*477.3/13157 < e_range[i+1]:
            diff_amp50[i].append((g_res50[j,0] - g_res50[j,1])*477.3/13157)
            diff_t50[i].append((t_res50[j,0] - t_res50[j,1])*1000)

diff_tt50 = [[] for i in range(6)]
for i in range(len(e_range)-1):
    for j in range(len(diff_t50[i])):
        if np.mean(diff_t50[i])-380 < diff_t50[i][j] < np.mean(diff_t50[i])+380:
            diff_tt50[i].append(diff_t50[i][j])
        

for i in range(len(e_range)-1):
#    print('mean i:', np.mean(diff_amp50[i]))
#    print('std i:', np.sqrt(np.var(diff_amp50[i])))
    print('mean ti:', np.mean(diff_tt50[i]))
    print('std ti:', np.sqrt(np.var(diff_tt50[i])))
    
#    plt.hist(diff_amp50[i], bins=60, color='gray')
#    plt.xlim(np.mean(diff_amp50[i])-23, np.mean(diff_amp50[i])+23)
    textstr = '\n'.join((
        r'$\mu=%.1f$ ps' % (np.mean(diff_tt50[i]), ), 
        r'$\sigma=%.1f$ ps' % ( np.sqrt(np.var(diff_tt50[i])), )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', alpha=0.5)
    
    # place a text box in upper left in axes coords

    if i <2:
        fig, ax = plt.subplots()
        ax.hist(diff_tt50[i], bins=35, color='gray')
        plt.xlim(np.mean(diff_tt50[i])-380, np.mean(diff_tt50[i])+380)
        plt.xlabel('original timing - recovered timing (ps)')

    elif i ==2:
        fig, ax = plt.subplots()
        ax.hist(diff_tt50[i], bins=59, color='gray')
        plt.xlim(np.mean(diff_tt50[i])-250, np.mean(diff_tt50[i])+250)
        plt.xlabel('original timing - recovered timing (ps)')

    elif i ==5:
        fig, ax = plt.subplots()
        ax.hist(diff_tt50[i], bins=60, color='gray')
        plt.xlim(np.mean(diff_tt50[i])-210, np.mean(diff_tt50[i])+210)
        plt.xlabel('original timing - recovered timing (ps)')

#    elif i ==1 or i==2:
#        fig, ax = plt.subplots()
#        ax.hist(diff_t50[i], bins=35, color='gray')
#        plt.xlim(np.mean(diff_t50[i])-380, np.mean(diff_t50[i])+380)
    else:
        fig, ax = plt.subplots()
        ax.hist(diff_tt50[i], bins=58, color='gray')#, label = 'mean: '+str(np.mean(diff_t50[i]))
        plt.xlim(np.mean(diff_tt50[i])-210, np.mean(diff_tt50[i])+210)
        plt.xlabel('original timing - recovered timing (ps)')
#        plt.legend(fontsize = 12)
        
    ax.text(0.02, 0.85, textstr, fontsize=16,
             transform=ax.transAxes)#, bbox=props
    savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\time50\time' + str(e_range[i+1]) )


#energy
    
diff_ampp50 = [[] for i in range(6)]
for i in range(len(e_range)-1):
    for j in range(len(diff_amp50[i])):
        if i < 4:
            if np.mean(diff_amp50[i])-15 < diff_amp50[i][j] < np.mean(diff_amp50[i])+15:
                diff_ampp50[i].append(diff_amp50[i][j])
        else:
            if np.mean(diff_amp50[i])-20 < diff_amp50[i][j] < np.mean(diff_amp50[i])+20:
                diff_ampp50[i].append(diff_amp50[i][j])            

for i in range(len(e_range)-1):
    print('mean i:', np.mean(diff_ampp50[i]))
    print('std i:', np.sqrt(np.var(diff_ampp50[i])))
#    print('mean ti:', np.mean(diff_tt50[i]))
#    print('std ti:', np.sqrt(np.var(diff_tt50[i])))
    
#    plt.hist(diff_amp50[i], bins=60, color='gray')
#    plt.xlim(np.mean(diff_amp50[i])-23, np.mean(diff_amp50[i])+23)
    textstr = '\n'.join((
        r'$\mu=%.1f$ keV' % (np.mean(diff_ampp50[i]), ), 
        r'$\sigma=%.1f$ keV' % ( np.sqrt(np.var(diff_ampp50[i])), )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', alpha=0.5)
    
    # place a text box in upper left in axes coords

    if i <6:
        fig, ax = plt.subplots()
        ax.hist(diff_ampp50[i], bins=35, color='gray')
        plt.xlabel('original peak - recovered peak \n (keV)')


    ax.text(0.02, 0.85, textstr, fontsize=16,
             transform=ax.transAxes)#, bbox=props
    savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\charge50\charge' + str(e_range[i+1]) )


        



#plt.hist(diff_t50[1],200)


#mean i: -5.990607282656101
#std i: 3.219040859598111
#mean i: -7.713109199321832
#std i: 3.3027541901539905
#mean i: -9.116279058142531
#std i: 3.395773053445996
#mean i: -10.619473294330014
#std i: 3.689123864701261
#mean i: -8.804022632896865
#std i: 4.431327533437651
#mean i: -2.4124805069606987
#std i: 5.366297557544715

#mean ti: -258.9225305756124
#std ti: 101.2629635300332
#mean ti: -259.53736152716215
#std ti: 82.55880575580392
#mean ti: -235.06927424948336
#std ti: 67.77077784251925
#mean ti: -191.54903384570028
#std ti: 55.40047721903093
#mean ti: -116.38729935499464
#std ti: 44.74562411145418
#mean ti: -44.36868319649093
#std ti: 38.96815366984458
    

#plot 2d hist
plt.hist2d((g_res50[:,0]-g_res50[:,1])*477.3/13157, g_res50[:,1]*477.3/13157, bins=220)
plt.xlabel('original peak - recovered peak \n (keV)')
plt.ylabel('recovered peak \n (keV)')
plt.tight_layout()
plt.xlim(-30,20)
plt.colorbar()

savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\charge50\chch50_2d')

plt.hist2d((t_res50[:,0] - t_res50[:,1])*1000, g_res50[:,1]*477.3/13157, bins=220)
plt.xlabel('original timing - recovered timing \n (ps)')
plt.ylabel('recovered energy \n (keV)')
plt.tight_layout()
plt.colorbar()
plt.xlim(-550,150)

savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\time50\t50_2d')


#get spread in uncertainty
from sklearn.utils import resample
for ii in range(6):
    mmean = []
    sstd = []
    mmeant = []
    sstdt = []
    for i in range(10):
        boot = resample(diff_ampp50[ii], replace=True, n_samples=2200)
        boot1 = resample(np.asarray(diff_tt50[ii]), replace=True, n_samples=2200)
    #    print(boot)
        mmean.append(np.mean(boot))
        sstd.append(np.sqrt(np.var(boot)))
        mmeant.append(np.mean(boot1))
        sstdt.append(np.sqrt(np.var(boot1)))
    print(np.sqrt(np.var(mmean)))#0.03005916781374867
    print(np.sqrt(np.var(sstdt)), '\n')#1.027954651505688

#0.059174567442890694
#0.8499732190498043 
#
#0.05308694958497315
#1.1328664650793427 
#
#0.04844332640897065
#1.1645579386937925 
#
#0.051917037387946466
#0.842277620054032 
#
#0.08015483472174273
#0.6825777900612054 
#
#0.12295226569632839
#0.5672685081781323 
    

#get spread in uncertainty
from sklearn.utils import resample
mmean = []
sstd = []
mmeant = []
sstdt = []
for i in range(10):
    boot = resample(dif_ch, replace=True, n_samples=10000)
    boot1 = resample(np.asarray(dif_t)*1000, replace=True, n_samples=10000)
#    print(boot)
    mmean.append(np.mean(boot))
    sstd.append(np.sqrt(np.var(boot)))
    mmeant.append(np.mean(boot1))
    sstdt.append(np.sqrt(np.var(boot1)))

print(np.mean(mmean))#-7.563461773410985
print(np.sqrt(np.var(mmean)))#0.04233146326677323

print(np.mean(sstd))#5.205066964867681
print(np.sqrt(np.var(sstd)))#0.04277673655957917

print(np.mean(mmeant))#-161.78181729890895
print(np.sqrt(np.var(mmeant)))#0.931768955566659

print(np.mean(sstdt))#98.85956238050703
print(np.sqrt(np.var(sstdt)))#0.7052259721860282
    



#60 MHz

      
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.interpolate import UnivariateSpline
N, Wn = signal.buttord((240/500),(275/500) , 3, 10) #(240/500),(275/500) , 3, 10  ,(270/500),(320/500) , 3, 10
b, a = signal.butter(N, Wn, 'low')

#h50_f = np.fft.fft(h50_t1[0:1050])
#rec_50 = np.zeros((len(res_50),1050))
#for i in range(len(res_50)):
#    out_x =  np.fft.fft(np.lib.pad(res_50[i,0], (0,50), 'constant', constant_values=(0., 0.)))/h50_f
#    out_xn = np.real(np.fft.ifft(out_x))
#    rec_50[i] = scipy.signal.filtfilt(b, a, out_xn)
    
h60_f = np.fft.fft(h60_t[0:1000])
rec_60 = np.zeros((len(res_60),1000))
for i in range(len(res_60)):
    out_x =  np.fft.fft(res_60[i,0])/h60_f
    out_xn = np.real(np.fft.ifft(out_x))
    rec_60[i] = scipy.signal.filtfilt(b, a, out_xn)



import copy
g_res60=[]
t_res60 = []
rres_60 = copy.deepcopy(res_60)
rrec_60 = copy.deepcopy(rec_60)
from scipy.interpolate import UnivariateSpline
max_res60 = np.zeros((len(rres_60),2))
for i in range(len(rec_60)):#len(rec_50)
    max_arg1 = 100 + np.argmax(rres_60[i,1,100:900])
    abcissa1 = np.asarray([ind for ind in range(max_arg1-4, max_arg1+3)])
    ordinate1 = rres_60[i,1,abcissa1]
    spl1 = UnivariateSpline(abcissa1, ordinate1,k=3)
    xs1 = np.linspace(max_arg1-4, max_arg1+3,100)
    max_res60[i,0] = np.max(spl1(xs1))
#    max_res50[i,0] = np.max(rres_50[i,1,100:900])

    max_arg2 = 100 + np.argmax(rec_60[i, 100:900])
    abcissa2 = np.asarray([ind for ind in range(max_arg2-4, max_arg2+3)])
    ordinate2 = rrec_60[i,abcissa2]
    spl2 = UnivariateSpline(abcissa2, ordinate2,k=3)
    xs2 = np.linspace(max_arg2-4, max_arg2+3,100)
    max_res60[i,1] = np.max(spl2(xs2))
#    max_res50[i,1] = np.max(rec_50[i, 100:900])
    
    
    if 2179.75<np.max(spl1(xs1)) < 17000. and -30. < (max_res60[i,0] - max_res60[i,1])/2**16*1000 < 20.:
        g_temp =[]
        g_temp.append(max_res60[i,0])
        g_temp.append(max_res60[i,1])
        g_res60.append(g_temp)
        
#        t_temp = []
#        for ii in range(len(xs1)):
#            if spl1(xs1)[ii] <= max_res50[i,0]/2 < spl1(xs1)[ii+1]:
#                yy1 = spl1(xs1)[ii]
#                xx1 = xs1[ii]
#                yy2 = spl1(xs1)[ii+1]
#                xx2 = xs1[ii+1]
#                break
#        slope = (yy1 - yy2) / (xx1 - xx2)
#        intrcept = yy1 - (slope*xx1) 
#        t_temp.append(((max_res50[i,0]/2) - intrcept)/slope )
#    
#        for ii in range(len(xs2)):
#            if spl2(xs2)[ii] <= max_res50[i,1]/2 < spl2(xs2)[ii+1]:
#                yy1 = spl2(xs2)[ii]
#                xx1 = xs2[ii]
#                yy2 = spl2(xs2)[ii+1]
#                xx2 = xs2[ii+1]
#                break
#        slope = (yy1 - yy2) / (xx1 - xx2)
#        intrcept = yy1 - (slope*xx1) 
#        t_temp.append(((max_res50[i,1]/2) - intrcept)/slope )
#            
#        t_res50.append(t_temp)
##        print (i, max_res50[i,0], max_res50[i,1])
##        break
#    
        abc = np.asarray([ind for ind in range(max_arg1-4, max_arg1+1)])
        ordi = rres_60[i,1,abc]
        t_temp = []
        for ii in range(len(abc)-1):
            if ordi[ii] < max_res60[i,0]/2 < ordi[ii+1]:
                yy1 = ordi[ii]
                xx1 = abc[ii]
                yy2 = ordi[ii+1]
                xx2 = abc[ii+1]
                break
            
        slope = (yy1 - yy2) / (xx1 - xx2)
        intrcept = yy1 - (slope*xx1) 
        t_temp.append(((max_res60[i,0]/2) - intrcept)/slope )
    
    
        abc = np.asarray([ind for ind in range(max_arg2-4, max_arg2+1)])
        ordi = rrec_60[i,abc]
        for ii in range(len(abc)-1):
            if ordi[ii] < max_res60[i,1]/2 < ordi[ii+1]:
                yy1 = ordi[ii]
                xx1 = abc[ii]
                yy2 = ordi[ii+1]
                xx2 = abc[ii+1]
                break
            
        slope = (yy1 - yy2) / (xx1 - xx2)
        intrcept = yy1 - (slope*xx1) 
        t_temp.append(((max_res60[i,1]/2) - intrcept)/slope )
        
    
        t_res60.append(t_temp)

#    if t_res50[i,0] - t_res50[i,1] < -2500:
#        plt.plot(rres_50[i,1]/2**16*1000)
#        plt.plot(rrec_50[i]/2**16*1000)    
#        break
#                
                

g_res60 = np.asarray(g_res60)
t_res60 = np.asarray(t_res60)


for i in range(len(res_50)):
    if 1500<max_res50[i,0]<2000:
        
        plt.figure()
        plt.plot(rres_50[i,1]/2**16*1000)
        plt.plot(rrec_50[i]/2**16*1000)
        break
#        plt.plot(rres_50[i,0]/2**16*1000)

for i in range(len(res_50)):
    if -0.39> t_res50[i,0] - t_res50[i,1] >-0.4:
        
        plt.figure()
        plt.plot(rres_50[i,1]/2**16*1000)
        plt.plot(rrec_50[i]/2**16*1000)
        break       
    

plt.hist((g_res60[:,0] - g_res60[:,1])/2**16*1000,80, color='gray')
plt.xlim(-13,4)
plt.ylabel('counts')
plt.xlabel('original peak - recovered peak \n (mV)')
plt.title('60 MHz', fontsize=18)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch60')

dif_ch60 = []
for i in range(len(g_res60)):
    if -13. < (g_res60[i,0] - g_res60[i,1])/2**16*1000 < 4:
        dif_ch60.append((g_res60[i,0] - g_res60[i,1])/2**16*1000)

print(np.mean(dif_ch60))#-4.836626739666186
print(np.sqrt(np.var(dif_ch60)))#2.2148678687878913



#plot 2d hist
plt.hist2d((g_res60[:,0]-g_res60[:,1])/2**16*1000, g_res60[:,1]/2**16*1000, bins=100)
plt.scatter((g_res60[:,1])/2**16*1000, g_res60[:,0]/2**16*1000)

plt.scatter(g_res60[0:3000,1]/2**16*1000, g_res60[0:3000,0]/2**16*1000,s=0.1, c='k', alpha=0.2)
plt.xlabel('recovered peak (mV)')#r'\textbf{time}
plt.ylabel('original peak (mV)')
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch60_line')

plt.hist2d((t_res60[:,0] - t_res60[:,1])*1000, g_res60[:,0], bins=100)


plt.hist((t_res60[:,0] - t_res60[:,1])*1000,250, color='gray')
plt.xlim(-550,200)
plt.ylabel('counts')
plt.xlabel('original timing - recovered timing \n (ps)')
plt.title('60 MHz', fontsize=18)
print(np.mean((t_res60[:,0] - t_res60[:,1])))#-4.252257682191539
print(np.sqrt(np.var((t_res60[:,0] - t_res60[:,1]))))#2.4252711802330493
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\t60')

dif_t6 = []
for i in range(len(t_res60)):
    if -600. < (t_res60[i,0] - t_res60[i,1])*1000 < 200.:
        dif_t6.append(t_res60[i,0] - t_res60[i,1])

print(np.mean(dif_t6)*1000)#-107.60532738671458
print(np.sqrt(np.var(dif_t6))*1000)#96.8089285018617


plt.scatter(t_res60[0:5000,1], t_res60[0:5000,0],s=0.1, c='k', alpha=0.2)
plt.xlabel('recovered timing (ns)')#r'\textbf{time}
plt.ylabel('original timing (ns)')
plt.xlim(124.5,128.5)
plt.ylim(124.5,128.5)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\t60_line')


plt.figure()    
plt.hist((max_res50[:,0] - max_res50[:,1]),1800,alpha=0.6)#/2**16*1000
plt.xlim(-6,15)

plt.hist(max_res60[:,0],200,alpha=0.7)#/2**16*1000
plt.hist(max_res60[:,1],200,alpha=0.6)


nbins = 100
fig7 = plt.figure()
hq, bnedgess  = np.histogram(g_res60[:,0],bins=nbins)
#plt.xlim([0, 25000])
#plt.ylim([0, 160])
plt.hist(g_res60[:,0], bins=nbins, alpha=0.6)
yxq = 0.8*np.max(hq[60:100])*np.ones(100)
yxq1 = 0.8*441.2*np.ones(100)
bne11=(bnedgess[1:]+bnedgess[:-1])/2
plt.plot(bne11,yxq)
plt.plot(bne11,yxq1)#11560
plt.xlabel('charge collected \n(arb. units)',fontsize=16)
plt.ylabel('frequency',fontsize=16)
plt.tight_layout()
plt.show()


# (0.8*467.5 at 11972)
keV = 477.3/11560* g_res60[:,0]

import matplotlib.pyplot as plt
# Estimate the histogram
nbins = 100
fig7 = plt.figure()
hq, bnedgess  = np.histogram(keV,bins=nbins)
plt.xlim([0.*477.3/11560, 16500*477.3/11560])
#plt.ylim([0, 160])
plt.hist(keV, bins=nbins)
yxq = 0.8*np.max(hq[60:100])*np.ones(100)
bne11=(bnedgess[1:]+bnedgess[:-1])/2
plt.plot(bne11,yxq)
plt.xlabel('charge collected \n(keVee)',fontsize=16)
plt.ylabel('frequency',fontsize=16)
plt.tight_layout()
plt.show()
 

plt.hist((g_res60[:,0] - g_res60[:,1])*477.3/11560,120, color='gray')
plt.xlim(-40,10)
plt.ylabel('counts')
plt.xlabel('original peak - recovered peak \n (keV)')
plt.title('60 MHz', fontsize=18)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch60keV')

dif_ch6 = []
for i in range(len(g_res60)):
    if -40. < (g_res60[i,0] - g_res60[i,1])*477.3/11560 < 10:
        dif_ch6.append((g_res60[i,0] - g_res60[i,1])*477.3/11560)

print(np.mean(dif_ch6))#-14.212945941344145
print(np.sqrt(np.var(dif_ch6)))#5.483092266113671

#plot 2d hist
plt.hist2d((g_res60[:,0]-g_res60[:,1])*477.3/11560, g_res60[:,1]*477.3/11560, bins=150)
plt.xlabel('original peak - recovered peak \n (keV)')
plt.ylabel('recovered peak \n (keV)')
plt.tight_layout()
plt.xlim(-35,15)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch60_2d')


plt.scatter((g_res60[:,1])*477.3/11560, g_res60[:,0]*477.3/11560)

plt.scatter(g_res60[0:4000,1]*477.3/11560, g_res60[0:4000,0]*477.3/11560,s=0.1, c='k', alpha=0.2)
plt.xlabel('recovered peak (keV)')#r'\textbf{time}
plt.ylabel('original peak (keV)')
plt.xlim(40,650)
plt.ylim(40,650)
plt.xticks(np.arange(100, 700, step=100))


savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch60_linekeV')

    
    
e_range = [90., 120.,150., 200., 300., 400.,600.]
diff_amp60 = [[] for i in range(6)]
diff_t60 = [[] for i in range(6)]

for i in range(len(e_range)-1):
    for j in range(len(g_res60[:,1])):
        if e_range[i] < g_res60[j,1]*477.3/11560 < e_range[i+1]:
            diff_amp60[i].append((g_res60[j,0] - g_res60[j,1])*477.3/11560)
            diff_t60[i].append((t_res60[j,0] - t_res60[j,1])*1000)

for i in range(len(e_range)-1):
    print('mean i:', np.mean(diff_amp60[i]))
    print('std i:', np.sqrt(np.var(diff_amp60[i])))
#    print('mean ti:', np.mean(diff_t60[i]))
#    print('std ti:', np.sqrt(np.var(diff_t60[i])))

#plt.hist(diff_t50[1],200)


#mean i: -7.267951483263939
#std i: 4.989330577141134
#mean i: -9.69003845385453
#std i: 4.716351595597545
#mean i: -12.18500671797813
#std i: 4.549102461068448
#mean i: -15.572286595218396
#std i: 4.78880000218279
#mean i: -17.666638855762486
#std i: 4.937238282809653
#mean i: -15.382494565272815
#std i: 5.772175908847493

#mean ti: -248.7252511785988
#std ti: 123.90999728264018
#mean ti: -252.00996638863728
#std ti: 101.96885458616498
#mean ti: -236.74365690810228
#std ti: 96.02381649310823
#mean ti: -199.88618072806761
#std ti: 87.04545018537162
#mean ti: -141.37183530581243
#std ti: 67.41209160483106
#mean ti: -72.48411420154476
#std ti: 58.075674631727594

#get spread in uncertainty
from sklearn.utils import resample
mmean = []
sstd = []
mmeant = []
sstdt = []
for i in range(10):
    boot = resample(dif_ch6, replace=True, n_samples=10000)
    boot1 = resample(np.asarray(dif_t6)*1000, replace=True, n_samples=10000)
#    print(boot)
    mmean.append(np.mean(boot))
    sstd.append(np.sqrt(np.var(boot)))
    mmeant.append(np.mean(boot1))
    sstdt.append(np.sqrt(np.var(boot1)))

print(np.mean(mmean))#-14.213723025746578
print(np.sqrt(np.var(mmean)))#0.03707059202950989

print(np.mean(sstd))#5.47500567875325
print(np.sqrt(np.var(sstd)))#0.048762343850211234

print(np.mean(mmeant))#-175.73393495994543
print(np.sqrt(np.var(mmeant)))#0.7522152218074709

print(np.mean(sstdt))#96.06575840919103
print(np.sqrt(np.var(sstdt)))#0.7393134467161594
  


    
e_range = [90., 120.,150., 200., 300., 400.,600.]
diff_amp60 = [[] for i in range(6)]
diff_t60 = [[] for i in range(6)]

for i in range(len(e_range)-1):
    for j in range(len(g_res60[:,1])):
        if e_range[i] < g_res60[j,1]*477.3/11560 < e_range[i+1]:
            diff_amp60[i].append((g_res60[j,0] - g_res60[j,1])*477.3/11560)
            diff_t60[i].append((t_res60[j,0] - t_res60[j,1])*1000)

diff_tt60 = [[] for i in range(6)]
for i in range(len(e_range)-1):
    for j in range(len(diff_t60[i])):
        if np.mean(diff_t60[i])-400 < diff_t60[i][j] < np.mean(diff_t60[i])+400:
            diff_tt60[i].append(diff_t60[i][j])
        

for i in range(len(e_range)-1):
#    print('mean i:', np.mean(diff_amp50[i]))
#    print('std i:', np.sqrt(np.var(diff_amp50[i])))
    print('mean ti:', np.mean(diff_tt60[i]))
    print('std ti:', np.sqrt(np.var(diff_tt60[i])))
    
#    plt.hist(diff_amp50[i], bins=60, color='gray')
#    plt.xlim(np.mean(diff_amp50[i])-23, np.mean(diff_amp50[i])+23)
    textstr = '\n'.join((
        r'$\mu=%.1f$ ps' % (np.mean(diff_tt60[i]), ), 
        r'$\sigma=%.1f$ ps' % ( np.sqrt(np.var(diff_tt60[i])), )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', alpha=0.5)
    
    # place a text box in upper left in axes coords

    if i <2:
        fig, ax = plt.subplots()
        ax.hist(diff_tt60[i], bins=35, color='gray')
        plt.xlim(np.mean(diff_tt60[i])-400, np.mean(diff_tt60[i])+400)
        plt.xlabel('original timing - recovered timing (ps)')

    elif i ==2:
        fig, ax = plt.subplots()
        ax.hist(diff_tt60[i], bins=59, color='gray')
        plt.xlim(np.mean(diff_tt60[i])-300, np.mean(diff_tt60[i])+300)
        plt.xlabel('original timing - recovered timing (ps)')

    elif i ==5:
        fig, ax = plt.subplots()
        ax.hist(diff_tt60[i], bins=60, color='gray')
        plt.xlim(np.mean(diff_tt60[i])-210, np.mean(diff_tt60[i])+210)
        plt.xlabel('original timing - recovered timing (ps)')

#    elif i ==1 or i==2:
#        fig, ax = plt.subplots()
#        ax.hist(diff_t50[i], bins=35, color='gray')
#        plt.xlim(np.mean(diff_t50[i])-380, np.mean(diff_t50[i])+380)
    else:
        fig, ax = plt.subplots()
        ax.hist(diff_tt60[i], bins=58, color='gray')#, label = 'mean: '+str(np.mean(diff_t50[i]))
        plt.xlim(np.mean(diff_tt60[i])-210, np.mean(diff_tt60[i])+210)
        plt.xlabel('original timing - recovered timing (ps)')
#        plt.legend(fontsize = 12)
        
    ax.text(0.02, 0.85, textstr, fontsize=16,
             transform=ax.transAxes)#, bbox=props
    savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\time60\time' + str(e_range[i+1]) )


#energy
    
diff_ampp60 = [[] for i in range(6)]
for i in range(len(e_range)-1):
    for j in range(len(diff_amp60[i])):
        if i < 4:
            if np.mean(diff_amp60[i])-15 < diff_amp60[i][j] < np.mean(diff_amp60[i])+15:
                diff_ampp60[i].append(diff_amp60[i][j])
        else:
            if np.mean(diff_amp60[i])-20 < diff_amp60[i][j] < np.mean(diff_amp60[i])+20:
                diff_ampp60[i].append(diff_amp60[i][j])            

for i in range(len(e_range)-1):
    print('mean i:', np.mean(diff_ampp60[i]))
    print('std i:', np.sqrt(np.var(diff_ampp60[i])))
#    print('mean ti:', np.mean(diff_tt50[i]))
#    print('std ti:', np.sqrt(np.var(diff_tt50[i])))
    
#    plt.hist(diff_amp50[i], bins=60, color='gray')
#    plt.xlim(np.mean(diff_amp50[i])-23, np.mean(diff_amp50[i])+23)
    textstr = '\n'.join((
        r'$\mu=%.1f$ keV' % (np.mean(diff_ampp60[i]), ), 
        r'$\sigma=%.1f$ keV' % ( np.sqrt(np.var(diff_ampp60[i])), )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', alpha=0.5)
    
    # place a text box in upper left in axes coords

    if i <6:
        fig, ax = plt.subplots()
        ax.hist(diff_ampp60[i], bins=35, color='gray')
        plt.xlabel('original peak - recovered peak \n (keV)')


    ax.text(0.02, 0.85, textstr, fontsize=16,
             transform=ax.transAxes)#, bbox=props
    savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\charge60\charge' + str(e_range[i+1]) )


        



#plt.hist(diff_t50[1],200)


#mean i: -7.578049650523575
#std i: 3.722073875994748
#mean i: -9.878093800955343
#std i: 3.722728025447158
#mean i: -12.341582888425952
#std i: 3.8930934518383795
#mean i: -15.675706997697612
#std i: 4.181525049222419
#mean i: -17.733055472054716
#std i: 4.590990904400713
#mean i: -15.414008965781452
#std i: 5.430133492750365

#mean ti: -251.92925744913936
#std ti: 114.85900286955997
#mean ti: -253.12000609137408
#std ti: 98.75799819431565
#mean ti: -238.2310657525804
#std ti: 80.26491549862268
#mean ti: -201.90157537588763
#std ti: 60.278542328414
#mean ti: -143.2744185965909
#std ti: 48.098271683628916
#mean ti: -74.01440416441386
#std ti: 43.17472243711694
    

#plot 2d hist
plt.hist2d((g_res60[:,0]-g_res60[:,1])*477.3/11560, g_res60[:,1]*477.3/11560, bins=220)
plt.xlabel('original peak - recovered peak \n (keV)')
plt.ylabel('recovered peak \n (keV)')
plt.tight_layout()
plt.xlim(-40,10)
plt.colorbar()


savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\charge60\chch60_2d')

plt.hist2d((t_res60[:,0] - t_res60[:,1])*1000, g_res60[:,1]*477.3/11560, bins=220)
plt.xlabel('original timing - recovered timing \n (ps)')
plt.ylabel('recovered energy \n (keV)')
plt.tight_layout()
plt.colorbar()

plt.xlim(-600,150)

savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\time60\t60_2d')

    






#65 MHz

      
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.interpolate import UnivariateSpline
N, Wn = signal.buttord((240/500),(275/500) , 3, 10 ) #(240/500),(275/500) , 3, 10  ,   (270/500),(320/500) , 3, 10
b, a = signal.butter(N, Wn, 'low')

#h50_f = np.fft.fft(h50_t1[0:1050])
#rec_50 = np.zeros((len(res_50),1050))
#for i in range(len(res_50)):
#    out_x =  np.fft.fft(np.lib.pad(res_50[i,0], (0,50), 'constant', constant_values=(0., 0.)))/h50_f
#    out_xn = np.real(np.fft.ifft(out_x))
#    rec_50[i] = scipy.signal.filtfilt(b, a, out_xn)
    
h65_f = np.fft.fft(h65_t[0:1000])
rec_65 = np.zeros((len(res_65),1000))
for i in range(len(res_65)):
    out_x =  np.fft.fft(res_65[i,0])/h65_f
    out_xn = np.real(np.fft.ifft(out_x))
    rec_65[i] = scipy.signal.filtfilt(b, a, out_xn)



import copy
g_res65=[]
t_res65 = []
rres_65 = copy.deepcopy(res_65)
rrec_65 = copy.deepcopy(rec_65)
from scipy.interpolate import UnivariateSpline
max_res65 = np.zeros((35000,2))#len(rres_65)
for i in range(35000):#len(rec_50)
    max_arg1 = 100 + np.argmax(rres_65[i,1,100:900])
    abcissa1 = np.asarray([ind for ind in range(max_arg1-4, max_arg1+3)])
    ordinate1 = rres_65[i,1,abcissa1]
    spl1 = UnivariateSpline(abcissa1, ordinate1,k=3)
    xs1 = np.linspace(max_arg1-4, max_arg1+3,100)
    max_res65[i,0] = np.max(spl1(xs1))
#    max_res50[i,0] = np.max(rres_50[i,1,100:900])

    max_arg2 = 100 + np.argmax(rec_65[i, 100:900])
    abcissa2 = np.asarray([ind for ind in range(max_arg2-4, max_arg2+3)])
    ordinate2 = rrec_65[i,abcissa2]
    spl2 = UnivariateSpline(abcissa2, ordinate2,k=3)
    xs2 = np.linspace(max_arg2-4, max_arg2+3,100)
    max_res65[i,1] = np.max(spl2(xs2))
#    max_res50[i,1] = np.max(rec_50[i, 100:900])
    
    
    if 1638.5920804525454<np.max(spl1(xs1)) < 17000. and -30. < (max_res65[i,0] - max_res65[i,1])/2**16*1000 < 20.:
        g_temp =[]
        g_temp.append(max_res65[i,0])
        g_temp.append(max_res65[i,1])
        g_res65.append(g_temp)
        
#        t_temp = []
#        for ii in range(len(xs1)):
#            if spl1(xs1)[ii] <= max_res65[i,0]/2 < spl1(xs1)[ii+1]:
#                yy1 = spl1(xs1)[ii]
#                xx1 = xs1[ii]
#                yy2 = spl1(xs1)[ii+1]
#                xx2 = xs1[ii+1]
#                break
#        slope = (yy1 - yy2) / (xx1 - xx2)
#        intrcept = yy1 - (slope*xx1) 
#        t_temp.append(((max_res65[i,0]/2) - intrcept)/slope )
#    
#        for ii in range(len(xs2)):
#            if spl2(xs2)[ii] <= max_res65[i,1]/2 < spl2(xs2)[ii+1]:
#                yy1 = spl2(xs2)[ii]
#                xx1 = xs2[ii]
#                yy2 = spl2(xs2)[ii+1]
#                xx2 = xs2[ii+1]
#                break
#        slope = (yy1 - yy2) / (xx1 - xx2)
#        intrcept = yy1 - (slope*xx1) 
#        t_temp.append(((max_res50[i,1]/2) - intrcept)/slope )
#            
#        t_res65.append(t_temp)
##        print (i, max_res50[i,0], max_res50[i,1])
##        break
#    
        abc = np.asarray([ind for ind in range(max_arg1-4, max_arg1+1)])
        ordi = rres_65[i,1,abc]
        t_temp = []
        for ii in range(len(abc)-1):
            if ordi[ii] < max_res65[i,0]/2 < ordi[ii+1]:
                yy1 = ordi[ii]
                xx1 = abc[ii]
                yy2 = ordi[ii+1]
                xx2 = abc[ii+1]
                break
            
        slope = (yy1 - yy2) / (xx1 - xx2)
        intrcept = yy1 - (slope*xx1) 
        t_temp.append(((max_res65[i,0]/2) - intrcept)/slope )
    
    
        abc = np.asarray([ind for ind in range(max_arg2-4, max_arg2+1)])
        ordi = rrec_65[i,abc]
        for ii in range(len(abc)-1):
            if ordi[ii] < max_res65[i,1]/2 < ordi[ii+1]:
                yy1 = ordi[ii]
                xx1 = abc[ii]
                yy2 = ordi[ii+1]
                xx2 = abc[ii+1]
                break
            
        slope = (yy1 - yy2) / (xx1 - xx2)
        intrcept = yy1 - (slope*xx1) 
        t_temp.append(((max_res65[i,1]/2) - intrcept)/slope )
        
    
        t_res65.append(t_temp)

#    if t_res50[i,0] - t_res50[i,1] < -2500:
#        plt.plot(rres_50[i,1]/2**16*1000)
#        plt.plot(rrec_50[i]/2**16*1000)    
#        break
#                
                

g_res65 = np.asarray(g_res65)
t_res65 = np.asarray(t_res65)


for i in range(len(res_50)):
    if 1500<max_res50[i,0]<2000:
        
        plt.figure()
        plt.plot(rres_50[i,1]/2**16*1000)
        plt.plot(rrec_50[i]/2**16*1000)
        break
#        plt.plot(rres_50[i,0]/2**16*1000)

for i in range(len(res_50)):
    if -0.39> t_res50[i,0] - t_res50[i,1] >-0.4:
        
        plt.figure()
        plt.plot(rres_50[i,1]/2**16*1000)
        plt.plot(rrec_50[i]/2**16*1000)
        break       
    

plt.hist((g_res65[:,0] - g_res65[:,1])/2**16*1000,80, color='gray')
plt.xlim(-10,4)
plt.ylabel('counts')
plt.xlabel('original peak - recovered peak \n (mV)')
plt.title('65 MHz', fontsize=18)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch65')

dif_ch65 = []
for i in range(len(g_res65)):
    if -10. < (g_res65[i,0] - g_res65[i,1])/2**16*1000 < 4:
        dif_ch65.append((g_res65[i,0] - g_res65[i,1])/2**16*1000)

print(np.mean(dif_ch65))#-3.347613819192282
print(np.sqrt(np.var(dif_ch65)))#1.9692694098591643


#plot 2d hist
plt.hist2d((g_res65[:,0]-g_res65[:,1])/2**16*1000, g_res65[:,1]/2**16*1000, bins=100)
plt.scatter((g_res65[:,1])/2**16*1000, g_res65[:,0]/2**16*1000)

plt.scatter(g_res65[0:3000,1]/2**16*1000, g_res65[0:3000,0]/2**16*1000,s=0.1, c='k', alpha=0.2)
plt.xlabel('recovered peak (mV)')#r'\textbf{time}
plt.ylabel('original peak (mV)')
plt.ylim(0,200)
plt.xlim(0,200)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch65_line')

plt.hist2d((t_res60[:,0] - t_res60[:,1])*1000, g_res60[:,0], bins=100)


plt.hist((t_res65[:,0] - t_res65[:,1])*1000,250, color='gray')
plt.xlim(-700,200)
plt.ylabel('counts')
plt.xlabel('original timing - recovered timing \n (ps)')
plt.title('65 MHz', fontsize=18)
plt.show()
print(np.mean((t_res65[:,0] - t_res65[:,1])))#-4.252257682191539
print(np.sqrt(np.var((t_res65[:,0] - t_res65[:,1]))))#2.4252711802330493
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\t65')

dif_t65 = []
for i in range(len(t_res65)):
    if -700. < (t_res65[i,0] - t_res65[i,1])*1000 < 200:
        dif_t65.append(t_res65[i,0] - t_res65[i,1])

print(np.mean(dif_t65)*1000)#-204.73215123993907
print(np.sqrt(np.var(dif_t65))*1000)#96.18471388911681

plt.scatter(t_res65[0:8000,1], t_res65[0:8000,0],s=0.1, c='k', alpha=0.2)
plt.xlabel('recovered timing (ns)')#r'\textbf{time}
plt.ylabel('original timing (ns)')
plt.xlim(125,129)
plt.ylim(125,129)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\t65_line')


plt.figure()    
plt.hist((max_res50[:,0] - max_res50[:,1]),1800,alpha=0.6)#/2**16*1000
plt.xlim(-6,15)

plt.hist(max_res65[:,0],200,alpha=0.7)#/2**16*1000
plt.hist(max_res65[:,1]/2**16*1000,200,alpha=0.6)


nbins = 100
fig7 = plt.figure()
hq, bnedgess  = np.histogram(g_res65[:,0],bins=nbins)
#plt.xlim([0, 25000])
#plt.ylim([0, 160])
plt.hist(g_res65[:,0], bins=nbins)
yxq = 0.8*np.max(hq[40:100])*np.ones(100)
yxq1 = 0.8*544*np.ones(100)
bne11=(bnedgess[1:]+bnedgess[:-1])/2
plt.plot(bne11,yxq)
plt.plot(bne11,yxq1)#8816
plt.xlabel('charge collected \n(arb. units)',fontsize=16)
plt.ylabel('frequency',fontsize=16)
plt.tight_layout()
plt.show()


# (0.8*467.5 at 16669)
keV = 477.3/8816 * g_res65[:,0]

import matplotlib.pyplot as plt
# Estimate the histogram
nbins = 100
fig7 = plt.figure()
hq, bnedgess  = np.histogram(keV,bins=nbins)
plt.xlim([0.*477.3/8816, 16500*477.3/8816])
#plt.ylim([0, 160])
plt.hist(keV, bins=nbins)
yxq = 0.8*544*np.ones(100)
bne11=(bnedgess[1:]+bnedgess[:-1])/2
plt.plot(bne11,yxq)
plt.xlabel('charge collected \n(keVee)',fontsize=16)
plt.ylabel('frequency',fontsize=16)
plt.tight_layout()
plt.show()
 

plt.hist((g_res65[:,0] - g_res65[:,1])*477.3/8816,130, color='gray')
plt.xlim(-38,15)
plt.ylabel('counts')
plt.xlabel('original peak - recovered peak \n (keV)')
plt.title('65 MHz', fontsize=18)
plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch65keV')

dif_ch = []
for i in range(len(g_res65)):
    if -38. < (g_res65[i,0] - g_res65[i,1])*477.3/8816 < 15:
        dif_ch.append((g_res65[i,0] - g_res65[i,1])*477.3/8816)

print(np.mean(dif_ch))#-13.040390159201435
print(np.sqrt(np.var(dif_ch)))#6.614103334011892

#plot 2d hist
plt.hist2d((g_res65[:,0]-g_res65[:,1])*477.3/8816, g_res65[:,0]*477.3/8816, bins=150)
plt.xlabel('original peak - recovered peak \n (keV)')
plt.ylabel('recovered peak \n (keV)')
plt.tight_layout()
plt.xlim(-35,15)
plt.ylim(50,700)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch65_2d')


plt.scatter((g_res65[:,1])*477.3/8816, g_res65[:,0]*477.3/8816)

plt.scatter(g_res65[0:4000,1]*477.3/8816, g_res65[0:4000,0]*477.3/8816,s=0.1, c='k', alpha=0.2)
plt.xlabel('recovered peak (keV)')#r'\textbf{time}
plt.ylabel('original peak (keV)')
plt.xlim(50,650)
plt.ylim(50,650)
plt.show()

savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\chch65_linekeV')

    

    
e_range = [90., 120.,150., 200., 300., 400.,600.]
diff_amp65 = [[] for i in range(6)]
diff_t655 = [[] for i in range(6)]

for i in range(len(e_range)-1):
    for j in range(len(g_res65[:,1])):
        if e_range[i] < g_res65[j,1]*477.3/8816 < e_range[i+1]:
            diff_amp65[i].append((g_res65[j,0] - g_res65[j,1])*477.3/8816)
            diff_t655[i].append((t_res65[j,0] - t_res65[j,1])*1000)

for i in range(len(e_range)-1):
#    print('mean i:', np.mean(diff_amp65[i]))
#    print('std i:', np.sqrt(np.var(diff_amp65[i])))
    print('mean ti:', np.mean(diff_t655[i]))
    print('std ti:', np.sqrt(np.var(diff_t655[i])))

#plt.hist(diff_t50[1],200)


#mean i: -5.561469238012977
#std i: 6.038836684179815
#mean i: -8.156642582592495
#std i: 5.540267857407978
#mean i: -10.184306509127808
#std i: 5.94922426336802
#mean i: -13.321271936179238
#std i: 5.9982915754899775
#mean i: -16.011040721816318
#std i: 6.2753451538227125
#mean i: -16.456488564880054
#std i: 6.675200986120497

#mean ti: -239.7192339261295
#std ti: 157.01282686807355
#mean ti: -257.38524885328496
#std ti: 127.42449774066962
#mean ti: -248.10056225792587
#std ti: 110.11424545808579
#mean ti: -223.9445839131965
#std ti: 89.73270766874018
#mean ti: -185.06292718322916
#std ti: 77.18090375146328
#mean ti: -139.13433854384405
#std ti: 52.34608438840117

#get spread in uncertainty
from sklearn.utils import resample
mmean = []
sstd = []
mmeant = []
sstdt = []
for i in range(10):
    boot = resample(dif_ch, replace=True, n_samples=10000)
    boot1 = resample(np.asarray(dif_t65)*1000, replace=True, n_samples=10000)
#    print(boot)
    mmean.append(np.mean(boot))
    sstd.append(np.sqrt(np.var(boot)))
    mmeant.append(np.mean(boot1))
    sstdt.append(np.sqrt(np.var(boot1)))

print(np.mean(mmean))#-13.029840068866585
print(np.sqrt(np.var(mmean)))#0.05508656570246369

print(np.mean(sstd))#6.62736430155135
print(np.sqrt(np.var(sstd)))#0.035464902570179244

print(np.mean(mmeant))#-204.72553933485585
print(np.sqrt(np.var(mmeant)))#0.42908994945936324

print(np.mean(sstdt))#96.65291004849078
print(np.sqrt(np.var(sstdt)))#0.7486244935779546
  

e_range = [90., 120.,150., 200., 300., 400.,600.]
diff_amp65 = [[] for i in range(6)]
diff_t65 = [[] for i in range(6)]

for i in range(len(e_range)-1):
    for j in range(len(g_res65[:,1])):
        if e_range[i] < g_res65[j,1]*477.3/8816 < e_range[i+1]:
            diff_amp65[i].append((g_res65[j,0] - g_res65[j,1])*477.3/8816)
            diff_t65[i].append((t_res65[j,0] - t_res65[j,1])*1000)

diff_tt65 = [[] for i in range(6)]
for i in range(len(e_range)-1):
    for j in range(len(diff_t65[i])):
        if np.mean(diff_t65[i])-600 < diff_t65[i][j] < np.mean(diff_t65[i])+600:
            diff_tt65[i].append(diff_t65[i][j])
        

for i in range(len(e_range)-1):
#    print('mean i:', np.mean(diff_amp50[i]))
#    print('std i:', np.sqrt(np.var(diff_amp50[i])))
    print('mean ti:', np.mean(diff_tt65[i]))
    print('std ti:', np.sqrt(np.var(diff_tt65[i])))
    
#    plt.hist(diff_amp50[i], bins=60, color='gray')
#    plt.xlim(np.mean(diff_amp50[i])-23, np.mean(diff_amp50[i])+23)
    textstr = '\n'.join((
        r'$\mu=%.1f$ ps' % (np.mean(diff_tt65[i]), ), 
        r'$\sigma=%.1f$ ps' % ( np.sqrt(np.var(diff_tt65[i])), )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', alpha=0.5)
    
    # place a text box in upper left in axes coords

    if i <2:
        fig, ax = plt.subplots()
        ax.hist(diff_tt65[i], bins=35, color='gray')
        plt.xlim(np.mean(diff_tt65[i])-500, np.mean(diff_tt65[i])+600)
        plt.xlabel('original timing - recovered timing (ps)')

    elif i ==2:
        fig, ax = plt.subplots()
        ax.hist(diff_tt65[i], bins=59, color='gray')
        plt.xlim(np.mean(diff_tt65[i])-350, np.mean(diff_tt65[i])+350)
        plt.xlabel('original timing - recovered timing (ps)')

    elif i ==5:
        fig, ax = plt.subplots()
        ax.hist(diff_tt65[i], bins=60, color='gray')
        plt.xlim(np.mean(diff_tt65[i])-210, np.mean(diff_tt65[i])+210)
        plt.xlabel('original timing - recovered timing (ps)')

#    elif i ==1 or i==2:
#        fig, ax = plt.subplots()
#        ax.hist(diff_t50[i], bins=35, color='gray')
#        plt.xlim(np.mean(diff_t50[i])-380, np.mean(diff_t50[i])+380)
    else:
        fig, ax = plt.subplots()
        ax.hist(diff_tt65[i], bins=58, color='gray')#, label = 'mean: '+str(np.mean(diff_t50[i]))
        plt.xlim(np.mean(diff_tt65[i])-300, np.mean(diff_tt65[i])+300)
        plt.xlabel('original timing - recovered timing (ps)')
#        plt.legend(fontsize = 12)
        
    ax.text(0.02, 0.85, textstr, fontsize=16,
             transform=ax.transAxes)#, bbox=props
    savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\time65\time' + str(e_range[i+1]) )


#energy
    
diff_ampp65 = [[] for i in range(6)]
for i in range(len(e_range)-1):
    for j in range(len(diff_amp65[i])):
        if i < 4:
            if np.mean(diff_amp65[i])-20 < diff_amp65[i][j] < np.mean(diff_amp65[i])+20:
                diff_ampp65[i].append(diff_amp65[i][j])
        else:
            if np.mean(diff_amp65[i])-30 < diff_amp65[i][j] < np.mean(diff_amp65[i])+30:
                diff_ampp65[i].append(diff_amp65[i][j])            

for i in range(len(e_range)-1):
    print('mean i:', np.mean(diff_ampp65[i]))
    print('std i:', np.sqrt(np.var(diff_ampp65[i])))
#    print('mean ti:', np.mean(diff_tt50[i]))
#    print('std ti:', np.sqrt(np.var(diff_tt50[i])))
    
#    plt.hist(diff_amp50[i], bins=60, color='gray')
#    plt.xlim(np.mean(diff_amp50[i])-23, np.mean(diff_amp50[i])+23)
    textstr = '\n'.join((
        r'$\mu=%.1f$ keV' % (np.mean(diff_ampp65[i]), ), 
        r'$\sigma=%.1f$ keV' % ( np.sqrt(np.var(diff_ampp65[i])), )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', alpha=0.5)
    
    # place a text box in upper left in axes coords

    if i <6:
        fig, ax = plt.subplots()
        ax.hist(diff_ampp65[i], bins=35, color='gray')
        plt.xlabel('original peak - recovered peak \n (keV)')


    ax.text(0.02, 0.85, textstr, fontsize=16,
             transform=ax.transAxes)#, bbox=props
    savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\charge65\charge' + str(e_range[i+1]) )


        



#plt.hist(diff_t50[1],200)


#mean i: -5.94784897893005
#std i: 4.743662958850826
#mean i: -8.296297679809623
#std i: 4.833995833597844
#mean i: -10.34773105792776
#std i: 4.990409462884803
#mean i: -13.416153876394251
#std i: 5.2659402479233535
#mean i: -16.10174142091928
#std i: 5.688631237037445
#mean i: -16.496726178326433
#std i: 6.421638874383728

#mean ti: -241.97780235206415
#std ti: 152.21852769670463
#mean ti: -257.9970888595103
#std ti: 125.34857688013705
#mean ti: -249.14643176586708
#std ti: 98.90918225149821
#mean ti: -225.1742959672462
#std ti: 73.55969591420234
#mean ti: -187.01953324818595
#std ti: 56.47916307087871
#mean ti: -139.37969964875305
#std ti: 47.36034807474935
    

#plot 2d hist
plt.hist2d((g_res65[:,0]-g_res65[:,1])*477.3/8816, g_res65[:,1]*477.3/8816, bins=220)
plt.xlabel('original peak - recovered peak \n (keV)')
plt.ylabel('recovered peak \n (keV)')
plt.tight_layout()
plt.xlim(-40,10)
plt.ylim(70,700)
plt.colorbar()

savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\charge65\chch65_2d')

plt.hist2d((t_res65[:,0] - t_res65[:,1])*1000, g_res65[:,1]*477.3/8816, bins=220)
plt.xlabel('original timing - recovered timing \n (ps)')
plt.ylabel('recovered energy \n (keV)')
plt.tight_layout()
plt.colorbar()
plt.xlim(-600,150)
plt.ylim(70,700)


savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\time65\t65_2d')

    



#55 MHz
      
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.interpolate import UnivariateSpline
N, Wn = signal.buttord((240/500),(275/500) , 3, 10) #(240/500),(290/500) , 3, 10, charge        (130/500),(170/500) , 3, 10, timing
b, a = signal.butter(N, Wn, 'low')

#h50_f = np.fft.fft(h50_t1[0:1050])
#rec_50 = np.zeros((len(res_50),1050))
#for i in range(len(res_50)):
#    out_x =  np.fft.fft(np.lib.pad(res_50[i,0], (0,50), 'constant', constant_values=(0., 0.)))/h50_f
#    out_xn = np.real(np.fft.ifft(out_x))
#    rec_50[i] = scipy.signal.filtfilt(b, a, out_xn)
    
from collections import deque 
h55_f = np.fft.fft(h55_t[0:1000])
rec_55 = np.zeros((len(res_55),1000))
for i in range(len(res_55)):
    out_x =  np.fft.fft(res_55[i,0])/h55_f#np.lib.pad(res_55[i,0], (0,100), 'constant', constant_values=(0., 0.))
    out_xn = np.real(np.fft.ifft(out_x))
    rec_55[i] = scipy.signal.filtfilt(b, a, out_xn)
    dd= deque(rec_55[i])
    dd.rotate(32)
    rec_55[i] = dd



import copy
g_res55=[]
t_res55 = []
rres_55 = copy.deepcopy(res_55)
rrec_55 = copy.deepcopy(rec_55)
from scipy.interpolate import UnivariateSpline
max_res55 = np.zeros((len(rres_55),2))
for i in range(len(rec_55)):#len(rec_50)
    max_arg1 = 100 + np.argmax(rres_55[i,1,100:900])
    abcissa1 = np.asarray([ind for ind in range(max_arg1-4, max_arg1+3)])
    ordinate1 = rres_55[i,1,abcissa1]
    spl1 = UnivariateSpline(abcissa1, ordinate1,k=3)
    xs1 = np.linspace(max_arg1-4, max_arg1+3,100)
    max_res55[i,0] = np.max(spl1(xs1))
#    max_res50[i,0] = np.max(rres_50[i,1,100:900])

    max_arg2 = 100 + np.argmax(rec_55[i, 100:900])
    abcissa2 = np.asarray([ind for ind in range(max_arg2-4, max_arg2+3)])
    ordinate2 = rrec_55[i,abcissa2]
    spl2 = UnivariateSpline(abcissa2, ordinate2,k=3)
    xs2 = np.linspace(max_arg2-4, max_arg2+3,100)
    max_res55[i,1] = np.max(spl2(xs2))
#    max_res50[i,1] = np.max(rec_50[i, 100:900])
    
    
    if 3000<np.max(spl1(xs1)) < 17000.:# and -15. < (max_res55[i,0] - max_res55[i,1])/2**16*1000 < 10.:
        g_temp =[]
        g_temp.append(max_res55[i,0])
        g_temp.append(max_res55[i,1])
        g_res55.append(g_temp)
        
#        t_temp = []
#        for ii in range(len(xs1)):
#            if spl1(xs1)[ii] <= max_res50[i,0]/2 < spl1(xs1)[ii+1]:
#                yy1 = spl1(xs1)[ii]
#                xx1 = xs1[ii]
#                yy2 = spl1(xs1)[ii+1]
#                xx2 = xs1[ii+1]
#                break
#        slope = (yy1 - yy2) / (xx1 - xx2)
#        intrcept = yy1 - (slope*xx1) 
#        t_temp.append(((max_res50[i,0]/2) - intrcept)/slope )
#    
#        for ii in range(len(xs2)):
#            if spl2(xs2)[ii] <= max_res50[i,1]/2 < spl2(xs2)[ii+1]:
#                yy1 = spl2(xs2)[ii]
#                xx1 = xs2[ii]
#                yy2 = spl2(xs2)[ii+1]
#                xx2 = xs2[ii+1]
#                break
#        slope = (yy1 - yy2) / (xx1 - xx2)
#        intrcept = yy1 - (slope*xx1) 
#        t_temp.append(((max_res50[i,1]/2) - intrcept)/slope )
#            
#        t_res50.append(t_temp)
##        print (i, max_res50[i,0], max_res50[i,1])
##        break
#    
        abc = np.asarray([ind for ind in range(max_arg1-4, max_arg1+1)])
        ordi = rres_55[i,1,abc]
        t_temp = []
        for ii in range(len(abc)-1):
            if ordi[ii] < max_res55[i,0]/2 < ordi[ii+1]:
                yy1 = ordi[ii]
                xx1 = abc[ii]
                yy2 = ordi[ii+1]
                xx2 = abc[ii+1]
                break
            
        slope = (yy1 - yy2) / (xx1 - xx2)
        intrcept = yy1 - (slope*xx1) 
        t_temp.append(((max_res55[i,0]/2) - intrcept)/slope )
    
    
        abc = np.asarray([ind for ind in range(max_arg2-4, max_arg2+1)])
        ordi = rrec_55[i,abc]
        for ii in range(len(abc)-1):
            if ordi[ii] < max_res55[i,1]/2 < ordi[ii+1]:
                yy1 = ordi[ii]
                xx1 = abc[ii]
                yy2 = ordi[ii+1]
                xx2 = abc[ii+1]
                break
            
        slope = (yy1 - yy2) / (xx1 - xx2)
        intrcept = yy1 - (slope*xx1) 
        t_temp.append(((max_res55[i,1]/2) - intrcept)/slope )
        
    
        t_res55.append(t_temp)

#    if t_res50[i,0] - t_res50[i,1] < -2500:
#        plt.plot(rres_50[i,1]/2**16*1000)
#        plt.plot(rrec_50[i]/2**16*1000)    
#        break
#                
                

g_res55 = np.asarray(g_res55)
t_res55 = np.asarray(t_res55)


for i in range(len(res_50)):
    if 1500<max_res50[i,0]<2000:
        
        plt.figure()
        plt.plot(rres_50[i,1]/2**16*1000)
        plt.plot(rrec_50[i]/2**16*1000)
        break
#        plt.plot(rres_50[i,0]/2**16*1000)

for i in range(len(res_50)):
    if -0.39> t_res50[i,0] - t_res50[i,1] >-0.4:
        
        plt.figure()
        plt.plot(rres_50[i,1]/2**16*1000)
        plt.plot(rrec_50[i]/2**16*1000)
        break       
    

plt.hist((g_res55[:,0] - g_res55[:,1])/2**16*1000,180, color='gray')
plt.xlim(-12,15)
plt.ylabel('counts')
plt.xlabel('original peak - recovered peak \n (mV)')
plt.title('55 MHz', fontsize=18)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\plots\chch55')

dif_ch55 = []
for i in range(len(g_res55)):
    if -12. < (g_res55[i,0] - g_res55[i,1])/2**16*1000 < 15:
        dif_ch65.append((g_res55[i,0] - g_res55[i,1])/2**16*1000)

print(np.mean(dif_ch65))#-2.6520885703296457
print(np.sqrt(np.var(dif_ch65)))#2.729525297644874


plt.hist((t_res55[:,0] - t_res55[:,1])*1000,80, color='gray')
plt.xlim(-3000,2000)
plt.ylabel('counts')
plt.xlabel('original timing - recovered timing \n (ps)')
plt.title('55 MHz', fontsize=18)
print(np.mean((t_res55[:,0] - t_res55[:,1])))#-4.252257682191539
print(np.sqrt(np.var((t_res55[:,0] - t_res55[:,1]))))#2.4252711802330493
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\plots\t55')

dif_t55 = []
for i in range(len(t_res55)):
    if -12. < (t_res55[i,0] - t_res55[i,1]) < 5:
        dif_t65.append(t_res55[i,0] - t_res55[i,1])

print(np.mean(dif_t55)*1000)#186.60861125936148
print(np.sqrt(np.var(dif_t55))*1000)#75.20319559761519

plt.figure()    
plt.hist((max_res50[:,0] - max_res50[:,1]),1800,alpha=0.6)#/2**16*1000
plt.xlim(-6,15)

plt.hist(max_res60[:,0]/2**16*1000,200,alpha=0.7)
plt.hist(max_res60[:,1]/2**16*1000,200,alpha=0.6)


#plot multiple pulses

for i in range(40,len(res_50)):
    if 8400<np.max(res_50[i,1,100:900])<8500:
        
        plt.figure()
        plt.plot(res_50[i,1]/2**16*1000)
        plt.plot(rec_50[i]/2**16*1000)
        plt.show()
        print(i)#587
        break

for i in range(40,len(res_55)):
    if 8400<np.max(res_55[i,1,100:900])<8500:
        
        plt.figure()
        plt.plot(res_55[i,1]/2**16*1000)
        plt.plot(rec_55[i]/2**16*1000)
        plt.show()
        print(i)#3105
        break

for i in range(1358,len(res_60)):
    if 8400<np.max(res_60[i,1,100:900])<8500:
        
        plt.figure()
        plt.plot(res_60[i,1]/2**16*1000)
        plt.plot(rec_60[i]/2**16*1000)
        plt.show()#146
        print(i)
        break

for i in range(400,len(res_65)):
    if 7900<np.max(res_65[i,1,100:900])<8000:
        
        plt.figure()
        plt.plot(res_65[i,1]/2**16*1000)
        plt.plot(rec_65[i]/2**16*1000)
        plt.show()
        print(i)#135
        break


#multiple pulses, 50 MHz
fig, axs = plt.subplots(4,1)
axs[0].plot(res_50[587,0,100:600])
axs[1].plot(res_55[3105,0,100:600],color='orange')
axs[2].plot(res_60[1395,0,100:600],color='g')
axs[3].plot(res_65[429,0,100:600],color='r')
axs[3].tick_params(axis="x", labelsize=11)
axs[3].tick_params(axis="y", labelsize=11)
axs[2].tick_params(axis="y", labelsize=11)
axs[1].tick_params(axis="y", labelsize=11)
axs[0].tick_params(axis="y", labelsize=11)
axs[0].set_xticks([])
axs[1].set_xticks([])
axs[2].set_xticks([])
plt.xlabel("sample number (ns)")
fig.text(-0.01,0.5, "sample value (ADC units)", ha="center", va="center", rotation=90,size= 16)
axs[2].set_yticks([0,-2500])
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\mult4')

plt.show()

plt.figure()
plt.plot(f0[30:90],np.abs(np.fft.fft(res_50[587,0]))[30:90])
plt.plot(f0[30:90],np.abs(np.fft.fft(res_55[3105,0]))[30:90],color='orange')
plt.plot(f0[30:90],np.abs(np.fft.fft(res_60[1395,0]))[30:90],color='g')
plt.plot(f0[30:90],np.abs(np.fft.fft(res_65[429,0]))[30:90],color='r')
plt.xlabel("frequency (MHz)")
plt.ylabel("amplitude (arb. units)")
plt.xticks(np.arange(30,90,5),fontsize= 11)
plt.yticks(fontsize= 11)

plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\mult4_freq')


#reconstrction/ residual 50 MHz


for i in range(250,len(res_50)):
    if 2450<np.max(res_50[i,1,100:900])<2500:
        
        plt.figure()
        plt.plot(res_50[i,1]/2**16*1000)
        plt.plot(rec_50[i]/2**16*1000)
        plt.show()
        print(i)#795
        break
    
for i in range(2500,len(res_50)):
    if 9400<np.max(res_50[i,1,100:900])<9500:
        
        plt.figure()
        plt.plot(res_50[i,1]/2**16*1000)
        plt.plot(rec_50[i]/2**16*1000)
        plt.show()
        print(i)#2538
        break

for i11 in range(400,len(res_50)):
    if 2350<np.max(res_50[i11,1,100:900])<2400:
        
        plt.figure()
        plt.plot(res_50[i11,1]/2**16*1000)
        plt.plot(rec_50[i11]/2**16*1000)
        plt.show()
        print(i11)#795
        break
    
for i22 in range(2540,len(res_50)):
    if 9400<np.max(res_50[i22,1,100:900])<9500:
        
        plt.figure()
        plt.plot(res_50[i22,1]/2**16*1000)
        plt.plot(rec_50[i22]/2**16*1000)
        plt.show()
        print(i22)#2538
        break

for i22 in range(2540,len(res_50)):
    if 9400<np.max(res_50[i22,1,100:900])<9500:
        
        plt.figure()
        plt.plot(res_50[i22,1]/2**16*1000)
        plt.plot(rec_50[i22]/2**16*1000)
        plt.show()
        print(i22)#2538
        break

for ii22 in range(2540,len(res_50)):
    if 11000<np.max(res_50[ii22,1,100:900])<11200:
        
        plt.figure()
        plt.plot(res_50[ii22,1]/2**16*1000)
        plt.plot(rec_50[ii22]/2**16*1000)
        plt.show()
        print(ii22)#2538
        break

fig, ax = plt.subplots(figsize=[6, 5])
plt.plot(f0[0:500],np.abs(np.fft.fft(res_50[i11,1]))[0:500], label = 'pulse height = 2400 ADC units', alpha=0.7)
plt.plot(f0[0:500],np.abs(np.fft.fft(res_50[i22,1]))[0:500], label = 'pulse height = 9500 ADC units', alpha= 0.7)
plt.xlabel('frequency (MHz)')
plt.ylabel('amplitude (arb. units)')
plt.legend(fontsize=10,loc=1)

# inset axes....
axins = ax.inset_axes([0.4, 0.4, 0.45, 0.45])
axins.plot(f0[0:500],np.abs(np.fft.fft(res_50[i11,1]))[0:500], alpha=0.7)
axins.plot(f0[0:500],np.abs(np.fft.fft(res_50[i22,1]))[0:500], alpha=0.7)

# sub region of the original image
x1, x2, y1, y2 = 235, 500, 25., 3400
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.tick_params(axis="x", labelsize=14)
axins.tick_params(axis="y", labelsize=14)

ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

ax.indicate_inset_zoom(axins)

plt.tight_layout()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\pulse_ht_freq')

#plot autocorrealtion
fig, ax = plt.subplots(figsize=[6, 5])
plt.plot(f0[0:1000],np.abs(np.fft.fft(cor22))[0:1000])
plt.xlabel('frequency (MHz)')
plt.ylabel('amplitude (arb. units)')

# inset axes....
axins = ax.inset_axes([0.4, 0.4, 0.45, 0.45])
axins.plot(f0[0:1000],np.abs(np.fft.fft(cor22))[0:1000])

# sub region of the original image
x1, x2, y1, y2 = 260, 500, 300000, 3000000
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.tick_params(axis="x", labelsize=11)
axins.tick_params(axis="y", labelsize=11)

ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

ax.indicate_inset_zoom(axins)

plt.tight_layout()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second/plots\autocorrelation')



fig, axs = plt.subplots(2,2)
axs[0,0].plot(res_50[795,1,100 + np.argmax(res_50[795,1,100:900]) - 20 : 100 + 
            np.argmax(res_50[795,1,100:900]) + 117], alpha = 0.75,label='original pulse')
#axs[0,0].legend()

axs[0,0].plot(rec_50[795,100 + np.argmax(res_50[795,1,100:900]) - 20 : 100 + 
            np.argmax(res_50[795,1,100:900]) + 117], alpha = 0.75,label='recovered pulse')
axs[0,0].legend(loc = 1, fontsize = 6)
# inset axes....
axins = axs[0,0].inset_axes([0.4, 0.35, 0.4, 0.4])
#axins.plot(res_50[795,1, np.argmax(res_50[795,1]) - 2 : 
#            np.argmax(res_50[795,1]) + 5] , alpha = 0.75)
#axins.plot(rec_50[795, np.argmax(res_50[795,1]) - 2 : 
#            np.argmax(res_50[795,1]) + 5] , alpha = 0.75)
axins.plot(res_50[795,1,100 + np.argmax(res_50[795,1,100:900]) - 20 : 100 + 
            np.argmax(res_50[795,1,100:900]) + 117], alpha = 0.75)
axins.plot(rec_50[795,100 + np.argmax(res_50[795,1,100:900]) - 20 : 100 + 
            np.argmax(res_50[795,1,100:900]) + 117],alpha = 0.75)
axins.tick_params(labelsize = 6)
# sub region of the original image
x1, x2, y1, y2 = 18, 23, 2150, 2650
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axs[0,0].indicate_inset_zoom(axins)

axs[0,1].plot(res_50[795,1,100 + np.argmax(res_50[795,1,100:900]) - 20 : 100 + 
            np.argmax(res_50[795,1,100:900]) + 117] - rec_50[795,100 + np.argmax(res_50[795,1,100:900]) - 20 : 100 + 
            np.argmax(res_50[795,1,100:900]) + 117], alpha = 0.7,label='residual',color='g')
axs[0,1].legend()
axs[0,1].set_ylim(-450,450)
#axs[1,0].plot(res_50[2538,1,100 + np.argmax(res_50[2538,1,100:900]) - 20 : 100 + 
#            np.argmax(res_50[2538,1,100:900]) + 117], alpha = 0.75,label='original pulse')
##axs[1,0].legend()
#axs[1,0].plot(rec_50[2538,100 + np.argmax(res_50[2538,1,100:900]) - 20 : 100 + 
#            np.argmax(res_50[2538,1,100:900]) + 117], alpha = 0.75,label='recovered pulse')
#axs[1,0].legend(loc = 1,fontsize = 6)
## inset axes....
#axins = axs[1,0].inset_axes([0.4, 0.35, 0.4, 0.4])
##axins.plot(res_50[795,1, np.argmax(res_50[795,1]) - 2 : 
##            np.argmax(res_50[795,1]) + 5] , alpha = 0.75)
##axins.plot(rec_50[795, np.argmax(res_50[795,1]) - 2 : 
##            np.argmax(res_50[795,1]) + 5] , alpha = 0.75)
#axins.plot(res_50[2538,1,100 + np.argmax(res_50[2538,1,100:900]) - 20 : 100 + 
#            np.argmax(res_50[2538,1,100:900]) + 117], alpha = 0.75)
#axins.plot(rec_50[2538,100 + np.argmax(res_50[2538,1,100:900]) - 20 : 100 + 
#            np.argmax(res_50[2538,1,100:900]) + 117],alpha = 0.75)
#axins.tick_params(labelsize = 6)
## sub region of the original image
#x1, x2, y1, y2 = 18, 22, 9350, 9808
#axins.set_xlim(x1, x2)
#axins.set_ylim(y1, y2)
#axs[1,0].indicate_inset_zoom(axins)
axs[1,0].plot(res_50[ii22,1,100 + np.argmax(res_50[ii22,1,100:900]) - 20 : 100 + 
            np.argmax(res_50[ii22,1,100:900]) + 117], alpha = 0.75,label='original pulse')
#axs[1,0].legend()
axs[1,0].plot(rec_50[ii22,100 + np.argmax(res_50[ii22,1,100:900]) - 20 : 100 + 
            np.argmax(res_50[ii22,1,100:900]) + 117], alpha = 0.75,label='recovered pulse')
axs[1,0].legend(loc = 1,fontsize = 6)
# inset axes....
axins = axs[1,0].inset_axes([0.4, 0.35, 0.4, 0.4])
#axins.plot(res_50[795,1, np.argmax(res_50[795,1]) - 2 : 
#            np.argmax(res_50[795,1]) + 5] , alpha = 0.75)
#axins.plot(rec_50[795, np.argmax(res_50[795,1]) - 2 : 
#            np.argmax(res_50[795,1]) + 5] , alpha = 0.75)
axins.plot(res_50[ii22,1,100 + np.argmax(res_50[ii22,1,100:900]) - 20 : 100 + 
            np.argmax(res_50[ii22,1,100:900]) + 117], alpha = 0.75)
axins.plot(rec_50[ii22,100 + np.argmax(res_50[ii22,1,100:900]) - 20 : 100 + 
            np.argmax(res_50[ii22,1,100:900]) + 117],alpha = 0.75)
axins.tick_params(labelsize = 6)
# sub region of the original image
x1, x2, y1, y2 = 19, 22, 10700, 11200
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axs[1,0].indicate_inset_zoom(axins)

axs[1,1].plot(res_50[ii22,1,100 + np.argmax(res_50[ii22,1,100:900]) - 20 : 100 + 
            np.argmax(res_50[ii22,1,100:900]) + 117] - rec_50[ii22,100 + np.argmax(res_50[2538,1,100:900]) - 20 : 100 + 
            np.argmax(res_50[ii22,1,100:900]) + 117], alpha = 0.7,label='residual',color='g')
axs[1,1].set_ylim(-450,450)

#axs[1,1].plot(res_50[2538,1,100 + np.argmax(res_50[2538,1,100:900]) - 20 : 100 + 
#            np.argmax(res_50[2538,1,100:900]) + 117] - rec_50[2538,100 + np.argmax(res_50[2538,1,100:900]) - 20 : 100 + 
#            np.argmax(res_50[2538,1,100:900]) + 117], alpha = 0.7,label='residual',color='g')
#axs[1,1].set_ylim(-350,350)
axs[0,0].tick_params(axis="x", labelsize=9)
axs[0,0].tick_params(axis="y", labelsize=9)
axs[0,1].tick_params(axis="x", labelsize=9)
axs[0,1].tick_params(axis="y", labelsize=9)
axs[1,0].tick_params(axis="x", labelsize=9)
axs[1,0].tick_params(axis="y", labelsize=9)
axs[1,1].tick_params(axis="x", labelsize=9)
axs[1,1].tick_params(axis="y", labelsize=9)
axs[1,1].legend()
fig.text(0.5, -0.01, 'sample number (ns)', ha='center',size= 16)
#plt.xlabel("sample number (ns)")
fig.text(-0.01,0.5, "sample value (ADC units)", ha="center", va="center", rotation=90,size= 16)
plt.suptitle('Detector 1, 50 MHz resonator',size= 18)
plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\residual50new')



#rms error 50 MHz   
#11560/477.3*90 = 2179.7611565053426
#11560/477.3*600 = 14531.741043368951
#plt.plot(res_50[8765,1]), plt.plot(rec_50[8765])
#np.sqrt(sum((np.asarray(pulse_50[2][789]) - np.asarray(p_rec_50[2])[789])**2)/len(np.asarray(pulse_50[2][789])))

adc_range = [13157/477.3*110.,13157/477.3*150.,13157/477.3*200., 13157/477.3*250., 13157/477.3*300.,13157/477.3*350, 13157/477.3*390, 13157/477.3*477.3]#, 
adc_mv = np.asarray(adc_range)/2**16*1000
pulse_50 = [[] for i in range(len(adc_range))]
p_rec_50 = [[] for i in range(len(adc_range))]


plt.plot(res_50[j,1, 100 + np.argmax(res_50[j,1,100:900]) - 6 : 100 + 
            np.argmax(res_50[123,1,100:900]) + 117])
plt.plot(rec_50[j,100 + np.argmax(res_50[j,1,100:900]) - 6 : 100 + 
            np.argmax(res_50[123,1,100:900]) + 117])
plt.show()

for i in range(len(adc_range)):
    for j in range(len(rec_50)):
        if adc_range[i]-50 < np.max(res_50[j,1,100:900]) < adc_range[i]+50:
            pulse_50[i].append(res_50[j,1, 100 + np.argmax(res_50[j,1,100:900]) - 7 : 100 + 
            np.argmax(res_50[j,1,100:900]) + 117])
            p_rec_50[i].append(rec_50[j,100 + np.argmax(res_50[j,1,100:900]) - 7 : 100 + 
            np.argmax(res_50[j,1,100:900]) + 117])

from sklearn.metrics import mean_squared_error

rmse = [[] for i in range(len(adc_range))]
for i in range(len(adc_range)):
    for j in range(len(pulse_50[i])):
        rmse[i].append(np.sqrt(mean_squared_error(pulse_50[i][j], p_rec_50[i][j] )))

avg_rmse = []
rmse_std  =[]
for i in range(len(rmse)):
    avg_rmse.append(np.average(rmse[i]))
    rmse_std.append(np.sqrt(np.var(rmse[i])))
    
print(avg_rmse)
print(rmse_std)
#[94.88508791685977, 101.86881277112666, 107.33454555407732, 111.60612794568566, 114.17357509251349, 117.34807141051988, 135.05313471572057]
#[9.351921399162972, 10.533929765319726, 11.915125520165628, 12.025327968762497, 10.204916758021085, 11.330274413265665, 41.12846718536694]
#plot
adc_range1 = ['2600', '3600', '4800', '7200', '9500', '11400']
plt.figure()
plt.errorbar(adc_range, avg_rmse, yerr=np.asarray(rmse_std),elinewidth = 1.5, capsize=3, label = '50 MHz resonator', alpha = 0.6,linewidth = 2)#, ecolor='black'
#plt.errorbar(adc_range, avg_rmse, yerr=np.asarray(rmse_std),elinewidth = 1.5, capsize=3, ecolor='black', label = '50 MHz resonator')#
#plt.errorbar(adc_range, np.asarray(avg_rmse)/np.asarray(adc_range), yerr=np.asarray(rmse_std)/np.asarray(adc_range),elinewidth = 1.5, capsize=3, label = '50 MHz resonator', alpha = 0.6,linewidth = 2)#

#plt.ylim(108,109)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('RMSE (ADC units)', fontsize= 16)
plt.xlabel('pulse height (ADC units)', fontsize= 16)
plt.legend(fontsize=8)
plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\rmse50')

##get spread in uncertainty
#from sklearn.utils import resample
#rmse_i = [[] for i in range(6)]
#rmse_std_i = [[] for i in range(6)]
#
#for i in range(6):
#    for j in range(10):
#        boot = resample(rmse[i], replace=True, n_samples=1000)
#        rmse_i[i].append(np.mean(boot))
#        rmse_std_i[i].append(np.sqrt(np.var(boot)))
#
#rmse_mean = [[] for i in range(6)]
#rmse_std = [[] for i in range(6)]
#for i in range(6):
#    rmse_mean[i] = np.mean(rmse_i[i])
#    rmse_std[i] = np.sqrt(np.var(rmse_std_i[i]))
#    



#60 MHz, noise analysis

#reconstrction/ residual 60 MHz


for i1 in range(568,len(res_60)):
    if 2450<np.max(res_60[i1,1,100:900])<2500:
        
        plt.figure()
        plt.plot(res_60[i1,1]/2**16*1000)
        plt.plot(rec_60[i1]/2**16*1000)
        plt.show()
        print(i1)#485
        break
for i2 in range(2800,len(res_60)):
    if 9400<np.max(res_60[i2,1,100:900])<9500:
        
        plt.figure()
        plt.plot(res_60[i2,1]/2**16*1000)
        plt.plot(rec_60[i2]/2**16*1000)
        plt.show()
        print(i2)#2507
        break

fig, axs = plt.subplots(2,2)
axs[0,0].plot(res_60[i1,1,100 + np.argmax(res_60[i1,1,100:900]) - 20 : 100 + 
            np.argmax(res_60[i1,1,100:900]) + 117], alpha = 0.75,label='original pulse')
axs[0,0].legend()

axs[0,0].plot(rec_60[i1,100 + np.argmax(res_60[i1,1,100:900]) - 20 : 100 + 
            np.argmax(res_60[i1,1,100:900]) + 117], alpha = 0.75,label='recovered pulse')
axs[0,0].legend()
axs[0,1].plot(res_60[i1,1,100 + np.argmax(res_60[i1,1,100:900]) - 20 : 100 + 
            np.argmax(res_60[i1,1,100:900]) + 117] - rec_60[i1,100 + np.argmax(res_60[i1,1,100:900]) - 20 : 100 + 
            np.argmax(res_60[i1,1,100:900]) + 117], alpha = 0.7,label='residual',color='g')
axs[0,1].legend()
axs[0,1].set_ylim(-350,350)
axs[1,0].plot(res_60[i2,1,100 + np.argmax(res_60[i2,1,100:900]) - 20 : 100 + 
            np.argmax(res_60[i2,1,100:900]) + 117], alpha = 0.75,label='original pulse')
axs[1,0].legend()
axs[1,0].plot(rec_60[i2,100 + np.argmax(res_60[i2,1,100:900]) - 20 : 100 + 
            np.argmax(res_60[i2,1,100:900]) + 117], alpha = 0.75,label='recovered pulse')
axs[1,0].legend()
axs[1,1].plot(res_60[i2,1,100 + np.argmax(res_60[i2,1,100:900]) - 20 : 100 + 
            np.argmax(res_60[i2,1,100:900]) + 117] - rec_60[i2,100 + np.argmax(res_60[i2,1,100:900]) - 20 : 100 + 
            np.argmax(res_60[i2,1,100:900]) + 117], alpha = 0.7,label='residual',color='g')
axs[1,1].set_ylim(-350,350)
axs[0,0].tick_params(axis="x", labelsize=9)
axs[0,0].tick_params(axis="y", labelsize=9)
axs[0,1].tick_params(axis="x", labelsize=9)
axs[0,1].tick_params(axis="y", labelsize=9)
axs[1,0].tick_params(axis="x", labelsize=9)
axs[1,0].tick_params(axis="y", labelsize=9)
axs[1,1].tick_params(axis="x", labelsize=9)
axs[1,1].tick_params(axis="y", labelsize=9)
plt.legend()
fig.text(0.5, -0.01, 'sample number (ns)', ha='center',size= 16)
#plt.xlabel("sample number (ns)")
fig.text(-0.01,0.5, "sample value (ADC units)", ha="center", va="center", rotation=90,size= 16)
plt.suptitle('Detector 3, 60 MHz resonator',size= 18)
plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\residual60')



#rms error 60 MHz   
#11560/477.3*90 = 2179.7611565053426
#11560/477.3*600 = 14531.741043368951
#plt.plot(res_50[8765,1]), plt.plot(rec_50[8765])
#np.sqrt(sum((np.asarray(pulse_50[2][789]) - np.asarray(p_rec_50[2])[789])**2)/len(np.asarray(pulse_50[2][789])))

adc_range60 = [11560/477.3*110.,11560/477.3*150., 11560/477.3*210., 11560/477.3*250., 11560/477.3*300., 11560/477.3*350., 11560/477.3*410., 11560/477.3*482]#, 
adc_mv = np.asarray(adc_range)/2**16*1000
pulse_60 = [[] for i in range(len(adc_range60))]
p_rec_60 = [[] for i in range(len(adc_range60))]

plt.figure()
plt.plot(res_60[j,1, 100 + np.argmax(res_60[j,1,100:900]) - 6 : 100 + 
            np.argmax(res_60[123,1,100:900]) + 117])
plt.plot(rec_60[j,100 + np.argmax(res_60[j,1,100:900]) - 6 : 100 + 
            np.argmax(res_60[123,1,100:900]) + 117])
plt.show()

for i in range(len(adc_range60)):
    for j in range(len(rec_60)):
        if adc_range60[i]-50 < np.max(res_60[j,1,100:900]) < adc_range60[i]+50:
            pulse_60[i].append(res_60[j,1, 100 + np.argmax(res_60[j,1,100:900]) - 7 : 100 + 
            np.argmax(res_60[j,1,100:900]) + 117])
            p_rec_60[i].append(rec_60[j,100 + np.argmax(res_60[j,1,100:900]) - 7 : 100 + 
            np.argmax(res_60[j,1,100:900]) + 117])

from sklearn.metrics import mean_squared_error

rmse = [[] for i in range(len(adc_range60))]
for i in range(len(adc_range60)):
    for j in range(len(pulse_60[i])):
        rmse[i].append(np.sqrt(mean_squared_error(pulse_60[i][j], p_rec_60[i][j] )))

avg_rmse60 = []
rmse_std60  =[]
for i in range(len(rmse)):
    avg_rmse60.append(np.average(rmse[i]))
    rmse_std60.append(np.sqrt(np.var(rmse[i])))
    
print(avg_rmse60)
print(rmse_std60)
#[97.24800474830019, 103.01104920465602, 111.2466101115908, 115.12815567502608, 118.62780208048795, 121.40724018626051, 124.32312525446305, 133.27362056873537]
#[9.321237246882482, 10.534222293409414, 12.08303170510503, 11.13016914299969, 10.743403664137022, 18.173355794052263, 27.664177106466916, 58.388985241638956]
#plot
adc_range1 = ['2600', '3600', '4800', '7200', '9500', '11400']
plt.figure()
plt.errorbar(adc_range60, avg_rmse60, yerr=np.asarray(rmse_std60),elinewidth = 1.5, capsize=3, label = '60 MHz resonator',alpha=0.6,linewidth = 2)#, ecolor='black'
#plt.errorbar(adc_range60, avg_rmse60, yerr=np.asarray(rmse_std60),elinewidth = 1.5, capsize=3, ecolor='black', label = '60 MHz resonator')#
plt.errorbar(adc_range60, np.asarray(avg_rmse60)/np.asarray(adc_range60), yerr=np.asarray(rmse_std60)/np.asarray(adc_range60),elinewidth = 1.5, capsize=3, label = '60 MHz resonator', alpha = 0.6,linewidth = 2)#

#plt.ylim(108,109)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('RMSE (arb. units.)', fontsize= 16)
plt.xlabel('pulse height (ADC units)', fontsize= 16)
plt.legend(fontsize=8)
plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\rmse60')


#65 MHz, noise analysis

#reconstrction/ residual 65 MHz


for i1 in range(760,len(res_65)):
    if 2450<np.max(res_65[i1,1,100:900])<2500:
        
        plt.figure()
        plt.plot(res_65[i1,1]/2**16*1000)
        plt.plot(rec_65[i1]/2**16*1000)
        plt.show()
        print(i1)#485
        break
for i2 in range(4200,len(res_65)):
    if 9400<np.max(res_65[i2,1,100:900])<9500:
        
        plt.figure()
        plt.plot(res_65[i2,1]/2**16*1000)
        plt.plot(rec_65[i2]/2**16*1000)
        plt.show()
        print(i2)#2507
        break

fig, axs = plt.subplots(2,2)
axs[0,0].plot(res_65[i1,1,100 + np.argmax(res_65[i1,1,100:900]) - 20 : 100 + 
            np.argmax(res_65[i1,1,100:900]) + 117], alpha = 0.75,label='original pulse')
axs[0,0].legend()

axs[0,0].plot(rec_65[i1,100 + np.argmax(res_65[i1,1,100:900]) - 20 : 100 + 
            np.argmax(res_65[i1,1,100:900]) + 117], alpha = 0.75,label='recovered pulse')
axs[0,0].legend()
axs[0,1].plot(res_65[i1,1,100 + np.argmax(res_65[i1,1,100:900]) - 20 : 100 + 
            np.argmax(res_65[i1,1,100:900]) + 117] - rec_65[i1,100 + np.argmax(res_65[i1,1,100:900]) - 20 : 100 + 
            np.argmax(res_65[i1,1,100:900]) + 117], alpha = 0.7,label='residual',color='g')
axs[0,1].legend()
axs[0,1].set_ylim(-420,350)
axs[1,0].plot(res_65[i2,1,100 + np.argmax(res_65[i2,1,100:900]) - 20 : 100 + 
            np.argmax(res_65[i2,1,100:900]) + 117], alpha = 0.75,label='original pulse')
axs[1,0].legend()
axs[1,0].plot(rec_65[i2,100 + np.argmax(res_65[i2,1,100:900]) - 20 : 100 + 
            np.argmax(res_65[i2,1,100:900]) + 117], alpha = 0.75,label='recovered pulse')
axs[1,0].legend()
axs[1,1].plot(res_65[i2,1,100 + np.argmax(res_65[i2,1,100:900]) - 20 : 100 + 
            np.argmax(res_65[i2,1,100:900]) + 117] - rec_65[i2,100 + np.argmax(res_65[i2,1,100:900]) - 20 : 100 + 
            np.argmax(res_65[i2,1,100:900]) + 117], alpha = 0.7,label='residual',color='g')
axs[1,1].set_ylim(-420,350)
axs[0,0].tick_params(axis="x", labelsize=9)
axs[0,0].tick_params(axis="y", labelsize=9)
axs[0,1].tick_params(axis="x", labelsize=9)
axs[0,1].tick_params(axis="y", labelsize=9)
axs[1,0].tick_params(axis="x", labelsize=9)
axs[1,0].tick_params(axis="y", labelsize=9)
axs[1,1].tick_params(axis="x", labelsize=9)
axs[1,1].tick_params(axis="y", labelsize=9)
plt.legend()
fig.text(0.5, -0.01, 'sample number (ns)', ha='center',size= 16)
#plt.xlabel("sample number (ns)")
fig.text(-0.01,0.5, "sample value (ADC units)", ha="center", va="center", rotation=90,size= 16)
plt.suptitle('Detector 4, 65 MHz resonator',size= 18)
plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\residual65')


#rms error 65 MHz   
#11560/477.3*90 = 2179.7611565053426
#11560/477.3*600 = 14531.741043368951
#plt.plot(res_50[8765,1]), plt.plot(rec_50[8765])
#np.sqrt(sum((np.asarray(pulse_50[2][789]) - np.asarray(p_rec_50[2])[789])**2)/len(np.asarray(pulse_50[2][789])))

adc_range65 = [8816/477.3*110.,8816/477.3*150., 8816/477.3*210., 8816/477.3*240., 8816/477.3*290., 8816/477.3*365., 8816/477.3*420., 8816/477.3*520]#, 
adc_mv = np.asarray(adc_range)/2**16*1000
pulse_65 = [[] for i in range(len(adc_range65))]
p_rec_65 = [[] for i in range(len(adc_range65))]

plt.figure()
plt.plot(res_65[j,1, 100 + np.argmax(res_65[j,1,100:900]) - 6 : 100 + 
            np.argmax(res_65[123,1,100:900]) + 117])
plt.plot(rec_65[j,100 + np.argmax(res_65[j,1,100:900]) - 6 : 100 + 
            np.argmax(res_65[123,1,100:900]) + 117])
plt.show()

for i in range(len(adc_range65)):
    for j in range(len(rec_65)):
        if adc_range65[i]-50 < np.max(res_65[j,1,100:900]) < adc_range65[i]+50:
            pulse_65[i].append(res_65[j,1, 100 + np.argmax(res_65[j,1,100:900]) - 7 : 100 + 
            np.argmax(res_65[j,1,100:900]) + 117])
            p_rec_65[i].append(rec_65[j,100 + np.argmax(res_65[j,1,100:900]) - 7 : 100 + 
            np.argmax(res_65[j,1,100:900]) + 117])

from sklearn.metrics import mean_squared_error

rmse = [[] for i in range(len(adc_range65))]
for i in range(len(adc_range65)):
    for j in range(len(pulse_65[i])):
        rmse[i].append(np.sqrt(mean_squared_error(pulse_65[i][j], p_rec_65[i][j] )))

avg_rmse65 = []
rmse_std65  =[]
for i in range(len(rmse)):
    avg_rmse65.append(np.average(rmse[i]))
    rmse_std65.append(np.sqrt(np.var(rmse[i])))
    
print(avg_rmse65)
print(rmse_std65)
#[97.24800474830019, 103.01104920465602, 111.2466101115908, 115.12815567502608, 118.62780208048795, 121.40724018626051, 124.32312525446305, 133.27362056873537]
#[9.321237246882482, 10.534222293409414, 12.08303170510503, 11.13016914299969, 10.743403664137022, 18.173355794052263, 27.664177106466916, 58.388985241638956]
#plot
adc_range1 = ['2600', '3600', '4800', '7200', '9500', '11400']
plt.figure()
plt.errorbar(adc_range65, avg_rmse65, yerr=np.asarray(rmse_std65),elinewidth = 1.5, capsize=3, label = '65 MHz resonator',alpha=0.6,linewidth = 2)#, ecolor='black'
#plt.errorbar(adc_range65, avg_rmse65, yerr=np.asarray(rmse_std65),elinewidth = 1.5, capsize=3, ecolor='black', label = '65 MHz resonator')#
plt.errorbar(adc_range65, np.asarray(avg_rmse65)/np.asarray(adc_range65), yerr=np.asarray(rmse_std65)/np.asarray(adc_range65),elinewidth = 1.5, capsize=3, label = '65 MHz resonator', alpha = 0.6,linewidth = 2)#

#plt.ylim(108,109)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('RMSE (arb. units.)', fontsize= 16)
plt.xlabel('pulse height (ADC units)', fontsize= 16)
plt.legend(fontsize=8)
plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\rmse65')

savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\rmse_all_1')



#plot impulse response
fig, ax = plt.subplots(figsize=[5, 4])
plt.plot(f0[0:500],h50[0:500])
plt.xlabel('frequency (MHz)')
plt.ylabel('amplitude (arb. units)')
plt.tight_layout()
# inset axes....
axins = ax.inset_axes([0.5, 0.5, 0.49, 0.49])
axins.plot(f0[0:500],h50[0:500])
# sub region of the original image
x1, x2, y1, y2 = 40, 60, 1., 1.8
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

ax.indicate_inset_zoom(axins)

plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\impres50')


fig, ax = plt.subplots(figsize=[5, 4])
plt.plot(f0[0:500],h55[0:500])
plt.xlabel('frequency (MHz)')
plt.ylabel('amplitude (arb. units)')
plt.tight_layout()
# inset axes....
axins = ax.inset_axes([0.5, 0.5, 0.49, 0.49])
axins.plot(f0[0:500],h55[0:500])
# sub region of the original image
x1, x2, y1, y2 = 45, 65, 1.2, 2.14
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

ax.indicate_inset_zoom(axins)

plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\impres55')



fig, ax = plt.subplots(figsize=[5, 4])
plt.plot(f0[0:500],h60[0:500])
plt.xlabel('frequency (MHz)')
plt.ylabel('amplitude (arb. units)')
plt.tight_layout()
# inset axes....
axins = ax.inset_axes([0.5, 0.5, 0.49, 0.49])
axins.plot(f0[0:500],h60[0:500])
# sub region of the original image
x1, x2, y1, y2 = 50, 70, 1.2, 2.2
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

ax.indicate_inset_zoom(axins)

plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\impres60')


fig, ax = plt.subplots(figsize=[5, 4])
plt.plot(f0[0:500],h65[0:500])
plt.xlabel('frequency (MHz)')
plt.ylabel('amplitude (arb. units)')
plt.tight_layout()
# inset axes....
axins = ax.inset_axes([0.5, 0.5, 0.49, 0.49])
axins.plot(f0[0:500],h65[0:500])
# sub region of the original image
x1, x2, y1, y2 = 55, 75, 1.45, 2.65
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

ax.indicate_inset_zoom(axins)

plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\impres65')


#rms error




#60 MHz

import copy
g_res60=[]
rres_60 = copy.deepcopy(res_60)
rrec_60 = copy.deepcopy(rec_60)
from scipy.interpolate import UnivariateSpline
max_res60 = np.zeros((len(rres_60),2))
for i in range(len(rec_60)):#len(rec_50)
    max_arg1 = 100 + np.argmax(rres_60[i,1,100:900])
    abcissa = np.asarray([ind for ind in range(max_arg1-2, max_arg1+3)])
    ordinate = rres_60[i,1,abcissa]
    spl = UnivariateSpline(abcissa, ordinate)
    xs = np.linspace(max_arg1-2, max_arg1+3,100)
    max_res60[i,0] = np.max(spl(xs))

    max_arg2 = 100 + np.argmax(rec_60[i, 100:900])
    abcissa = np.asarray([ind for ind in range(max_arg2-2, max_arg2+3)])
    ordinate = rrec_60[i,abcissa]
    spl = UnivariateSpline(abcissa, ordinate)
    xs = np.linspace(max_arg1-2, max_arg1+3,100)
    max_res60[i,1] = np.max(spl(xs))
    
    if np.max(spl(xs)) < 15000.:
        g_temp =[]
        g_temp.append(max_res60[i,0])
        g_temp.append(max_res60[i,1])
        g_res60.append(g_temp)
#        print (i, max_res50[i,0], max_res50[i,1])
#        break

g_res60 = np.asarray(g_res60)
#    
#    if max_res60[i,0] - max_res60[i,1] > 1000.:
#        print (i, max_res60[i,0], max_res60[i,1])
#        break
plt.figure()
plt.plot(rres_60[23,1])
plt.plot(rrec_60[23])
plt.plot(rres_60[23,0])


plt.hist((g_res60[:,0] - g_res60[:,1])/2**16*1000,400)


plt.figure()    
plt.hist((max_res60[:,0] - max_res60[:,1]),1800)
plt.xlim(-6,15)

plt.hist(max_res60[:,0],200,alpha=0.7)
plt.hist(max_res60[:,1],200,alpha=0.6)

for i in range(len(res_50)):
    if 2500<max_res50[i,0]<3000:
        
        plt.figure()
        plt.plot((rres_50[i,1,120:180] - rrec_50[i,120:180])/2**16*1000)
#        plt.plot(rres_50[i,1]/2**16*1000)
#        plt.plot(rrec_50[i]/2**16*1000)
        break
for i in range(len(res_50)):
    if 8000<max_res50[i,0]<9000:
        plt.plot((rres_50[i,1,120:180] - rrec_50[i,120:180])/2**16*1000)
#        plt.plot(rres_50[i,1]/2**16*1000)
#        plt.plot(rrec_50[i]/2**16*1000)
        break

#
#from scipy.optimize import minimize, rosen, rosen_der
#from scipy import optimize
#def f(x):
#
#    N, Wn = signal.buttord((x[0]/500),(x[1]/500), x[2], x[3])
#    b, a = signal.butter(abs(int(N)), Wn, 'low')
#    if len(a) <2:
#        return 1000000000.
#    else:
#        
#    
#        h50_f = np.fft.fft(h50_t[0:1000])
#        rec_50 = np.zeros((len(res_50),1000))
#        for i in range(len(res_50)):
#            out_x =  np.fft.fft(res_50[i,0])/h50_f
#            out_xn = np.real(np.fft.ifft(out_x))
#            rec_50[i] = scipy.signal.filtfilt(b, a, out_xn)
#            
#        
#        from scipy.interpolate import UnivariateSpline
#        max_res50 = np.zeros((len(rres_50),2))
#        for i in range(len(rec_50)):#len(rec_50)
#            max_arg1 = 100 + np.argmax(rres_50[i,1,100:900])
#            abcissa = np.asarray([ind for ind in range(max_arg1-2, max_arg1+3)])
#            ordinate = rres_50[i,1,abcissa]
#            spl = UnivariateSpline(abcissa, ordinate)
#            xs = np.linspace(max_arg1-2, max_arg1+3,100)
#            max_res50[i,0] = np.max(spl(xs))
#        #    max_res50[i,0] = np.max(rres_50[i,1,100:900])
#        
#            max_arg2 = 100 + np.argmax(rec_50[i, 100:900])
#            abcissa = np.asarray([ind for ind in range(max_arg2-2, max_arg2+3)])
#            ordinate = rrec_50[i,abcissa]
#            spl = UnivariateSpline(abcissa, ordinate)
#            xs = np.linspace(max_arg1-2, max_arg1+3,100)
#            max_res50[i,1] = np.max(spl(xs))
#        
#        dif_res501 = []
#        for i in range(len(rec_50)):
#            if -200. < max_res50[i,0] - max_res50[i,1] < 2000:
#                dif_res501.append(max_res50[i,0] - max_res50[i,1])
#            
#        
#        return np.sqrt(np.var(dif_res501))
#
#x0 = [150, 200, 0.5, 9]
#res_min = minimize(f, x0, method='TNC',bounds = ((140, 190), (160, 250), (0.01, 3), (5, 100)))
#
#res_min = optimize.anneal(f, x0, schedule='boltzmann',
#                          full_output=True, maxiter=500, lower=[140,160,0.01,5],
#                          upper=[190,250,3,100], dwell=250, disp=True)
#
#rranges = (slice(-4, 4, 0.25), slice(-4, 4, 0.25))
#
#fmin = 1000000000.
#for i in range(140,270,20):
#    for j in range(160,300,20):
#        for k in range(1):
#            for l in range(1):
#                if j>k:
#                    x = [i,j,3.,10.]
#                    if f(x)< fmin:
#                        xmin = [i,j,k,l]
#                        fmin = f(x)



import copy
g_res55=[]
rres_55 = copy.deepcopy(res_55)
rrec_55 = copy.deepcopy(rec_55)
from scipy.interpolate import UnivariateSpline
max_res55 = np.zeros((len(rres_55),2))
for i in range(len(rec_55)):#len(rec_50)
    max_arg1 = 100 + np.argmax(rres_55[i,1,100:900])
    abcissa = np.asarray([ind for ind in range(max_arg1-2, max_arg1+3)])
    ordinate = rres_55[i,1,abcissa]
    spl = UnivariateSpline(abcissa, ordinate)
    xs = np.linspace(max_arg1-2, max_arg1+3,100)
    max_res55[i,0] = np.max(spl(xs))
#    max_res50[i,0] = np.max(rres_50[i,1,100:900])

    max_arg2 = 100 + np.argmax(rec_55[i, 100:900])
    abcissa = np.asarray([ind for ind in range(max_arg2-2, max_arg2+3)])
    ordinate = rrec_55[i,abcissa]
    spl = UnivariateSpline(abcissa, ordinate)
    xs = np.linspace(max_arg1-2, max_arg1+3,100)
    max_res55[i,1] = np.max(spl(xs))
#    max_res50[i,1] = np.max(rec_50[i, 100:900])
    
    
    if np.max(spl(xs)) < 17000.:
        g_temp =[]
        g_temp.append(max_res55[i,0])
        g_temp.append(max_res55[i,1])
        g_res55.append(g_temp)
#        print (i, max_res50[i,0], max_res50[i,1])
#        break

g_res55 = np.asarray(g_res55)

for i in range(len(res_55)):
    if 1500<max_res55[i,0]<5000:
        
        plt.figure()
        plt.plot(rres_55[i,1]/2**16*1000)
        plt.plot(rrec_55[i]/2**16*1000)
        break
#        plt.plot(rres_50[i,0]/2**16*1000)

plt.hist((g_res55[:,0] - g_res55[:,1])/2**16*1000,400)
plt.xlim(-15,10)


plt.figure()    
plt.hist((max_res55[:,0] - max_res55[:,1]),1800,alpha=0.6)#/2**16*1000
plt.xlim(-6,15)

plt.hist(max_res55[:,0],200,alpha=0.7)
plt.hist(max_res55[:,1],600,alpha=0.6)




import copy
g_res65=[]
rres_65 = copy.deepcopy(res_65)
rrec_65 = copy.deepcopy(rec_65)
from scipy.interpolate import UnivariateSpline
max_res65 = np.zeros((len(rres_65),2))
for i in range(len(rec_65)):#len(rec_50)
    max_arg1 = 100 + np.argmax(rres_65[i,1,100:900])
    abcissa = np.asarray([ind for ind in range(max_arg1-2, max_arg1+3)])
    ordinate = rres_65[i,1,abcissa]
    spl = UnivariateSpline(abcissa, ordinate)
    xs = np.linspace(max_arg1-2, max_arg1+3,100)
    max_res65[i,0] = np.max(spl(xs))
#    max_res50[i,0] = np.max(rres_50[i,1,100:900])

    max_arg2 = 100 + np.argmax(rec_65[i, 100:900])
    abcissa = np.asarray([ind for ind in range(max_arg2-2, max_arg2+3)])
    ordinate = rrec_65[i,abcissa]
    spl = UnivariateSpline(abcissa, ordinate)
    xs = np.linspace(max_arg1-2, max_arg1+3,100)
    max_res65[i,1] = np.max(spl(xs))
#    max_res50[i,1] = np.max(rec_50[i, 100:900])
    
    
    if np.max(spl(xs)) < 15000.:
        g_temp =[]
        g_temp.append(max_res65[i,0])
        g_temp.append(max_res65[i,1])
        g_res65.append(g_temp)
#        print (i, max_res50[i,0], max_res50[i,1])
#        break

g_res65 = np.asarray(g_res65)

for i in range(len(res_65)):
    if 1500<max_res65[i,0]<5000:
        
        plt.figure()
        plt.plot(rres_65[i,1]/2**16*1000)
        plt.plot(rrec_65[i]/2**16*1000)
        break
#        plt.plot(rres_50[i,0]/2**16*1000)

plt.hist((g_res65[:,0] - g_res65[:,1])/2**16*1000,400)
plt.xlim(-15,10)


plt.figure()    
plt.hist((max_res65[:,0] - max_res65[:,1]),1800,alpha=0.6)#/2**16*1000
plt.xlim(-6,15)

plt.hist(max_res50[:,0],200,alpha=0.7)
plt.hist(max_res50[:,1],600,alpha=0.6)



#last chapter fft
rmse_noise50 = [[],[]]
rmse_noise60 = [[],[]]
rmse_noise65 = [[],[]]

for i in range(10000,20000):
    rmse_noise50[0].append(np.sqrt(np.var(res_50[i,0,0:120])))
    rmse_noise50[1].append(np.sqrt(np.var(res_50[i,1,0:120])))
    rmse_noise60[0].append(np.sqrt(np.var(res_60[i,0,0:120])))
    rmse_noise60[1].append(np.sqrt(np.var(res_60[i,1,0:120])))
    rmse_noise65[0].append(np.sqrt(np.var(res_65[i,0,0:120])))
    rmse_noise65[1].append(np.sqrt(np.var(res_65[i,1,0:120])))

print(np.mean(rmse_noise50[0]))
print(np.mean(rmse_noise50[1]))
print(np.sqrt(np.var(rmse_noise50[0])))
print(np.sqrt(np.var(rmse_noise50[1])))
#28.782442643288466
#23.670044621819578
#2.9283648323860656
#2.7325762211253486


print(np.mean(rmse_noise60[0]))
print(np.mean(rmse_noise60[1]))
print(np.sqrt(np.var(rmse_noise60[0])))
print(np.sqrt(np.var(rmse_noise60[1])))
#28.720090895911177
#25.972572797591795
#3.3400272936637823
#3.3945712583337713


print(np.mean(rmse_noise65[0]))
print(np.mean(rmse_noise65[1]))
print(np.sqrt(np.var(rmse_noise65[0])))
print(np.sqrt(np.var(rmse_noise65[1])))

#28.741369034525892
#25.196324118276205
#3.083622938882828
#3.0021511256846796
sipm_b = np.zeros(120,dtype=complex)
mult_b = np.zeros(120,dtype=complex)
for j in range(120):
    for i in range(10000):
        sipm_b[j] = sipm_b[j] + np.abs(np.fft.fft(res_50[i,1,0:120])[j])
        mult_b[j] = mult_b[j] + np.abs(np.fft.fft(res_50[i,0,0:120])[j])
        

sipm_b1 = np.zeros(120)
mult_b1 = np.zeros(120)
for j in range(120):
    for i in range(10000):
        sipm_b1[j] = sipm_b1[j] + res_50[i,1,j]
        mult_b1[j] = mult_b1[j] + res_50[i,0,j]

#plot
plt.plot(mult_b1/10000, label = 'multiplexed signal baseline')
plt.plot(sipm_b1/10000, label = 'SiPM input signal baseline')
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 10
plt.legend(fontsize = 10)
plt.xlabel('sample number (ns)')
plt.ylabel('sample value (ADC units)')
plt.show()

plt.plot(f01[0:60], np.abs(np.fft.fft ((mult_b1/10000)))[0:60], label = 'multiplexed signal baseline')
plt.plot(f01[0:60], np.abs(np.fft.fft((sipm_b1/10000)))[0:60], label = 'SiPM input signal baseline')
plt.show()

plt.plot(f01[0:60], np.abs(mult_b/10000)[0:60], label = 'multiplexed signal baseline')
plt.plot(f01[0:60], np.abs(sipm_b/10000)[0:60], label = 'SiPM input signal baseline')
plt.legend(fontsize = 10)
plt.xlabel('frequency (MHz)')
plt.ylabel('amplitude (arb. units)')
plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\baselines_freq1')

savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\baselines_avg')


#plot
plt.plot(res_50[80,0,0:220], label = 'multiplexed signal baseline')
plt.plot(res_50[80,1,0:220], label = 'SiPM input signal baseline')
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 10
plt.legend(fontsize = 10)
plt.xlabel('sample number (ns)')
plt.ylabel('sample value (ADC units)')

plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\baselines')




N1 = 120
fs = 1000
f01 = np.arange(N1)
f01 = (fs * 1. /N1) * f01

plt.plot(f01[0:60], np.abs(np.fft.fft(res_50[80,0,0:120]))[0:60], label = 'multiplexed signal baseline')
plt.plot(f01[0:60], np.abs(np.fft.fft(res_50[80,1,0:120]))[0:60], label = 'SiPM input signal baseline')
plt.legend(fontsize = 10)
plt.xlabel('frequency (MHz)')
plt.ylabel('amplitude (arb. units)')
plt.show()
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\plots\baselines_freq')

#cut-off frequency of multiplexer
cf = 1 / np.sqrt((1/500**2) + (1/650**2))
cf1 = 1 / np.sqrt((1/650**2) + (1/650**2))