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

from matplotlib import rcParams
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8
rcParams['font.family'] = 'sans-serif'#sans-serif
#rcParams['font.sans-serif'] = ['Verdana']
#rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = False
rcParams['figure.figsize'] = 5,3#5, 3


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


vch1 = np.zeros((2,4,100000,1024))
tcell = np.zeros((2,100000))
#read board header
with open(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\data_100000', 'rb') as f:
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
        if cvc >99998: #number of events to read (n-1)
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
a00 = np.zeros((4,99998,1000))
#a2 = np.zeros((10000,10000))
chchch = [1,3] 
for i1 in range(4):
    cct =0
    for i in range(99998):
        y1 = np.longdouble(vch1[0,i1,i]) - np.mean(np.longdouble(vch1[0,i1,i,5:105])) #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
        x1 = time_samples[i1,int(tcell[0,i])] #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
        
        #linear interp
        xs = np.arange(15,1015,1.)#160/4000 = 0.04
        f2 = interp1d(x1,y1) #,kind='previous'
        a00[i1,i] = f2(xs)#f2(xs)#- np.mean(f2(xs)),y1[15:1015]


aa00 = np.zeros((1,99998,1000))
for i1 in range(1):
    cct =0
    for i in range(99998):
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
for i in range(99998):
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
    elif 10 + np.argmax(d1[10:500])== 54 or 10 + np.argmax(d1[10:500])== 55 or 10 + np.argmax(d1[10:500])== 56:            
#    if 53 + np.argmax(d1[53:57]) == 54 or 53 + np.argmax(d1[53:57]) == 55:
#        if d1[53 + np.argmax(d1[53:57])] - d1[53 + np.argmax(d1[53:57]) - 1] > 0. and d1[53 + np.argmax(d1[53:57])] - d1[53 + np.argmax(d1[53:57]) - 2] > 5000.:
#            if d1[53 + np.argmax(d1[53:57])] - d1[53 + np.argmax(d1[53:57]) + 1] > 0. and d1[53 + np.argmax(d1[53:57])] - d1[53 + np.argmax(d1[53:57]) + 2] > 5000.:
        res_temp55.append(a00[3,i])
        res_temp55.append(aa00[0,i])#SiPM siganla connected to second board
        res_55.append(res_temp55)


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
res_55 = np.asarray(res_55)
res_60 = np.asarray(res_60)
res_65 = np.asarray(res_65)

        
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
N, Wn = signal.buttord((240/500),(290/500) , 3, 10) #(240/500),(290/500) , 3, 10, charge        (130/500),(170/500) , 3, 10, timing
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


import copy
g_res50=[]
t_res50 = []
rres_50 = copy.deepcopy(res_50)
rrec_50 = copy.deepcopy(rec_50)
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
    
    
    if np.max(spl1(xs1)) < 17000. and -15. < (max_res50[i,0] - max_res50[i,1])/2**16*1000 < 10.:
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
        ordi = rres_50[i,1,abc]
        t_temp = []
        for ii in range(len(abc)-1):
            if ordi[ii] < max_res50[i,0]/2 < ordi[ii+1]:
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
            if ordi[ii] < max_res50[i,1]/2 < ordi[ii+1]:
                yy1 = ordi[ii]
                xx1 = abc[ii]
                yy2 = ordi[ii+1]
                xx2 = abc[ii+1]
                break
            
        slope = (yy1 - yy2) / (xx1 - xx2)
        intrcept = yy1 - (slope*xx1) 
        t_temp.append(((max_res50[i,1]/2) - intrcept)/slope )
        
    
        t_res50.append(t_temp)

#    if t_res50[i,0] - t_res50[i,1] < -2500:
#        plt.plot(rres_50[i,1]/2**16*1000)
#        plt.plot(rrec_50[i]/2**16*1000)    
#        break
#                
                

g_res50 = np.asarray(g_res50)
t_res50 = np.asarray(t_res50)


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
    

plt.hist((g_res50[:,0] - g_res50[:,1])/2**16*1000,80, color='gray')
plt.xlim(-12,5)
plt.ylabel('counts')
plt.xlabel('original peak - recovered peak \n (mV)')
plt.title('50 MHz', fontsize=18)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\plots\chch50')

dif_ch = []
for i in range(len(g_res50)):
    if -12. < (g_res50[i,0] - g_res50[i,1])/2**16*1000 < 5:
        dif_ch.append((g_res50[i,0] - g_res50[i,1])/2**16*1000)

print(np.mean(dif_ch))#-4.2535592588299505
print(np.sqrt(np.var(dif_ch)))#2.469892724011246


plt.hist((t_res50[:,0] - t_res50[:,1])*1000,200, color='gray')
plt.xlim(-50,400)
plt.ylabel('counts')
plt.xlabel('original timing - recovered timing \n (ps)')
plt.title('50 MHz', fontsize=18)
print(np.mean((t_res50[:,0] - t_res50[:,1])))#-4.252257682191539
print(np.sqrt(np.var((t_res50[:,0] - t_res50[:,1]))))#2.4252711802330493
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\plots\t50')

dif_t = []
for i in range(len(t_res50)):
    if -12. < (t_res50[i,0] - t_res50[i,1]) < 5:
        dif_t.append(t_res50[i,0] - t_res50[i,1])

print(np.mean(dif_t)*1000)#193.82665877557594
print(np.sqrt(np.var(dif_t))*1000)#81.1768048614788

plt.figure()    
plt.hist((max_res50[:,0] - max_res50[:,1]),1800,alpha=0.6)#/2**16*1000
plt.xlim(-6,15)

plt.hist(max_res50[:,0],200,alpha=0.7)#/2**16*1000
plt.hist(max_res50[:,1]/2**16*1000,200,alpha=0.6)
    



#60 MHz

      
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.interpolate import UnivariateSpline
N, Wn = signal.buttord((130/500),(170/500) , 3, 10) #(240/500),(290/500) , 3, 10, charge        (130/500),(170/500) , 3, 10, timing
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
    
    
    if 3000<np.max(spl1(xs1)) < 17000. and -15. < (max_res60[i,0] - max_res60[i,1])/2**16*1000 < 10.:
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
plt.xlim(-12,5)
plt.ylabel('counts')
plt.xlabel('original peak - recovered peak \n (mV)')
plt.title('60 MHz', fontsize=18)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\plots\chch60')

dif_ch60 = []
for i in range(len(g_res60)):
    if -12. < (g_res60[i,0] - g_res60[i,1])/2**16*1000 < 5:
        dif_ch60.append((g_res60[i,0] - g_res60[i,1])/2**16*1000)

print(np.mean(dif_ch60))#-3.3499192061413328
print(np.sqrt(np.var(dif_ch60)))#2.18760168712437


plt.hist((t_res60[:,0] - t_res60[:,1])*1000,200, color='gray')
plt.xlim(-50,400)
plt.ylabel('counts')
plt.xlabel('original timing - recovered timing \n (ps)')
plt.title('60 MHz', fontsize=18)
print(np.mean((t_res60[:,0] - t_res60[:,1])))#-4.252257682191539
print(np.sqrt(np.var((t_res60[:,0] - t_res60[:,1]))))#2.4252711802330493
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\plots\t60')

dif_t60 = []
for i in range(len(t_res60)):
    if -12. < (t_res60[i,0] - t_res60[i,1]) < 5:
        dif_t60.append(t_res60[i,0] - t_res60[i,1])

print(np.mean(dif_t60)*1000)#187.63670461713318
print(np.sqrt(np.var(dif_t60))*1000)#90.43995468000999

plt.figure()    
plt.hist((max_res50[:,0] - max_res50[:,1]),1800,alpha=0.6)#/2**16*1000
plt.xlim(-6,15)

plt.hist(max_res60[:,0]/2**16*1000,200,alpha=0.7)
plt.hist(max_res60[:,1]/2**16*1000,200,alpha=0.6)





#65 MHz

      
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.interpolate import UnivariateSpline
N, Wn = signal.buttord((240/500),(290/500) , 3, 10) #(240/500),(290/500) , 3, 10, charge        (130/500),(170/500) , 3, 10, timing
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
max_res65 = np.zeros((len(rres_65),2))
for i in range(len(rec_65)):#len(rec_50)
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
    
    
    if np.max(spl1(xs1)) < 17000. and -15. < (max_res65[i,0] - max_res65[i,1])/2**16*1000 < 10.:
        g_temp =[]
        g_temp.append(max_res65[i,0])
        g_temp.append(max_res65[i,1])
        g_res65.append(g_temp)
        
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
plt.xlim(-12,5)
plt.ylabel('counts')
plt.xlabel('original peak - recovered peak \n (mV)')
plt.title('65 MHz', fontsize=18)
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\plots\chch65')

dif_ch65 = []
for i in range(len(g_res65)):
    if -12. < (g_res65[i,0] - g_res65[i,1])/2**16*1000 < 5:
        dif_ch65.append((g_res65[i,0] - g_res65[i,1])/2**16*1000)

print(np.mean(dif_ch65))#-3.24767331243414
print(np.sqrt(np.var(dif_ch65)))#1.925717930566363


plt.hist((t_res65[:,0] - t_res65[:,1])*1000,200, color='gray')
plt.xlim(-50,400)
plt.ylabel('counts')
plt.xlabel('original timing - recovered timing \n (ps)')
plt.title('65 MHz', fontsize=18)
print(np.mean((t_res65[:,0] - t_res65[:,1])))#-4.252257682191539
print(np.sqrt(np.var((t_res65[:,0] - t_res65[:,1]))))#2.4252711802330493
savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\plots\t65')

dif_t65 = []
for i in range(len(t_res65)):
    if -12. < (t_res65[i,0] - t_res65[i,1]) < 5:
        dif_t65.append(t_res65[i,0] - t_res65[i,1])

print(np.mean(dif_t65)*1000)#186.60861125936148
print(np.sqrt(np.var(dif_t65))*1000)#75.20319559761519

plt.figure()    
plt.hist((max_res50[:,0] - max_res50[:,1]),1800,alpha=0.6)#/2**16*1000
plt.xlim(-6,15)

plt.hist(max_res65[:,0],200,alpha=0.7)#/2**16*1000
plt.hist(max_res65[:,1]/2**16*1000,200,alpha=0.6)




#55 MHz
      
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.interpolate import UnivariateSpline
N, Wn = signal.buttord((240/500),(290/500) , 3, 10) #(240/500),(290/500) , 3, 10, charge        (130/500),(170/500) , 3, 10, timing
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
    dd.rotate(31)
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
    
    
    if np.max(spl(xs)) < 15000.:
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