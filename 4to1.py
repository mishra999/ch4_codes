
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

from matplotlib import rcParams
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['legend.fontsize'] = 8
rcParams['font.family'] = 'sans-serif'#sans-serif
#rcParams['font.sans-serif'] = ['Verdana']
#rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = False
rcParams['figure.figsize'] = 5,3#5, 3
 

#with open("/Users/mishraji/Downloads/Polimi_meshADS/drs4_parser/first", "rb") as f:
#    byte = f.read(16)
#    byte = f.read(4)
#
#    while byte != b"":
#        # Do stuff with byte.
#        byte = f.read(4)
#        
#        
#import numpy as np
#with open('/Users/mishraji/Downloads/Polimi_meshADS/drs4_parser/first', 'rb') as f:
#    params = np.fromfile(f, dtype=np.dtype('S1'), count=1)
#    data = np.fromfile(f, dtype=np.float64, count=1) # I
#
#par = params.decode('UTF-8')

import sys



class drs_fdm_ej204_data:
    def __init__(self,fileName, nboards, nchannels, nrecords):#parser
        self.fileName = fileName
        self.file2Read = open(fileName,'rb')
        self.nchannels = nchannels
        self.nboards = nboards
        self.nrecords = nrecords
        self.samples = 1024
        self.tcell = np.zeros((self.nboards,self.nrecords))
        np.fromfile(self.file2Read, dtype=np.dtype('S4'), count=1)
        np.fromfile(self.file2Read, dtype=np.dtype('S4'), count=1)
        self.time_bins = np.zeros((self.nboards,self.nchannels,1024))
        self.time_samples = np.zeros((self.nboards,self.nchannels,self.samples,self.samples))#channel,trigger cell, timebins
        self.vch1 = np.zeros((self.nboards,4,self.nrecords,self.samples)) #record values

        
        #define data types
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
        
        c1 =0
        while(1):
            b = np.fromfile(self.file2Read, dtype=bh, count=1)
            bb = b['c1'].astype(str)
#            print(b['c2'])
            if bb!='B#':
                #event header found
                self.file2Read.seek(-4,1)
                break
#            print('board serial number',b['c2'])
            for i in range(5):#keep looping for time bins for all channels
                b = np.fromfile(self.file2Read, dtype=ch1, count=1)
                bb = b['c1'].astype(str)
#                print(bb)
                if bb != 'C':
                    self.file2Read.seek(-4,1)
                    break
#                i11 = int(b['c2'])
#            print('found time calibration of channel', i11)
                b = np.fromfile(self.file2Read, dtype=np.float32, count=1024)
                self.time_bins[c1,i] = b
#            print(b)
            c1 +=1
        nb = c1
#    print('number of boards', c1)
    
        cvc = 0 #counter for number of events to read
        while(1): #loop over events
            be = np.fromfile(self.file2Read, dtype=eh, count=1)
            if not be:
                break
#        print('found event', int(be['serial']), int(be['sec']), int(be['millisec']))
            for i1 in range(nb):#number of boards
                b1 = np.fromfile(self.file2Read, dtype=bh, count=1)
                bbb = b1['c1'].astype(str)
                if bbb != 'B#':
                    print('invalid board header....exiting....')
                    sys.exit()
                
                bt = np.fromfile(self.file2Read, dtype=tch, count=1)
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
                    b = np.fromfile(self.file2Read, dtype=ch1, count=1)
                    bb = b['c1'].astype(str)
                    if bb != 'C':
                        self.file2Read.seek(-4,1)
                        break
#                print(b['c2'])
                    ch_ind = int(b['c2'])-1
                    s = np.fromfile(self.file2Read, dtype=np.int32, count=1)#get scaler
                    v = list(np.fromfile(self.file2Read, dtype=np.ushort, count=1024))#get sample value
#                v[:] = [x - np.mean(v[15:1015]) for x in v]
                    self.vch1[i1,ch_ind,cvc] = v 
#                plt.plot(v)
#                print(vch1[ch_ind,cvc])
                
            cvc +=1
            if cvc >self.nrecords-5: #number of events to read (n-1)
                break   
            
        
        #get time arrays for all trigger cells
#        self.time_samples = np.zeros((2,1024,1024))#channel,trigger cell, timebins
        for l2 in range(self.nboards):
            if l2==0:
                for i in range(self.nchannels):#channles
                    for j in range(1024): #trigger cell
                        temptime = np.zeros(1024)
                        for k in range(1024): #timebins
                            q, r = divmod(j+k,1024)
                            if q:
                                temptime[k] = np.sum(time_bins[l2,i,j:(j+k)]) + np.sum(time_bins[l2,i,0:r])
                            else:
                                temptime[k] = np.sum(time_bins[l2,i,j:(j+k)])
                
                        self.time_samples[l2,i,j] = np.copy(temptime)
        
                #time alignment
                for j in range(1024):#trigger cells
                    t1 = 0
                    t2 = 0
                    time1 = self.time_samples[l2,0,j]
                    t1 = time1[(1024-j) % 1024]
                    for ii in range(1,4):
                        time2 = self.time_samples[l2,ii,j]
                        t2 = time2[(1024-j) % 1024]
                    
                        dt = t1 - t2
                        for j1 in range(1024):
                            self.time_samples[l2,ii,j,j1] += dt
            
            if l2==1:
                for i in range(1):# 1 channle on second board
                    for j in range(1024): #trigger cell
                        temptime = np.zeros(1024)
                        for k in range(1024): #timebins
                            q, r = divmod(j+k,1024)
                            if q:
                                temptime[k] = np.sum(time_bins[l2,i,j:(j+k)]) + np.sum(time_bins[l2,i,0:r])
                            else:
                                temptime[k] = np.sum(time_bins[l2,i,j:(j+k)])
                
                        self.time_samples[l2,i,j] = np.copy(temptime)
        
                #using only one channel, no need for time allignment 

    def pulses_baseline_correction(self):
        a00 = np.zeros((self.nboards,self.nchannels,self.nrecords,1000))
        for l2 in range(self.nboards):
            if l2 == 0:
                for i1 in range(self.nchannels):
                    for i in range(self.nrecords):
                        y1 = np.longdouble(self.vch1[l2,i1,i]) - np.mean(np.longdouble(self.vch1[l2,i1,i,5:105])) #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
                        x1 = self.time_samples[l2,i1,int(self.tcell[l2,i])] #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
                        #linear interp
                        xs = np.arange(15,1015,1.)#160/4000 = 0.04
                        f2 = interp1d(x1,y1,kind='previous') #,kind='previous'
                        a00[l2,i1,i] = f2(xs)#f2(xs)#- np.mean(f2(xs)),y1[15:1015]
#                        print(a00[l2,i1,i], i, i1, l2)
                        
            if l2 == 1:
                for i1 in range(1):#self.nchannels
                    for i in range(self.nrecords):
                        y1 = np.longdouble(self.vch1[l2,i1,i]) - np.mean(np.longdouble(self.vch1[l2,i1,i,5:105])) #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
                        x1 = self.time_samples[l2,i1,int(self.tcell[l2,i])] #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
                        
                        #linear interp
                        xs = np.arange(15,1015,1.)#160/4000 = 0.04
                        f2 = interp1d(x1,y1,kind='previous') #,kind='previous'
                        a00[l2,i1,i] = f2(xs)#f2(xs)#- np.mean(f2(xs)),y1[15:1015]
#                        print(a00[l2,i1,i], i, l2)
        return a00
    
    def resonator_select(self, a00):        
        res_50 = []
        res_55 = []
        res_60 = []
        res_65 = []
        r_extra = []
        for i in range(self.nrecords):
            res_temp50 = []
            res_temp55 = []
            res_temp60 = []
            res_temp65 = []
            d1 = np.abs(np.fft.fft(a00[0,3,i]))
            if 10 + np.argmax(d1[10:500])== 48 or 10 + np.argmax(d1[10:500])== 49 or 10 + np.argmax(d1[10:500])== 50 or 10 + np.argmax(d1[10:500])== 51:
                res_temp50.append(a00[0,3,i])
                res_temp50.append(a00[0,0,i])
                res_50.append(res_temp50)
        
            elif 10 + np.argmax(d1[10:500])== 53 or 10 + np.argmax(d1[10:500])== 54 or 10 + np.argmax(d1[10:500])== 55 or 10 + np.argmax(d1[10:500])== 56:            
                res_temp55.append(a00[0,3,i])
                res_temp55.append(a00[1,0,i])#SiPM siganla connected to second board
                res_55.append(res_temp55)
        
            elif 10 + np.argmax(d1[10:500])== 59 or 10 + np.argmax(d1[10:500])== 60 or 10 + np.argmax(d1[10:500])== 61:
        
                res_temp60.append(a00[0,3,i])
                res_temp60.append(a00[0,1,i])
                res_60.append(res_temp60)
        
            elif 10 + np.argmax(d1[10:500])== 64 or 10 + np.argmax(d1[10:500])== 65 or 10 + np.argmax(d1[10:500])== 66:            
                res_temp65.append(a00[0,3,i])
                res_temp65.append(a00[0,2,i])
                res_65.append(res_temp65)
        return np.asarray(res_50), np.asarray(res_55), np.asarray(res_60), np.asarray(res_65)


    def pulse_recovery(self, res_50, res_55, res_60, res_65, h50_t, h55_t, h60_t, h65_t):
        import numpy as np
        from scipy.signal import butter, lfilter, freqz
        import matplotlib.pyplot as plt
        import scipy
        from scipy import signal
        N, Wn = signal.buttord((140/500),(180/500) , 0.5, 10)
        b, a = signal.butter(N, Wn, 'low')   
        #50 MHz resonator
        h50_f = np.fft.fft(h50_t)
        rec_50 = np.zeros((len(res_50),1000))
        for i in range(len(res_50)):
            out_x =  np.fft.fft(res_50[i,0])/h50_f
            out_xn = np.real(np.fft.ifft(out_x))
            rec_50[i] = scipy.signal.filtfilt(b, a, out_xn)
        #55 MHz resonator
        from collections import deque 
        h55_f = np.fft.fft(h55_t)
        rec_55 = np.zeros((len(res_55),1000))
        for i in range(len(res_55)):
            out_x =  np.fft.fft(res_55[i,0])/h55_f
            out_xn = np.real(np.fft.ifft(out_x))
            rec_55[i] = scipy.signal.filtfilt(b, a, out_xn)
            dd= deque(rec_55[i])
            dd.rotate(31)
            rec_55[i] = dd
        #60 MHz resonator
        h60_f = np.fft.fft(h60_t)
        rec_60 = np.zeros((len(res_60),1000))
        for i in range(len(res_60)):
            out_x =  np.fft.fft(res_60[i,0])/h60_f
            out_xn = np.real(np.fft.ifft(out_x))
            rec_60[i] = scipy.signal.filtfilt(b, a, out_xn)
        #65 MHz resonator
        h65_f = np.fft.fft(h65_t)
        rec_65 = np.zeros((len(res_65),1000))
        for i in range(len(res_65)):
            out_x =  np.fft.fft(res_65[i,0])/h65_f
            out_xn = np.real(np.fft.ifft(out_x))
            rec_65[i] = scipy.signal.filtfilt(b, a, out_xn)
        
        return rec_50, rec_55, rec_60, rec_65
    
    def amp_timing(self):
        
                                  
if __name__ == '__main__':

    qq=drs_fdm_ej204_data(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\data_100000',2,4,99995)
    aaa00 = qq.pulses_baseline_correction()
    a50, a55, a60, a65 = qq.resonator_select(aaa00)
    
    h50_t = qq.imp_response()
    h50 = np.abs(np.fft.fft(h50_t))
    plt.figure()
    plt.plot(f0[0:500],np.abs(h50)[0:500])
    plt.xlabel('frequency (MHz)')
    plt.ylabel('amplitude (arb. units)')
    plt.tight_layout()


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

p1_ind = np.zeros(3)
for i1 in range(3):
    for i in range(9999):
        if 500. < np.max(a00[i1,i]) <1000.:#3500,4000
            p1_ind[i1] = i
            plt.plot(a00[i1,i])
            break

p1_ind1 = np.zeros(1)
for i1 in range(1):
    for i in range(999):
        if 1500. < np.max(aa00[i1,i]) <6000.:
            p1_ind1[i1] = i
            plt.plot(aa00[i1,i])
            break

xx_inp_ring = np.fft.fft(a00[3,int(p1_ind[2])])
xx_inp_ring = np.fft.fft(a00[3,int(p1_ind1[0])])

#xx_inp_ring = np.fft.fft(a00[nn,a0_start[nn]:(a0_start[nn]+1500)])
#out_x = xx_inp_ring[0:1500]/hd1[0:1500]
out_x = xx_inp_ring/h65
out_xn = np.real(np.fft.ifft(out_x))
plt.figure()
from matplotlib.legend_handler import HandlerLine2D
#mm, = plt.plot(out_xn[300+500:400+500],'r',label='recovered') #recoverd anode pulse
mm, = plt.plot(out_xn,'r',label='recovered') #recoverd anode pulse
nn, = plt.plot(a00[2,int(p1_ind[2])],'g',label='original')#original anode pulse
#nn, = plt.plot(aa00[0,int(p1_ind1[0])],'g',label='original')#original anode pulse

plt.xlabel('sample number',fontsize=16)
plt.ylabel('sample value',fontsize=16)
plt.legend(loc=4)

plt.figure()
plt.plot(f0[0:500],np.abs(np.fft.fft(out_xn))[0:500])
plt.plot(f0[0:500],np.abs(np.fft.fft(a00[2,int(p1_ind[2])]))[0:500])

plt.xlabel('frequency (MHz)', fontsize=16)
plt.ylabel('power\n(arb. units)', fontsize=16)
plt.plot(out_xn)
plt.plot(a22[7])

output_signal = scipy.signal.filtfilt(b, a, out_xn)
#output_signal1 = scipy.signal.filtfilt(b1, a1, out_xn)
plt.plot(f0[0:500],np.abs(np.fft.fft(output_signal))[0:500])

plt.figure()
from matplotlib.legend_handler import HandlerLine2D
#mm, = plt.plot(output_signal[800:900],'r',label='recovered') #recoverd anode pulse
mm, = plt.plot(output_signal[100:240]/2**16*1000,'b',label='recovered',linewidth=2.0,alpha=0.7) #recoverd anode pulse
#mm, = plt.plot(output_signal[110:210]/2**16*1000,'b',label='recovered') #recoverd anode pulse
#mm1, = plt.plot(output_signal1[300:400],'r',label='recovered1') #recoverd anode pulse
nn, = plt.plot(a00[2,int(p1_ind[2]),100:240]/2**16*1000,'g',label='original',linewidth=2.0,alpha=0.7)#original anode pulse
#nn, = plt.plot(aa00[0,int(p1_ind1[0]),110-4:210-4]/2**16*1000,'g',label='original')#original anode pulse

plt.title('65 MHz',fontsize=20)
plt.xlabel('time (ns)',fontsize=16)
plt.ylabel('sample value (mV)',fontsize=16)
plt.legend(loc=1)
plt.tight_layout()

plt.plot(a00[2,123,120:240],'g')
plt.xlabel('sample number',fontsize=16)
plt.ylabel('sample value (ADC units)',fontsize=16)
plt.legend(loc=1)
plt.tight_layout()
plt.plot(a00[3,123,60:600],'g')
plt.xlabel('sample number',fontsize=16)
plt.ylabel('sample value (ADC units)',fontsize=16)
plt.legend(loc=1)
plt.tight_layout()

## 2 run this part second (method 1)

#do correlation in time , take average in time, then dive in frequency domain
M1 = 10000
from scipy import signal
corr1 = np.zeros((M1,1000))
corr2 = np.zeros((M1,1000))
corr3 = np.zeros((M1,1000)) #autocorrelation between output
h1 = np.zeros(1000,dtype=np.complex)
h2 = np.zeros(1000,dtype=np.complex)
h = np.zeros((1000),dtype=np.complex)
for i in range(M1):
    corr1[i] = signal.correlate(a00[1,i], a00[0,i], mode='same') / len(a00[0,0])
#    corr1[i] = signal.correlate(np.lib.pad(a0[i], (500,500), 'constant', constant_values=(0., 0.)), np.lib.pad(a2[i], (500,500), 'constant', constant_values=(0., 0.)), mode='same') #/ len(a0[0])
#    corr1[i] = np.lib.pad(cor1, (500,500), 'constant', constant_values=(0., 0.))
    corr2[i] = signal.correlate(a00[0,i], a00[0,i], mode='same') / len(a00[1,0])
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
corrr1 =  np.zeros(1000)
corrr2 =  np.zeros(1000)
#corrr3 =  np.zeros(2000)
for i in range(1000):
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
h = h1 / h2


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
plt.plot(f0[0:500],np.abs(h[0:500]))
plt.xlabel('frequency (MHz)')
plt.ylabel('amplitude (arb. units)')
#plt.ylim(0,2.5)
plt.tight_layout()

h_data = np.real(np.fft.ifft(h))
plt.figure()
plt.plot(h_data)
h_data1 = np.real(h_data)
np.savetxt('imp_res55MHz.txt', h_data1)
## 3 ends

plt.plot(f0[0:500],np.abs(h[0:500]))
plt.xlabel('frequency (MHz)')
plt.ylabel('amplitude (arb. units)')
plt.tight_layout()

np.savetxt('/home/radians/tunl/4to1mult/impres/50MHz/impres_freq.txt', np.abs(h))


plt.plot(f0[0:500],np.abs(np.fft.fft(corrr1))[0:500])
plt.plot(f0[0:500],np.abs(np.fft.fft(corrr2))[0:500])

#h = h1/h2
N1 = 2000
fs = 500
f0 = np.arange(N1)
f0 = (fs * 1. /N1) * f0

plt.plot(f0[0:1000],np.abs(h)[0:1000])
plt.xlabel('frequency (MHz)', fontsize=16)
plt.ylabel('power\n(arb. units)', fontsize=16)
h_data = np.fft.ifft(h)
plt.plot(h_data)
plt.xlabel('sample number',fontsize=16)
plt.ylabel('sample value',fontsize=16)

plt.plot(corr1[89])
plt.plot(corrr22)

plt.figure()
h_data = np.real(np.fft.ifft(h))
plt.plot(h_data)
plt.xlabel('sample number',fontsize=16)
plt.ylabel('sample value',fontsize=16)

for i in range(1000):
    plt.plot(a00[0,i])
    
plt.plot(a00[3,5000]/2**16*1000)
plt.xlabel('sample number (ns)',fontsize=16)
plt.ylabel('sample value (mV)',fontsize=16)
plt.tight_layout()

plt.plot(a00[2,5000]/2**16*1000)
plt.xlabel('sample number (ns)',fontsize=16)
plt.ylabel('sample value (mV)',fontsize=16)
plt.tight_layout()

## 2 ends
res_50 = []
res_55 = []
res_60 = []
res_65 = []
r_extra = []
for i in range(10000):
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
    if 10 + np.argmax(d1[10:500])== 48 or 10 + np.argmax(d1[10:500])== 49 or 10 + np.argmax(d1[10:500])== 50 or 10 + np.argmax(d1[10:500])== 51:
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


#                cnt += 1  
#                cnt65 += 1
#    if cnt>1:
#        if cnt50 > 0:
#            res_50.pop()
#        if cnt55 > 0:
#            res_55.pop()
#        if cnt60 > 0:
#            res_60.pop()
#        if cnt65 > 0:
#            res_65.pop()
#
#    if cnt ==0:
#        r_extra.append(a00[3,i])
    else:
        r_extra.append(a00[3,i])

len(res_50) + len(res_55) + len(res_60) + len(res_65)


res_50 = np.asarray(res_50)
plt.plot(np.abs(np.fft.fft(res_50[657,0])))                
    diff = np.diff(d1)
    diff2 = np.diff(diff)
    plt.plot(d1)
    plt.plot(diff)
    plt.plot(diff2)
    break

r_extra = np.asarray(r_extra)
plt.plot(np.abs(np.fft.fft(r_extra[6])))                
plt.plot(r_extra[6])