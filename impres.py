
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
rcParams['figure.figsize'] = 6, 5#5, 3

import sys




class drs_fdm_parser:
    def __init__(self,fileName,resonator, nboards, nchannels, nrecords):#parser
        self.fileName = fileName
        self.file2Read = open(fileName,'rb')
        self.nboards = nboards
        self.nrecords = nrecords
        self.samples = 1024
        self.resonator = resonator
        self.tcell = np.zeros((nboards,nrecords))
        np.fromfile(self.file2Read, dtype=np.dtype('S4'), count=1)
        np.fromfile(self.file2Read, dtype=np.dtype('S4'), count=1)
        self.time_bins = np.zeros((nboards,4,1024))
        self.time_samples = np.zeros((nchannels,self.samples,self.samples))#channel,trigger cell, timebins
        self.vch1 = np.zeros((self.nboards,4,self.nrecords,self.samples)) #record values
#        self.pad = pad
        
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
                self.tcell[i1,cvc] = bt['tc'] #get trigger cell
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
            if cvc >self.nrecords-1: #number of events to read (n-1)
                break   
            
        
        #get time arrays for all trigger cells
        self.time_samples = np.zeros((2,1024,1024))#channel,trigger cell, timebins
        for i in range(2):#channles
            for j in range(1024): #trigger cell
                temptime = np.zeros(1024)
                for k in range(1024): #timebins
                    q, r = divmod(j+k,1024)
                    if q:
                        temptime[k] = np.sum(self.time_bins[0,i,j:(j+k)]) + np.sum(self.time_bins[0,i,0:r])
                    else:
                        temptime[k] = np.sum(self.time_bins[0,i,j:(j+k)])
        
                self.time_samples[i,j] = np.copy(temptime)

        #time alignment
        for j in range(1024):#trigger cells
            t1 = 0
            t2 = 0
            time1 = self.time_samples[0,j]
            t1 = time1[(1024-j) % 1024]
    
            time2 = self.time_samples[1,j]
            t2 = time2[(1024-j) % 1024]
    
            dt = t1 - t2
            for j1 in range(1024):
                self.time_samples[1,j,j1] += dt            
        

    def imp_response(self):
        a00 = np.zeros((2,self.nrecords,1000))
        res = {'50': 0, '55': 0, '60': 1, '65': 2}
#a2 = np.zeros((10000,10000))
        chchch = [res[self.resonator],3] 
        for i1 in range(2):
            cct =0
            for i in range(self.nrecords):
                y1 = self.vch1[0,chchch[i1],i] - np.mean(self.vch1[0,chchch[i1],i,15:1015]) #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
                x1 = self.time_samples[i1,int(self.tcell[0,i])] #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
#        print(i)
        #linear interp
                xs = np.arange(15,1015,1.)#160/4000 = 0.04
                f2 = interp1d(x1,y1,kind='previous')# InterpolatedUnivariateSpline(x1, y1,k=1)#,kind='previous'
                a00[i1,i] = f2(xs)#f2(xs)#- np.mean(f2(xs)),y1[15:1015]

        ## 2 run this part second (method 1)
        
        #do correlation in time , take average in time, then dive in frequency domain
        M1 = self.nrecords
        from scipy import signal
        corr1 = np.zeros((M1,1999))
        corr2 = np.zeros((M1,1999))
        corr3 = np.zeros((M1,1999)) #autocorrelation between output
        h1 = np.zeros(1999,dtype=np.complex)
        h2 = np.zeros(1999,dtype=np.complex)
        h = np.zeros((1999),dtype=np.complex)
        for i in range(M1):
            corr1[i] = signal.correlate(a00[1,i], a00[0,i], mode='full') / 1999
        #    corr1[i] = signal.correlate(np.lib.pad(a0[i], (500,500), 'constant', constant_values=(0., 0.)), np.lib.pad(a2[i], (500,500), 'constant', constant_values=(0., 0.)), mode='same') #/ len(a0[0])
        #    corr1[i] = np.lib.pad(cor1, (500,500), 'constant', constant_values=(0., 0.))
            corr2[i] = signal.correlate(a00[0,i], a00[0,i], mode='full') / 1999
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
#        corrr11 = np.lib.pad(corrr1, (0,self.pad), 'constant', constant_values=(0., 0.))
        corrr2 = corrr2/M1
#        corrr22 = np.lib.pad(corrr2, (0,self.pad), 'constant', constant_values=(0., 0.))
        #for i in range(2000):
        #    h1[i] = np.fft.fft(Decimal(corrr1[i]))
        #    h2[i] = np.fft.fft(Decimal(corrr2[i]))
        h1 = np.fft.fft(corrr1)
        h2 = np.fft.fft(corrr2)
        h = h1 / h2
        h_t = np.real(np.fft.ifft(h))
        return h_t , corrr2, corrr1

#
#time_bins = np.zeros((2,4,1024)) #for two channels
##time = np.zeros((1,2,1024))
##wave = np.zeros((1,2,1024))
#nb = 0 #number of boards
#
##count the negative timebins
#cct = 0
#for i in range(1024):
#    if time_bins[0,0,i]<=0:
#        cct +=1
#print(cct)
#
#
#vch1 = np.zeros((2,4,10000,1024))
#tcell = np.zeros((2,10000))
##read board header
#with open(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\data2\imp_res55MHz_normal4.dat', 'rb') as f:
#    b = np.fromfile(f, dtype=np.dtype('S4'), count=1)
##    print('file header=',b.astype(str))
#    b = np.fromfile(f, dtype=np.dtype('S4'), count=1)
##    print('time header=',b.astype(str))
#    c1 =0
#    while(1):
#        b = np.fromfile(f, dtype=bh, count=1)
#        bb = b['c1'].astype(str)
##        print(b['c2'])
#        if bb!='B#':
#            #event header found
#            f.seek(-4,1)
#            break
##        print('board serial number',b['c2'])
#        for i in range(5):#keep looping for time bins for all channels
#            b = np.fromfile(f, dtype=ch1, count=1)
#            bb = b['c1'].astype(str)
##            print(bb)
#            if bb != 'C':
#                f.seek(-4,1)
#                break
#            i11 = int(b['c2'])
##            print('found time calibration of channel', i11)
#            b = np.fromfile(f, dtype=np.float32, count=1024)
#            time_bins[c1,i] = b
##            print(b)
#        c1 +=1
#    nb = c1
##    print('number of boards', c1)
#    
#    cvc = 0 #counter for number of events to read
#    while(1): #loop over events
#        be = np.fromfile(f, dtype=eh, count=1)
#        if not be:
#            break
##        print('found event', int(be['serial']), int(be['sec']), int(be['millisec']))
#        for i1 in range(nb):#number of boards
#            b1 = np.fromfile(f, dtype=bh, count=1)
#            bbb = b1['c1'].astype(str)
#            if bbb != 'B#':
#                print('invalid board header....exiting....')
#                sys.exit()
#                
#            bt = np.fromfile(f, dtype=tch, count=1)
#            bb = bt['c1'].astype(str)
#            if bb != 'T#':
#                print('invalid trigger cell....exiting....')
#                sys.exit()            
#            if nb > 1:
#                bserial = b1['c2'].astype(str)
##                print('board serial is ' ,bserial)
#                
##            plt.figure()
#            tcell[i1,cvc] = bt['tc'] #get trigger cell
#            for ch in range(4):#get channels data
##                print('we are hre')
#                b = np.fromfile(f, dtype=ch1, count=1)
#                bb = b['c1'].astype(str)
#                if bb != 'C':
#                    f.seek(-4,1)
#                    break
##                print(b['c2'])
#                ch_ind = int(b['c2'])-1
#                s = np.fromfile(f, dtype=np.int32, count=1)#get scaler
#                v = list(np.fromfile(f, dtype=np.ushort, count=1024))#get voltage
##                v[:] = [x - np.mean(v[15:1015]) for x in v]
#                vch1[i1,ch_ind,cvc] = v
##                plt.plot(v)
##                print(vch1[ch_ind,cvc])
#                
#                 
#
##                plt.plot(v)
#        
##                for i4 in range(1024):#convert data to volts
##                    wave[i1,ch_ind,i4] = (v[i4] / 65536. + be['range']/1000.0 - 0.5)
##                    #calculate time for each cell of present channel
##                    for j2 in range(i4):
##                        time[i1,ch_ind,i4] += time_bins[i1,ch_ind,((j2+bt['tc'])%1024)] 
##                vch1[cvc] = wave[i1,ch_ind] #saving data
##                tch1[cvc] = time[i1,ch_ind] #saving data
##                print('channel ch',ch)
##                print(tch1[cvc])
##            #allign cell 0 of all channels
##            t1 = time[i1,0,(1024-bt['tc']) % 1024]
##            for chn in range(1,2):
##                t2 = time[i1,chn,(1024-bt['tc']) % 1024]
##                dt = t1 - t2
##                for i5 in range(1024):
##                    time[i1,chn,i5] += dt
##            t1 = 0
##            t2 = 0
##            thres = 0.3
#        cvc +=1
#        if cvc >9999: #number of events to read (n-1)
#            break
#        
#
##plt.plot(vch1[2])        
#
##get time arrays for all trigger cells
#time_samples = np.zeros((2,1024,1024))#channel,trigger cell, timebins
#for i in range(2):#channles
#    for j in range(1024): #trigger cell
#        temptime = np.zeros(1024)
#        for k in range(1024): #timebins
#            q, r = divmod(j+k,1024)
#            if q:
#                temptime[k] = np.sum(time_bins[0,i,j:(j+k)]) + np.sum(time_bins[0,i,0:r])
#            else:
#                temptime[k] = np.sum(time_bins[0,i,j:(j+k)])
#        
#        time_samples[i,j] = np.copy(temptime)
#
##time alignment
#for j in range(1024):#trigger cells
#    t1 = 0
#    t2 = 0
#    time1 = time_samples[0,j]
#    t1 = time1[(1024-j) % 1024]
#    
#    time2 = time_samples[1,j]
#    t2 = time2[(1024-j) % 1024]
#    
#    dt = t1 - t2
#    for j1 in range(1024):
#        time_samples[1,j,j1] += dt
#        
##get time arrays for all trigger cells of second board
#time_samples1 = np.zeros((1,1024,1024))#channel,trigger cell, timebins
#for i in range(1):#channles
#    for j in range(1024): #trigger cell
#        temptime = np.zeros(1024)
#        for k in range(1024): #timebins
#            q, r = divmod(j+k,1024)
#            if q:
#                temptime[k] = np.sum(time_bins[1,i,j:(j+k)]) + np.sum(time_bins[1,i,0:r])
#            else:
#                temptime[k] = np.sum(time_bins[1,i,j:(j+k)])
#        
#        time_samples1[i,j] = np.copy(temptime)
#
##time alignment
#for j in range(1024):#trigger cells
#    t1 = 0
#    t2 = 0
#    time1 = time_samples1[0,j]
#    t1 = time1[(1024-j) % 1024]
#    
#
#
#print(np.average(time_bins[0,0]))        
##just plot
#for i in range(3999,4000):
##    plt.figure()
#    plt.plot(time_samples[0,int(tcell[0,i])],vch1[0,0,i])
#    plt.plot(time_samples[1,int(tcell[0,i])],vch1[0,3,i])
#    
##find an average pulse
##avg_pulse = np.zeros((2000,90)) #30+60 = 90
##visualize the pulses
#for i in range(5):
#    plt.plot(vch1[0,i,np.argmax(vch1[0,i])-30:np.argmax(vch1[0,i])+60]/ np.max(vch1[0,i]))
#
#avg_pulse = vch1[0,0,np.argmax(vch1[0,i])-30:np.argmax(vch1[0,i])+60] / np.max(vch1[0,0])
#
##avg_pulse[1] = (avg_pulse[0] + (vch1[0,1,np.argmax(vch1[0,i])-30:np.argmax(vch1[0,i])+60] 
##                                    / np.max(vch1[0,1]))) / 2.
#
#
#from scipy.interpolate import CubicSpline
#from scipy.interpolate import interp1d
#
#
#for i in range(20000):#avg[m] = (m/(m+1))*avg[m-1] + v[m]/(m+1)
#    y1 = vch1[0,i]  #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
#    x1 = time_samples[0,int(tcell[i])] #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
##    cs = CubicSpline(x1, y1)
##    xs = np.arange(time_samples[0,int(tcell[i]),np.argmax(vch1[0,i])-30],
##                   time_samples[0,int(tcell[i]),np.argmax(vch1[0,i])+50], 
##          (time_samples[0,int(tcell[i]),np.argmax(vch1[0,i])+50] - 
##            time_samples[0,int(tcell[i]),np.argmax(vch1[0,i])-30])/500)
##    ys = cs(xs)
#    
#    #linear interp
#    xs = np.arange(0,1010,1.)
#    f2 = interp1d(x1,y1,fill_value="extrapolate")
#    ys = f2(xs)
#    
#    
##    plt.plot(xs[np.argmax(ys)-150:np.argmax(ys)+150],ys[np.argmax(ys)-150:np.argmax(ys)+150])
#    if i == 0:
#        avg_pulse = ys[np.argmax(ys)-150:np.argmax(ys)+500] / np.max(ys)
#    else:
#        avg_pulse = ((avg_pulse)*i/(i+1)) + (ys[np.argmax(ys)-150:np.argmax(ys)+500] 
#                                    / (np.max(ys)*(i+1)))
##        try:
##            avg_pulse = ((avg_pulse)*i/(i+1)) + (ys[np.argmax(ys)-150:np.argmax(ys)+150] 
##                                    / (np.max(ys)*(i+1)))
###            cs = CubicSpline(x1, y1)
###            
##        except ValueError:
###            print('Oops!  Negative time')
##            plt.figure()
##            plt.plot(xs,ys)
##            print(i)
##            break
##        avg_pulse = ((avg_pulse)*i/(i+1)) + (ys[np.argmax(ys)-100:np.argmax(ys)+200] 
##                                    / (np.max(ys)*(i+1)))
#    
#    
#plt.plot(xs[np.argmax(avg_pulse)-150:np.argmax(avg_pulse)+500],avg_pulse)
#max_time = xs[np.argmax(avg_pulse)]
#p30max = 0.3*np.max(avg_pulse)
#N=1000
#fs = np.arange(0,1000,1)
#plt.plot(np.abs(np.fft.fft(avg_pulse))[0:500])
#
#xss= xs[np.argmax(avg_pulse)-150:np.argmax(avg_pulse)+200]
#for j in range(350-1):
#    if avg_pulse[j] <= p30max < avg_pulse[j+1]:
#        yy1 = avg_pulse[j]
#        xx1 = xss[j]
#        yy2 = avg_pulse[j+1]
#        xx2 = xss[j+1]
#        break
#            
#slope = (yy1 - yy2) / (xx1 - xx2)
#intrcept = yy1 - (slope*xx1) 
#timestamp30 = (p30max - intrcept)/slope 
#
#delta_t = max_time - timestamp30
#
#print(delta_t)
#plt.plot(xss,avg_pulse)
#
#plt.plot(xss,[p30max for number in range(len(xss))],label='0.3*max_amplitude')
#
#
#plt.plot(time_samples[0,int(tcell[i]),np.argmax(vch1[0,i])-30:np.argmax(vch1[0,i])+50],
#                      vch1[0,0,np.argmax(vch1[0,i])-30:np.argmax(vch1[0,i])+50] / np.max(vch1[0,0]))
#
#
##plt.plot(finalCFD)
##CFD algorithm
#t30 = np.zeros((2,10000,10000))
#
#for i1 in range(2):
#    cct =0
#    for i in range(10000):
#        y1 = vch1[i1,i]  #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
#        x1 = time_samples[i1,int(tcell[i])] #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
#        
#        #linear interp
#        xs = np.arange(0,1000,1.)#160/4000 = 0.04
#        f2 = interp1d(x1,y1,fill_value="extrapolate")
#        ys = f2(xs)
##        plt.plot(x1,y1)
#        atten_copy = np.zeros(len(xs))
#        delayed_copy = np.zeros(len(xs))#delay = 2.25ns/0.25 = 9
#        
#        atten_copy = np.copy(0.3*ys)
#        delayed_copy[2:len(xs)] = np.copy(ys[0:len(xs)-2])
#        finalCFD = delayed_copy - atten_copy
##        plt.plot(finalCFD)
#        u=0
#        for j in range(np.argmax(finalCFD)-5, np.argmax(finalCFD)):
#            if finalCFD[j] <= 0. < finalCFD[j+1]:
#                slope = (finalCFD[j] - finalCFD[j+1])/(xs[j]-xs[j+1])
#                intrcept = finalCFD[j] - (slope*xs[j])
#                t30[i1,i] = -1*intrcept/slope
#                cct += 1
#                u=1
#                break
#        if u==0 and i1==0:
#            t30[i1,i] = 10
#        elif u==0 and i1==1:
#            t30[i1,i] = -10
#    print(cct)
#
#        
##print(t30)
#t30_diff = np.subtract(t30[0],t30[1])
#t30_d = []
#for i in range(len(t30_diff)):
#    if -1.5 <= t30_diff[i] < 1.5:
#        t30_d.append(t30_diff[i])
#                
#                
#plt.hist(t30_d,100)
#plt.xlim(-1.5,1.5)
#plt.title('1 MSPS',fontsize = 16)
#plt.xlabel('time difference between \n coincident pulses',fontsize = 16)
#plt.ylabel('counts',fontsize = 16)
#print(np.mean(t30_d))
#print(np.sqrt(np.var(t30_d)))
#
#
#
##get spread in std
#spread_uncert = [[] for i in range(10)]
#spread_mean = [[] for i in range(10)]
#for i in range(10):
#    spread_uncert[i] = np.sqrt(np.var(t30_d[i*int(len(t30_d)/10):(i+1)*int(len(t30_d)/10)]))
#    spread_mean[i] = np.mean(t30_d[i*int(len(t30_d)/10):(i+1)*int(len(t30_d)/10)]) 
#print(spread_uncert)
#spread_un_mean = np.mean(spread_uncert)
#spread_un_un = np.sqrt(np.var(spread_uncert))
#mean_mean = np.mean(spread_mean)
#mean_un = np.sqrt(np.var(spread_mean))
#
#print(spread_un_mean)
#print(spread_un_un)
#mean_mean1 = []
#mean_mean1.append(mean_mean)
#mean_un1 = []
#mean_un1.append(mean_un)
#spread_mean = []
#spread_mean.append(spread_un_mean)
#spread_un =[]
#spread_un.append(spread_un_un)
#
#np.savetxt('spread_mean.txt',spread_mean)
#np.savetxt('spread_un.txt',spread_un)
#np.savetxt('mean_mean.txt',mean_mean1)
#np.savetxt('mean_un.txt',mean_un1)
#
##time_difference.append(t30_d)
##time_difference[1] = t30_d
#np.savetxt('time_diff.txt',t30_d)
#
#
#
#time_res = []
#time_res.append(np.sqrt(np.var(t30_d)))
#
#np.savetxt('1GHz_std.txt',time_res)
#
#time_mean = []
#time_mean.append(np.mean(t30_d))
#
#np.savetxt('1GHz_mean.txt',time_mean)
#
#
##new method
#
##new method timing
#   
##from scipy.interpolate import CubicSpline
#timestamp1 = np.zeros((2,4000))
#for i1 in range(2):
#    for i in range(4000):
#    
#        y1 = vch1[i1,i,np.argmax(vch1[i1,i])-10:np.argmax(vch1[i1,i])+11]
#        x1 = time_samples[i1,int(tcell[i]),np.argmax(vch1[i1,i])-10:np.argmax(vch1[i1,i])+11]
##        for j1 in range(len(x1)-1):
##            if x1[j1+1]<x1[j1]:
##                x1[j], x1[j+1] = x1[j+1], x1[j]
##                y1[j], y1[j+1] = y1[j+1], y1[j]
##                break
##                
##        try:
##            cs = CubicSpline(x1, y1)
##            
##        except ValueError:
##            print('Oops!  Negative time')
##            plt.figure()
##            plt.plot(time_samples[i1,int(tcell[i])],vch1[i1,i])
##            print(i)
##            break
#        cs = CubicSpline(x1, y1)
#        xs = np.arange(390,395,5./100)
#    #find time at 30% of the amplitude using linear interpolation
#        p30max = 0.3 * np.max(cs(xs))
#        
#        for j in range(len(y1)-1):
#            if y1[j] <= p30max < y1[j+1]:
#                yy1 = y1[j]
#                xx1 = x1[j]
#                yy2 = y1[j+1]
#                xx2 = x1[j+1]
#            
#        slope = (yy1 - yy2) / (xx1 - xx2)
#        intrcept = yy1 - (slope*xx1) 
#        timestamp1[i1,i] = (p30max - intrcept)/slope          
#        
##        for j in range(len(xs)-1):
##            if cs(xs)[j] <= p30max < cs(xs)[j+1]:
##                yy1 = cs(xs)[j]
##                xx1 = xs[j]
##                yy2 = cs(xs)[j+1]
##                xx2 = xs[j+1]
##            
##        slope = (yy1 - yy2) / (xx1 - xx2)
##        intrcept = yy1 - (slope*xx1) 
##        timestamp1[i1,i] = (p30max - intrcept)/slope  
#
#ts = []
#for i in range(4000):
#    if -1. < timestamp1[0,i]-timestamp1[1,i] <1.:
#        ts.append(timestamp1[0,i]-timestamp1[1,i])
#        
##plt.hist(timestamp1[0]-timestamp1[1],200)
#plt.figure()
#                
#plt.hist(ts,100)
#plt.xlim(-1.,1.)
#plt.title('1 GSPS',fontsize = 16)
#plt.xlabel('time difference between \n coincident pulses',fontsize = 16)
#plt.ylabel('counts',fontsize = 16)
#print(np.mean(ts))
#print(np.sqrt(np.var(ts)))
#
#time_res = []
#time_res.append(np.sqrt(np.var(ts)))
#
#np.savetxt('1GHz_std.txt',time_res)
#
#
#
#    
#
##from here
#from scipy.interpolate import InterpolatedUnivariateSpline
#from scipy.interpolate import CubicSpline
#from scipy.interpolate import interp1d
#a00 = np.zeros((2,10000,1000))
##a2 = np.zeros((10000,10000))
#chchch = [1,3] 
#for i1 in range(2):
#    cct =0
#    for i in range(10000):
#        y1 = vch1[0,chchch[i1],i] - np.mean(vch1[0,chchch[i1],i,15:1015]) #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
#        x1 = time_samples[i1,int(tcell[0,i])] #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
##        print(i)
#        #linear interp
#        xs = np.arange(15,1015,1.)#160/4000 = 0.04
#        f2 = interp1d(x1,y1,kind='previous')# InterpolatedUnivariateSpline(x1, y1,k=1)#,kind='previous'
#        a00[i1,i] = f2(xs)#f2(xs)#- np.mean(f2(xs)),y1[15:1015]
#
### 2 run this part second (method 1)
#
##do correlation in time , take average in time, then dive in frequency domain
#M1 = 10000
#from scipy import signal
#corr1 = np.zeros((M1,1999))
#corr2 = np.zeros((M1,1999))
#corr3 = np.zeros((M1,1999)) #autocorrelation between output
#h1 = np.zeros(1999,dtype=np.complex)
#h2 = np.zeros(1999,dtype=np.complex)
#h = np.zeros((1999),dtype=np.complex)
#for i in range(M1):
#    corr1[i] = signal.correlate(a00[1,i], a00[0,i], mode='full') / len(a00[0,0])
##    corr1[i] = signal.correlate(np.lib.pad(a0[i], (500,500), 'constant', constant_values=(0., 0.)), np.lib.pad(a2[i], (500,500), 'constant', constant_values=(0., 0.)), mode='same') #/ len(a0[0])
##    corr1[i] = np.lib.pad(cor1, (500,500), 'constant', constant_values=(0., 0.))
#    corr2[i] = signal.correlate(a00[0,i], a00[0,i], mode='full') / len(a00[1,0])
##    corr2[i] = signal.correlate(np.lib.pad(a2[i], (500,500), 'constant', constant_values=(0., 0.)), np.lib.pad(a2[i], (500,500), 'constant', constant_values=(0., 0.)), mode='same')
##    corr2[i] = np.lib.pad(cor2, (500,500), 'constant', constant_values=(0., 0.))
##    corr3[i] = signal.correlate(a0[i], a0[i], mode='same') #/ len(a0[0])
##    plt.plot(corr2)
##    plt.xlabel('sample number',fontsize=16)
##    plt.ylabel('autocorrelation',fontsize=16)
###    h1[i] = np.fft.fft(corr1[i])
##    plt.plot(f0[0:50],np.abs(h1)[0:50])
###    h2[i] = np.fft.fft(corr2[i])
##    plt.plot(f0[0:1000],np.abs(h2)[0:1000])
##    plt.xlabel('frequency (MHz)', fontsize=16)
##    plt.ylabel('power\n(arb. units)', fontsize=16)
#
###    h[i] = h1[i]/h2[i]
##    plt.plot(f0[0:50],np.abs(h)[0:50])
##    plt.xlabel('frequency (MHz)', fontsize=16)
##    plt.ylabel('power\n(arb. units)', fontsize=16)
##corr11 = np.zeros((2000))
##corr22 = np.zeros(2000)
#corrr1 =  np.zeros(1999)
#corrr2 =  np.zeros(1999)
##corrr3 =  np.zeros(2000)
#for i in range(1999):
#    for j in range(M1):
#        corrr1[i] = corrr1[i] + corr1[j,i]
#        corrr2[i] = corrr2[i] + corr2[j,i]
##        (corrr1[i]) = Decimal(corrr1[i]) + Decimal(corr1[j,i]/M1)
##        (corrr2[i]) = Decimal(corrr2[i]) + Decimal(corr2[j,i]/M1)
#
##        corrr3[i] = corrr3[i] + corr3[j,i]
##        corr11[i] = corr11[i] + corr1[j,i]
##        corr22[i] = corr22[i] + corr2[j,i]
##corr11 = corr11/10000
##corr22 = corr22/10000
#corrr1 = corrr1/M1
##corrr11 = np.lib.pad(corrr1, (0,2000), 'constant', constant_values=(0., 0.))
#corrr2 = corrr2/M1
##corrr22 = np.lib.pad(corrr2, (0,2000), 'constant', constant_values=(0., 0.))
##for i in range(2000):
##    h1[i] = np.fft.fft(Decimal(corrr1[i]))
##    h2[i] = np.fft.fft(Decimal(corrr2[i]))
#h1 = np.fft.fft(corrr1)
#h2 = np.fft.fft(corrr2)
#h65 = h1 / h2
#
#plt.plot(np.fft.ifft(h65))
#h_data = np.fft.ifft(h65)[0:1000]
#plt.plot(corrr1)
##to get autocorellation for output y
#corrr3 = corrr3/M1
#corrr33 = np.lib.pad(corrr3, (0,500), 'constant', constant_values=(0., 0.))
## if no zero padding
##h1 = np.fft.fft(corrr1)
##h2 = np.fft.fft(corrr2)
##h = h1 / h2
N1 = 1999
fs = 1000
f0 = np.arange(N1)
f0 = (fs * 1. /N1) * f0
#plt.figure()
#plt.plot(f0[0:500],np.abs(np.fft.fft(h_data[0:1000]))[0:500])
##plt.plot(f0[55:75],np.abs(h65[55:75]))
#plt.xlabel('frequency (MHz)')
#plt.ylabel('amplitude (arb. units)')
##plt.ylim(0,2.5)
#plt.tight_layout()
#
#fig, ax1 = plt.subplots()
#rcParams['xtick.labelsize'] = 16
#rcParams['ytick.labelsize'] = 16
#ax1.plot(f0[0:500],np.abs(h55[0:500]))
#plt.xlabel('frequency (MHz)')
#plt.ylabel('amplitude (arb. units)')
## These are in unitless percentages of the figure size. (0,0 is bottom left)
#
#rcParams['xtick.labelsize'] = 9
#rcParams['ytick.labelsize'] = 9
#left, bottom, width, height = [0.45, 0.4, 0.4, 0.4]
#ax2 = fig.add_axes([left, bottom, width, height])
#ax2.plot(f0[45:65],np.abs(h55[45:65]))
#plt.show()
#rcParams['xtick.labelsize'] = 16
#rcParams['ytick.labelsize'] = 16
#
#
#h_data = np.real(np.fft.ifft(h))
#plt.figure()
#plt.plot(h_data)
#h_data1 = np.real(h_data)
#np.savetxt('imp_res55MHz.txt', h_data1)
### 3 ends
#
#plt.plot(f0[0:500],np.abs(h[0:500]))
#plt.xlabel('frequency (MHz)')
#plt.ylabel('amplitude (arb. units)')
#plt.tight_layout()
#
#np.savetxt('/home/radians/tunl/4to1mult/impres/50MHz/impres_freq.txt', np.abs(h))
#
#
#plt.plot(f0[0:500],np.abs(np.fft.fft(corrr1))[0:500])
#plt.plot(f0[0:500],np.abs(np.fft.fft(corrr2))[0:500])
#
##h = h1/h2
#N1 = 2000
#fs = 500
#f0 = np.arange(N1)
#f0 = (fs * 1. /N1) * f0
#
#plt.plot(f0[0:1000],np.abs(h50)[0:1000])
#plt.xlabel('frequency (MHz)', fontsize=16)
#plt.ylabel('power\n(arb. units)', fontsize=16)
#h_data = np.fft.ifft(h)
#plt.plot(h_data)
#plt.xlabel('sample number',fontsize=16)
#plt.ylabel('sample value',fontsize=16)
#
#plt.plot(corr1[89])
#plt.plot(corrr22)
#
#plt.figure()
#h_data = np.real(np.fft.ifft(h))
#plt.plot(h_data)
#plt.xlabel('sample number',fontsize=16)
#plt.ylabel('sample value',fontsize=16)
### 2 ends


if __name__ == '__main__':

#    qq=drs_fdm_parser(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\impres50_2.dat','50',2,2,15000)
#    h50_t = qq.imp_response()
#    h50 = np.abs(np.fft.fft(h50_t))
#    plt.figure()
#    plt.plot(f0[0:500],np.abs(h50)[0:500])
#    plt.xlabel('frequency (MHz)')
#    plt.ylabel('amplitude (arb. units)')
#    plt.tight_layout()
#
#    qq=drs_fdm_parser(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\impres55_2.dat','55',2,2,15000)
#    h55_t = qq.imp_response()
#    h55 = np.abs(np.fft.fft(h55_t))
#    plt.figure()
#    plt.plot(f0[0:500],np.abs(h55)[0:500])
#    plt.xlabel('frequency (MHz)')
#    plt.ylabel('amplitude (arb. units)')
#    plt.tight_layout()
#
#    qq=drs_fdm_parser(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\impres60_2.dat','60',2,2,15000)
#    h60_t = qq.imp_response()
#    h60 = np.abs(np.fft.fft(h60_t))
#    plt.figure()
#    plt.plot(f0[0:500],np.abs(h60)[0:500])
#    plt.xlabel('frequency (MHz)')
#    plt.ylabel('amplitude (arb. units)')
#    plt.tight_layout()
##    
#    qq=drs_fdm_parser(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\impres65_2.dat','65',2,2,15000)
#    h65_t = qq.imp_response()
#    h65 = np.abs(np.fft.fft(h65_t))
#    plt.figure()
#    plt.plot(f0[0:500],np.abs(h65)[0:500])
#    plt.xlabel('frequency (MHz)')
#    plt.ylabel('amplitude (arb. units)')
#    plt.tight_layout()
    
    
#second
    qq=drs_fdm_parser(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\impres550_2.dat','50',2,2,23000)
    h50_t, cor22, cor11 = qq.imp_response()
    h50 = np.abs(np.fft.fft(h50_t[0:1000]))
    plt.figure()
    plt.plot(f0[0:500],h50[0:500])
    plt.xlabel('frequency (MHz)')
    plt.ylabel('amplitude (arb. units)')
    plt.tight_layout()
#    plt.plot(f0,np.abs(np.fft.fft(cor2)))
#    plt.plot(f0,np.abs(np.fft.fft(cor1)))

    qq=drs_fdm_parser(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\impres555_2.dat','55',2,2,23000)
    h55_t, cor22, cor11 = qq.imp_response()
    h55 = np.abs(np.fft.fft(h55_t[0:1000]))
    plt.figure()
    plt.plot(f0[0:500],np.abs(h55)[0:500])
    plt.xlabel('frequency (MHz)')
    plt.ylabel('amplitude (arb. units)')
    plt.tight_layout()

    qq=drs_fdm_parser(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\impres560_2.dat','60',2,2,23000)
    h60_t, cor22, cor11 = qq.imp_response()
    h60 = np.abs(np.fft.fft(h60_t[0:1000]))
    plt.figure()
    plt.plot(f0[0:500],np.abs(h60)[0:500])
    plt.xlabel('frequency (MHz)')
    plt.ylabel('amplitude (arb. units)')
    plt.tight_layout()
#    
    qq=drs_fdm_parser(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\second\impres565_2.dat','65',2,2,23000)
    h65_t, cor22, cor11 = qq.imp_response()
    h65 = np.abs(np.fft.fft(h65_t[0:1000]))
    plt.figure()
    plt.plot(f0[0:500],np.abs(h65[0:1000])[0:500])
    plt.xlabel('frequency (MHz)')
    plt.ylabel('amplitude (arb. units)')
    plt.tight_layout()
    
    
#    qq=drs_fdm_parser(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\impres50_2.dat','50',2,2,15000,0)
#    h50_t = qq.imp_response()
#    qq1=drs_fdm_parser(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\impres50_2.dat','50',2,2,15000,100)
#    h50_t1 = qq1.imp_response()
#    plt.plot(h50_t)
#    plt.plot(h50_t1)
#    h501t = np.abs(np.fft.fft(h50_t1[0:1050]))
#    plt.plot(h501t)
#    h501 = np.abs(np.fft.fft(h50_t[0:1000]))
#    plt.plot(h501)
    
    
#third
    qq=drs_fdm_parser(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\third\imp50_ej204_30000_2.dat','50',2,2,39950)
    h50_t, cor2, cor1 = qq.imp_response()
    h50 = np.abs(np.fft.fft(h50_t[0:1000]))
    plt.figure()
    plt.plot(f0[0:500],h50[0:500])
    plt.xlabel('frequency (MHz)')
    plt.ylabel('amplitude (arb. units)')
    plt.tight_layout()
    savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\third\plots\impres50')

    
    plt.figure()
    plt.plot(f0,np.abs(np.fft.fft(cor2)))
    plt.plot(f0,np.abs(np.fft.fft(cor1)))  
    
    
    
    qq=drs_fdm_parser(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\third\imp55_ej204_40000_2.dat','55',2,2,39950)
    h55_t, cor2, cor1 = qq.imp_response()
    h55 = np.abs(np.fft.fft(h55_t[0:1000]))
    plt.figure()
    plt.plot(f0[0:500],h55[0:500])
    plt.xlabel('frequency (MHz)')
    plt.ylabel('amplitude (arb. units)')
    plt.tight_layout()
    savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\third\plots\impres55')

    plt.figure()
    plt.plot(f0,np.abs(np.fft.fft(cor2)))
    plt.plot(f0,np.abs(np.fft.fft(cor1))) 

    qq=drs_fdm_parser(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\third\imp60_ej204_40000_2.dat','50',2,2,39950)
    h60_t, cor2, cor1 = qq.imp_response()
    h60 = np.abs(np.fft.fft(h60_t[0:1000]))
    plt.figure()
    plt.plot(f0[0:500],h60[0:500])
    plt.xlabel('frequency (MHz)')
    plt.ylabel('amplitude (arb. units)')
    plt.tight_layout()
    savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\third\plots\impres60')

    plt.figure()
    plt.plot(f0,np.abs(np.fft.fft(cor2)))
    plt.plot(f0,np.abs(np.fft.fft(cor1))) 

    qq=drs_fdm_parser(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\third\imp65_ej204_40000_2.dat','50',2,2,39950)
    h65_t, cor2, cor1 = qq.imp_response()
    h65 = np.abs(np.fft.fft(h65_t[0:1000]))
    plt.figure()
    plt.plot(f0[0:500],h65[0:500])
    plt.xlabel('frequency (MHz)')
    plt.ylabel('amplitude (arb. units)')
    plt.tight_layout()
    savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\final\third\plots\impres65')

    plt.figure()
    plt.plot(f0,np.abs(np.fft.fft(cor2)))
    plt.plot(f0,np.abs(np.fft.fft(cor1))) 