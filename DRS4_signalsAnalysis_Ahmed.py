'''
    This code filters and analyzes the DRS4 signals through the following steps:
        1. Takes a data array as a numpy array on the format [eventNo, channelNo, variable, value]
            where:
                variable is 0 for rime and 1 for amplitude.
                value is the 1024 values
        
        2. Applies baseline correction, then multiple-peak finding to eliminate signals with sharp peaks (discharges).
        3. Applies two methods to detect the peak amplitude and estimated the timestamp of half-peak on the leading edge.
        
        
    To run it, in an IPython console run:
        "run DRS4_signalsAnalysis.py prepare <dataFileName>"
        then
        "run DRS4_signalsAnalysis.py analyze <dataFileName> <method>"
        
        or
        "run DRS4_signalsAnalysis.py do <dataFileName> <method>"
'''


import sys

import numpy as np

import copy

from scipy import interpolate


def saveNumpyArray(arrayName, array):
    np.save(str.format('{}', arrayName), array)
    return

def loadNumpyArray(arrayFileName):
    return np.load(str.format('{}.npy', arrayFileName))

'''===================================
>>>>>>>>> Data preparation <<<<<<<<<<<
==================================='''

def baselineCorrection(eventData, baselineRange=[50, 125]):
    '''
        do baseline correction using the average counts over the given range
    '''
    baselineRange_low, baselineRange_high = baselineRange[0], baselineRange[1]
    
    NoOfChannels = len(eventData[:, 0, 0])
    
    for ch in range(NoOfChannels):
        #data[e, ch, 0, :] = data[e, ch, 0, :] - data[e, ch, 0, 0]             # use this when ch 0 on the DRS4 wasn't used
        eventData[ch, 1, :] = eventData[ch, 1, :] - np.mean( eventData[ch, 1, baselineRange_low:baselineRange_high] )
    
    return eventData


def CenteredMovingAverage(window, counts):
    '''
        A centered-moving average function for signal smoothening
        - taken from Melinda's drs4ana.cc code
    '''
    filteredCounts = np.zeros(len(counts))
    if(window%2 == 0):
        window = window + 1
        print 'this filter is designed to function with odd window sizes. Setting window to ', window 
    
    halfWindow = int(window / 2)
    summation = np.sum(counts[:window])
    filteredCounts[:halfWindow+1] = summation / window
    
    for i in np.arange(halfWindow+1, len(counts)-halfWindow):
        summation = summation - counts[i - halfWindow - 1]
        summation = summation + counts[i + halfWindow]
        filteredCounts[i] = summation / window
    
    filteredCounts[len(counts)-halfWindow:] = summation / window
    
    return filteredCounts


def identifyMultiplePeaks(eventData):
    '''
        identify multiple peaks (indicating accidental coincidences) on signals of an event
        - Using smoothening then sample differencing
    '''
    
    NoOfChannels = len(eventData[:, 0, 0])
    flag = False
    window = 20       # roughly 4ns
    
    # doing baseline correction
    correctedEventData = baselineCorrection(eventData)
    
    for ch in range(NoOfChannels):
        peaks = 0
        originalCounts = correctedEventData[ch, 1, :]
        
        # checking if there is a peak at the start
        if np.all(originalCounts[ :window] > 5.0):
            flag = True
            break
        
        # checking if there is a peak at the end
        if np.all(originalCounts[1024-window: ] > 5.0):
            flag = True
            break
        
        # To be extra sure that the peak finding algorithm later works,
        # check that the signal with max amplitude is none other than the "primary" coincidence one
        if np.argmax(originalCounts) < window or np.argmax(originalCounts) > 1024-window:
            flag = True
            break
        
        # finding the number of peaks in the middle
        Counts = CenteredMovingAverage(21, originalCounts)  # aggressive signal smoothening to eliminate
                                                            # sudden dips on the signal 
        Counts = Counts + 50
        
        diff = Counts[1:] - Counts[:-1]
        
        for i in np.arange(window, 1024-window):
            if diff[i] <= 0 and diff[i-1] > 0:
                if originalCounts[i] > 5.0:
                    peaks += 1
            if peaks > 1:
                flag = True
                break
    return flag, correctedEventData


def deleteMultiplePeakSignals(data):
    '''
        identifies and deletes events having signals with multiple peaks
    '''
    
    NoOfEvents = len(data)
    delEvent_list = []
    
    for e in range(NoOfEvents):
        if e%1000 == 0: print 'filtering event: ', e
        hasMultiplePeaks, data[e, :, :, :] = identifyMultiplePeaks(data[e, :, :, :])
        if hasMultiplePeaks == True: delEvent_list.append(e)
    
    delEvent_list = list(set(delEvent_list))    
    data = np.delete(data, delEvent_list, axis = 0)
    
    return data



'''================================
>>>>>>>>> Data analysis <<<<<<<<<<<
================================'''

def tHalf(peak, t, counts):
    '''
        A function to find the time of a pulse half amplitude
    '''
    
    half_peak = peak/2.0
    
    approximatePeak_index = np.argmax(counts)   # to limit the search range to before the peak for faster performance
    half_peak_index = approximatePeak_index-50 + (np.abs(counts[approximatePeak_index-50:approximatePeak_index] - half_peak)).argmin()
    
    t_half = np.interp(half_peak, counts[half_peak_index-1 : half_peak_index + 2], t[half_peak_index-1 : half_peak_index + 2])
    
    return t_half



def amplitudeAndTime(data, method = '2', thresholdRange = [50, 125]):
    '''
        A function to find the amplitude timestamp of signals
    '''
    
    analyzedData = copy.deepcopy(data)
    
    NoOfEvents = len(analyzedData)
    NoOfChannels = len(analyzedData[0, :, 0, 0])
    
    pulsePeaks = np.zeros([NoOfEvents, NoOfChannels])
    pulseTimestamps = np.zeros([NoOfEvents, NoOfChannels])
    
    belowThreshold_list = []
    thresholdRange_low, thresholdRange_high = thresholdRange[0], thresholdRange[1]
    
    if method == '1':
        for e in range(NoOfEvents):
            if e%1000 == 0: print str.format('Smoothening signals of event {:d}', e)
    
            for ch in range(NoOfChannels):
                
                # smothen the signal
                analyzedData[e, ch, 1, :] = CenteredMovingAverage(5, analyzedData[e, ch, 1, :])
                
                peak = np.max(analyzedData[e, ch, 1, :])
                threshold = np.mean(analyzedData[e, ch, 1, thresholdRange_low:thresholdRange_high]) + 6 * np.std(analyzedData[e, ch, 1, thresholdRange_low:thresholdRange_high])    # 6 standard deviaitons above the mean baseline
                
                if peak < threshold:
                    belowThreshold_list.append(e)
                    break
                else:
                    pulsePeaks[e, ch] = peak
                    pulseTimestamps[e, ch] = tHalf(peak, analyzedData[e, ch, 0, :], analyzedData[e, ch, 1, :])
    
    if method == '2':
        for e in range(NoOfEvents):
            if e%1000 == 0: print str.format('Analyzing signals of event {:d}', e)
    
            for ch in range(NoOfChannels):
                
                # fitting using quadratic spline around the maximum bin only
                maxVIndex = np.argmax(analyzedData[e, ch, 1, :])
                indices = np.array( [indx for indx in range(maxVIndex - 6, maxVIndex + 12)] )
                abscissa = analyzedData[e, ch, 0, indices]
                ordinate = analyzedData[e, ch, 1, indices]
                spline = interpolate.UnivariateSpline(abscissa, ordinate, k = 3)
                peak = np.max( spline(analyzedData[e, ch, 0, indices]) )
                
                threshold = np.mean(analyzedData[e, ch, 1, thresholdRange_low:thresholdRange_high]) + 6 * np.std(analyzedData[e, ch, 1, thresholdRange_low:thresholdRange_high])    # 6 standard deviaitons above the mean baseline
                
                if peak < threshold:
                    belowThreshold_list.append(e)
                    break
                else:
                    pulsePeaks[e, ch] = peak
                    pulseTimestamps[e, ch] = tHalf(peak, analyzedData[e, ch, 0, :], analyzedData[e, ch, 1, :])
        
    
    pulsePeaks = np.delete(pulsePeaks, belowThreshold_list, axis = 0)
    pulseTimestamps = np.delete(pulseTimestamps, belowThreshold_list, axis = 0)
    
    return pulsePeaks, pulseTimestamps


#%% 


if __name__ == '__main__':
    '''
        main function to execute according to the "command" variable
        command: 'prepare' or 'analyze' or 'do'
    '''
    
    command  = sys.argv[1]
    
    if command == 'prepare':
        inFileName   = sys.argv[2]
        
        # load  data array
        rawData = loadNumpyArray(inFileName+'_raw')
        
        # search and delete events with multiple peaks and do baseline correction
        filteredData = deleteMultiplePeakSignals(rawData)
        rawData = None                                                             # deleting it to free memory
        del rawData
        # save filtered data numpy array
        saveNumpyArray(inFileName+'_filtered', filteredData)
        filteredData = None                                                        # deleting it to free memory
        del filteredData
    
    
    if command == 'analyze':
        inFileName = sys.argv[2]
        method   = sys.argv[3]
        
        # load filtered data numpy array
        data = loadNumpyArray(inFileName+'_filtered')
        # do the analysis
        pulsePeaks, pulseTimestamps = amplitudeAndTime(data, method)
        # save the results arrays
        saveNumpyArray(inFileName+'_pulsePeaks-Method'+method, pulsePeaks)
        saveNumpyArray(inFileName+'_pulseTimestamps-Method'+method, pulseTimestamps)
        
        
    if command == 'do ':
        inFileName = sys.argv[2]
        method   = sys.argv[3]
        
        # load  data array
        rawData = loadNumpyArray(inFileName+'_raw')
        
        # search and delete events with multiple peaks and do baseline correction
        filteredData = deleteMultiplePeakSignals(rawData)
        rawData = None                                                             # deleting it to free memory
        del rawData
        # save filtered data numpy array
        saveNumpyArray(inFileName+'_filtered', filteredData)
        
        # do the analysis
        pulsePeaks, pulseTimestamps = amplitudeAndTime(filteredData, method)
        # save the results arrays
        saveNumpyArray(inFileName+'_pulsePeaks-Method'+method, pulsePeaks)
        saveNumpyArray(inFileName+'_pulseTimestamps-Method'+method, pulseTimestamps)
        