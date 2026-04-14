import glob
import numpy as np
from scipy.signal import resample_poly
import wfdb
import _pickle as pickle
import os
import random
from datetime import datetime
from scipy.interpolate import CubicHermiteSpline
import scipy.signal

np.random.seed(42)
random.seed(42)

def getWindowFilter(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    # return medfilt(signal, kernel_size=window_size)

def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=250, order=5,
                    use_window=False, window_size=9):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    filtered = scipy.signal.filtfilt(b, a, signal)
    if use_window:
        filtered = getWindowFilter(filtered, window_size)
    return filtered

def remove_baseline(segment):
    h = np.arange(len(segment))
    # use linear function to fit the baseline
    coefficients = np.polyfit(h, segment, 1)
    g_h = np.polyval(coefficients, h)
    # remove the baseline
    x_hat = segment - g_h
    return x_hat

def find_basepoint(segment, start_idx, direction, basepoint):
    for i in range(start_idx, start_idx + direction * int(2 * len(segment) / 5), direction):
        if i < 0 or i >= len(segment):
            break
        if segment[i] == basepoint:
            return i
        if (segment[i] - basepoint) * (segment[i + direction] - basepoint) < 0:
            return i

    return None
def hermite_interpolation(last_segment, current_segment):
    # to find the basepoints
    basepoint = current_segment[0]
    p1_idx = find_basepoint(last_segment, len(last_segment) - 1, -1, basepoint)
    basepoint = last_segment[-1]
    p2_idx = find_basepoint(current_segment, 0, 1, basepoint)

    if p1_idx is not None and p2_idx is not None:
        if (len(last_segment) - p1_idx) < p2_idx:
            p2_idx = 0
        else:
            p1_idx = len(last_segment) - 1
    elif p1_idx is not None:
        p2_idx = 0
    elif p2_idx is not None:
        p1_idx = len(last_segment) - 1
    else:
        p1_idx = len(last_segment) - len(last_segment) // 5
        p2_idx = len(last_segment) // 5

    # obtain the bound of the junction
    p1 = last_segment[p1_idx]
    p2 = current_segment[p2_idx]

    # calculate 1-order derivative
    derivative1 = (last_segment[p1_idx] - last_segment[p1_idx - 11]) / 10 if p1_idx > 0 else 0
    derivative2 = (current_segment[p2_idx + 11] - current_segment[p2_idx]) / 10 if p2_idx < len(current_segment) - 1 else 0

    # HERMITE intepolation
    # calculate the length of the junction
    x = np.array([0, len(last_segment) - p1_idx + p2_idx])
    y = np.array([p1, p2])
    dydx = np.array([0.05*derivative1, 0.05*derivative2])
    interp = CubicHermiteSpline(x, y, dydx)
    new_points = interp(np.arange(len(last_segment) - p1_idx + p2_idx + 1))

    # replace the junction
    last_segment[p1_idx:] = new_points[:len(last_segment) - p1_idx]
    current_segment[:p2_idx + 1] = new_points[len(last_segment) - p1_idx:]

    return last_segment, current_segment

def process_signal(signal, n_indices, samples):
    segments = []
    prev_end = 0

    for i in range(len(n_indices) - 1):
        start = samples[n_indices[i]]
        end = samples[n_indices[i + 1]]
        midpoint = (start + end) // 2
        segment = signal[prev_end:midpoint]
        segment = remove_baseline(segment)

        if i > 0:
            last_segment = segments[-1]
            last_segment, segment = hermite_interpolation(last_segment, segment)

        segments.append(segment)

        prev_end = midpoint

    # the last segment
    segment = signal[prev_end:]
    segment = remove_baseline(segment)

    if len(segments) > 0:
        last_segment = segments[-1]
        last_segment, segment = hermite_interpolation(last_segment, segment)

    segments.append(segment)

    return np.concatenate(segments)

def prepare(QTpath='./data/qt-database-1.0.0/'):
    newFs = 360  
    segment_length = 10  
    namesPath = glob.glob(QTpath + "/*.dat")

    QTDatabaseSignals = dict()

    for i in namesPath:
        aux = i.split('.dat')  
        register_name = os.path.basename(aux[0])  # recording id
        signal, fields = wfdb.rdsamp(aux[0])  
        fs = fields['fs'] 
        # for key in fields:
        #    print(key, fields[key])

        signalsRe = list()

        auxSig1 = signal[:, 0] 
        auxSig2 = signal[:, 1] 

        ######preprocessing######
        # pu0, pu1 are the annotations of channel 1 & 2
        ann_pu0 = wfdb.rdann(aux[0], 'pu0')
        symbols_pu0 = ann_pu0.symbol
        samples_pu0 = ann_pu0.sample
        symbols_pu0 = np.array(symbols_pu0)

        ann_pu1 = wfdb.rdann(aux[0], 'pu1')
        symbols_pu1 = ann_pu1.symbol
        samples_pu1 = ann_pu1.sample
        symbols_pu1 = np.array(symbols_pu1)

        # 'N' is the label of QRS complex
        n_indices_pu0 = np.where(symbols_pu0 == 'N')[0]
        n_indices_pu1 = np.where(symbols_pu1 == 'N')[0]

        auxSig1 = bandpass_filter(auxSig1, use_window=True)
        processed_signal_pu0 = process_signal(auxSig1, n_indices_pu0, samples_pu0)

        auxSig2 = bandpass_filter(auxSig2, use_window=True)
        processed_signal_pu1 = process_signal(auxSig2, n_indices_pu1, samples_pu1)

        processed_signal_pu0 = resample_poly(processed_signal_pu0, newFs, fs) 
        # avoiding edge effect 
        processed_signal_pu0 = processed_signal_pu0[1:-1] 

        processed_signal_pu1 = resample_poly(processed_signal_pu1, newFs, fs)
        processed_signal_pu1 = processed_signal_pu1[1:-1] 

        # length 3600 = 360 * 10
        segment_samples = int(segment_length * newFs)

        # Uniformly crop
        for start in range(0, len(processed_signal_pu0), segment_samples):
            end = start + segment_samples
            if end > len(processed_signal_pu0):  
                break
            segment = processed_signal_pu0[start:end]  
            signalsRe.append(segment)

        # Randomly crop
        for _ in range(90):
            if len(processed_signal_pu0) >= segment_samples:
                start = np.random.randint(0, len(processed_signal_pu0) - segment_samples + 1)
                end = start + segment_samples
                segment = processed_signal_pu0[start:end]
                # segment = normalize(segment)
                signalsRe.append(segment)

        for start in range(0, len(processed_signal_pu1), segment_samples):
            end = start + segment_samples
            if end > len(processed_signal_pu1):  
                break
            segment = processed_signal_pu1[start:end]  
            signalsRe.append(segment)  

        for _ in range(90):
            if len(processed_signal_pu1) >= segment_samples:
                start = np.random.randint(0, len(processed_signal_pu1) - segment_samples + 1)
                end = start + segment_samples
                segment = processed_signal_pu1[start:end]
                signalsRe.append(segment)

        QTDatabaseSignals[register_name] = signalsRe
        
    with open('data/QTDatabase.pkl', 'wb') as output:
        pickle.dump(QTDatabaseSignals, output)
    print('=========================================================')
    print('MIT QT database saved as pickle file')
