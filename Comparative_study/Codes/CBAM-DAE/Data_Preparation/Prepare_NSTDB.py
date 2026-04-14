import numpy as np
import wfdb
import _pickle as pickle
from scipy.signal import resample_poly

def prepare(NSTDBPath='data/mit-bih-noise-stress-test-database-1.0.0/'):
    # newFs = 400  # 新的采样频率
    # fs = 360  # 原始采样频率
    bw_signals, bw_fields = wfdb.rdsamp(NSTDBPath + 'bw')
    em_signals, em_fields = wfdb.rdsamp(NSTDBPath + 'em')
    ma_signals, ma_fields = wfdb.rdsamp(NSTDBPath + 'ma')

    for key in bw_fields:
        print(key, bw_fields[key])

    for key in em_fields:
        print(key, em_fields[key])

    for key in ma_fields:
        print(key, ma_fields[key])

    # # 归一化处理
    # def normalize(signals):
    #     min_val = np.min(signals, axis=0)
    #     max_val = np.max(signals, axis=0)
    #     normalized_signals = (signals - min_val) / (max_val - min_val)
    #     return normalized_signals
    # bw_signals = resample_poly(bw_signals, newFs, fs)
    # bw_signals = bw_signals[1:-1]  # 舍弃一头一尾两个点（避免边缘效应）
    # em_signals = resample_poly(em_signals, newFs, fs)
    # em_signals = em_signals[1:-1] 
    # ma_signals = resample_poly(ma_signals, newFs, fs)
    # ma_signals = ma_signals[1:-1] 
    # bw_signals = normalize(bw_signals)
    # em_signals = normalize(em_signals)
    # ma_signals = normalize(ma_signals)

    # Save Data
    with open('data/NoiseBWL.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump([bw_signals, em_signals, ma_signals], output)
    print('=========================================================')
    print('MIT BIH data noise stress test database (NSTDB) saved as pickle')
