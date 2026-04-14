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

# 设置随机种子
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
    # 使用一次多项式函数拟合基线漂移
    coefficients = np.polyfit(h, segment, 1)
    g_h = np.polyval(coefficients, h)
    # 从原始信号中减去拟合的多项式函数
    x_hat = segment - g_h
    return x_hat

def find_baseline_point(segment, start_idx, direction, baseline):
    for i in range(start_idx, start_idx + direction * int(2 * len(segment) / 5), direction):
        if i < 0 or i >= len(segment):
            break
        if segment[i] == baseline:
            return i
        if (segment[i] - baseline) * (segment[i + direction] - baseline) < 0:
            return i

    return None
def hermite_interpolation(last_segment, current_segment):
    # 找基准点
    baseline = current_segment[0]
    p1_idx = find_baseline_point(last_segment, len(last_segment) - 1, -1, baseline)
    baseline = last_segment[-1]
    p2_idx = find_baseline_point(current_segment, 0, 1, baseline)

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

    # 获取 p1 和 p2 的值
    p1 = last_segment[p1_idx]
    p2 = current_segment[p2_idx]

    # 计算一阶导数
    derivative1 = (last_segment[p1_idx] - last_segment[p1_idx - 11]) / 10 if p1_idx > 0 else 0
    derivative2 = (current_segment[p2_idx + 11] - current_segment[p2_idx]) / 10 if p2_idx < len(current_segment) - 1 else 0

    # HERMITE 插值
    # 计算插值范围的长度
    x = np.array([0, len(last_segment) - p1_idx + p2_idx])
    y = np.array([p1, p2])
    dydx = np.array([0.05*derivative1, 0.05*derivative2])
    interp = CubicHermiteSpline(x, y, dydx)
    new_points = interp(np.arange(len(last_segment) - p1_idx + p2_idx + 1))

    # 替换原来的点
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

    # 处理最后一个片段
    segment = signal[prev_end:]
    segment = remove_baseline(segment)

    if len(segments) > 0:
        last_segment = segments[-1]
        last_segment, segment = hermite_interpolation(last_segment, segment)

    segments.append(segment)

    # 拼接所有片段
    return np.concatenate(segments)

#  # 归一化处理
# def normalize(signals):
#     min_val = np.min(signals, axis=0)
#     max_val = np.max(signals, axis=0)
#     normalized_signals = (signals - min_val) / (max_val - min_val)
#     return normalized_signals

def prepare(QTpath='./data/qt-database-1.0.0/'):
    # 目标采样频率
    newFs = 360  # 重采样到 400 Hz
    segment_length = 10  # 提取 10 秒长度的片段

    # 获取所有 .dat 文件的路径
    namesPath = glob.glob(QTpath + "/*.dat")

    # 最终存储所有信号和处理后的片段的字典
    QTDatabaseSignals = dict()

    for i in namesPath:
        # 读取信号文件
        aux = i.split('.dat')  # 去掉 .dat 后缀
        register_name = os.path.basename(aux[0])  # 取最后一个分割符后的字符串作为 register_name
        signal, fields = wfdb.rdsamp(aux[0])  # 读取信号数据和元信息
        fs = fields['fs']  # 原始采样频率
        # for key in fields:
        #    print(key, fields[key])

        # 初始化存储片段的列表
        signalsRe = list()

        # 提取两个通道的信号数据
        auxSig1 = signal[:, 0]  # 提取第一个通道的数据
        auxSig2 = signal[:, 1]  # 提取第二个通道的数据

        ######preprocessing######

        # 读取第一个通道的标注
        ann_pu0 = wfdb.rdann(aux[0], 'pu0')
        symbols_pu0 = ann_pu0.symbol
        samples_pu0 = ann_pu0.sample
        symbols_pu0 = np.array(symbols_pu0)

        # 读取第二个通道的标注
        ann_pu1 = wfdb.rdann(aux[0], 'pu1')
        symbols_pu1 = ann_pu1.symbol
        samples_pu1 = ann_pu1.sample
        symbols_pu1 = np.array(symbols_pu1)

        # 找到标注为'N'的索引
        n_indices_pu0 = np.where(symbols_pu0 == 'N')[0]
        n_indices_pu1 = np.where(symbols_pu1 == 'N')[0]

        # 处理第一个通道的信号
        auxSig1 = bandpass_filter(auxSig1, use_window=True)
        processed_signal_pu0 = process_signal(auxSig1, n_indices_pu0, samples_pu0)

        # 处理第二个通道的信号
        auxSig2 = bandpass_filter(auxSig2, use_window=True)
        processed_signal_pu1 = process_signal(auxSig2, n_indices_pu1, samples_pu1)

        # 对第一个通道整体重采样
        processed_signal_pu0 = resample_poly(processed_signal_pu0, newFs, fs)  # 重采样到 400 Hz
        processed_signal_pu0 = processed_signal_pu0[1:-1]  # 舍弃一头一尾两个点（避免边缘效应）

        # 对第二个通道整体重采样
        processed_signal_pu1 = resample_poly(processed_signal_pu1, newFs, fs)  # 重采样到 400 Hz
        processed_signal_pu1 = processed_signal_pu1[1:-1]  # 舍弃一头一尾两个点（避免边缘效应）

        # 计算 10 秒长度的样本点数
        segment_samples = 3584

        # 处理第一个通道的信号 - 原有的切分
        for start in range(0, len(processed_signal_pu0), segment_samples):
            end = start + segment_samples
            if end > len(processed_signal_pu0):  # 如果剩余信号不足 10 秒，则舍弃
                break
            segment = processed_signal_pu0[start:end]  # 提取 10 秒长度的片段
            # segment = normalize(segment)
            signalsRe.append(segment)  # 添加到 signalsRe 列表

        # 处理第一个通道的信号 - 随机采样
        for _ in range(90):
            if len(processed_signal_pu0) >= segment_samples:
                start = np.random.randint(0, len(processed_signal_pu0) - segment_samples + 1)
                end = start + segment_samples
                segment = processed_signal_pu0[start:end]
                # segment = normalize(segment)
                signalsRe.append(segment)

        # 处理第二个通道的信号 - 原有的切分
        for start in range(0, len(processed_signal_pu1), segment_samples):
            end = start + segment_samples
            if end > len(processed_signal_pu1):  # 如果剩余信号不足 10 秒，则舍弃
                break
            segment = processed_signal_pu1[start:end]  # 提取 10 秒长度的片段
            # segment = normalize(segment)
            signalsRe.append(segment)  # 添加到 signalsRe 列表

        # 处理第二个通道的信号 - 随机采样
        for _ in range(90):
            if len(processed_signal_pu1) >= segment_samples:
                start = np.random.randint(0, len(processed_signal_pu1) - segment_samples + 1)
                end = start + segment_samples
                segment = processed_signal_pu1[start:end]
                # segment = normalize(segment)
                signalsRe.append(segment)

        # 将处理后的片段存储到字典中
        QTDatabaseSignals[register_name] = signalsRe

    # 将数据保存为 pickle 文件
    with open('data/QTDatabase.pkl', 'wb') as output:
        pickle.dump(QTDatabaseSignals, output)
    print('=========================================================')
    print('MIT QT database saved as pickle file')
