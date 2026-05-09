import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import stats
import pandas as pd
import pingouin as pg
import pickle
import matplotlib.pyplot as plt

def align_and_stats(gt, sig, avgrr, sampling_rate=360):
    n_gt = len(gt)
    aligned_sig = np.full(n_gt, np.nan)
    i_sig = 0
    n_sig = len(sig)
    tolerance_sec = 0.2
    tolerance_samples = int(tolerance_sec * sampling_rate)
    for i_gt in range(n_gt):
        gt_val = gt[i_gt]
        lower_bound = gt_val - tolerance_samples
        upper_bound = gt_val + tolerance_samples
        while i_sig < n_sig and sig[i_sig] < lower_bound:
            i_sig += 1
        if i_sig == n_sig:
            continue
        if sig[i_sig] < upper_bound:
            aligned_sig[i_gt] = sig[i_sig]
            i_sig += 1
    false_negatives = np.sum(np.isnan(aligned_sig))
    true_positives = len(aligned_sig) - false_negatives
    false_positives = len(sig) - true_positives
    absolute_error = 0
    for i in range(len(aligned_sig)):
        if not np.isnan(aligned_sig[i]):
            absolute_error += np.abs(gt[i] - aligned_sig[i])
    absolute_error += false_negatives * avgrr

    return false_negatives, true_positives, false_positives, absolute_error


def run_analysis(signal_array, gt_array, sampling_rate=360):
    """
    主函数
    :param signal_array: np.array (5024, 3600)
    :param ann_list: list of np.array (2, n)
    :param sampling_rate: 采样率 (Hz), 请根据实际情况修改 (常见为 360, 500, 1000)
    """
    
    print(f"开始处理 {len(signal_array)} 条信号，采样率设为 {sampling_rate} Hz...")
    false_negatives = np.full(5, 0)
    true_positives = np.full(5, 0)
    false_positives = np.full(5, 0)
    absolute_error = np.full(5, 0)
    count = len(signal_array)
    for i in range(len(signal_array)):
        sig = signal_array[i]
        gt = gt_array[i]
        gt = nk.ecg_clean(gt, sampling_rate=sampling_rate)
        sig = nk.ecg_clean(sig, sampling_rate=sampling_rate)
        # if len(r_peaks_gt) != len(r_peaks_sig):
        #     print(f"警告：第 {i} 条信号 r_peaks 数量不匹配，跳过")
        #     continue
        # --- 1. 计算 Ground Truth ---
        try:
            gt_info = nk.ecg_findpeaks(gt, sampling_rate=sampling_rate)
            gt_r_peaks = gt_info['ECG_R_Peaks']
            _, waves_peak = nk.ecg_delineate(gt, gt_r_peaks, sampling_rate=sampling_rate, method="peak", show=False, show_type="all")
            # plt.show()
            # plt.close()
            gt_p_peaks = waves_peak['ECG_P_Peaks']
            gt_q_peaks = waves_peak['ECG_Q_Peaks']
            gt_s_peaks = waves_peak['ECG_S_Peaks']
            gt_t_peaks = waves_peak['ECG_T_Peaks']
        except Exception as e:
            print(f"警告：第 {i} 条gt信号处理失败：{e}")
            gt_r_peaks = []
            gt_p_peaks = []
            gt_q_peaks = []
            gt_s_peaks = []
            gt_t_peaks = []
            count -= 1
            continue

        # gt_p_peaks = 1000 * np.array(gt_p_peaks) / 360
        # gt_q_peaks = 1000 * np.array(gt_q_peaks) / 360
        # gt_r_peaks = 1000 * np.array(gt_r_peaks) / 360
        # gt_s_peaks = 1000 * np.array(gt_s_peaks) / 360
        # gt_t_peaks = 1000 * np.array(gt_t_peaks) / 360
        gt_lists = [gt_p_peaks, gt_q_peaks, gt_r_peaks, gt_s_peaks, gt_t_peaks]
        r_peaks = np.array(gt_r_peaks)
        avgrr = np.mean(np.diff(r_peaks))

        try:        
            sig_info = nk.ecg_findpeaks(sig, sampling_rate=sampling_rate)
            sig_r_peaks = sig_info['ECG_R_Peaks']
            _, waves_peak = nk.ecg_delineate(sig, sig_r_peaks, sampling_rate=sampling_rate, method="peak", show=False, show_type="all")
            # plt.show()
            # plt.close()
            sig_p_peaks = waves_peak['ECG_P_Peaks']
            sig_q_peaks = waves_peak['ECG_Q_Peaks']
            sig_s_peaks = waves_peak['ECG_S_Peaks']
            sig_t_peaks = waves_peak['ECG_T_Peaks']
        except Exception as e:
            print(f"警告：第 {i} 条sig信号处理失败：{e}")
            sig_r_peaks = []
            sig_p_peaks = []
            sig_q_peaks = []
            sig_s_peaks = []
            sig_t_peaks = []
        # sig_p_peaks = 1000 * np.array(sig_p_peaks) / 360
        # sig_q_peaks = 1000 * np.array(sig_q_peaks) / 360
        # sig_r_peaks = 1000 * np.array(sig_r_peaks) / 360
        # sig_s_peaks = 1000 * np.array(sig_s_peaks) / 360
        # sig_t_peaks = 1000 * np.array(sig_t_peaks) / 360
        sig_lists = [sig_p_peaks, sig_q_peaks, sig_r_peaks, sig_s_peaks, sig_t_peaks]
        for j in range(5):
            gt_current = gt_lists[j]
            sig_current = sig_lists[j]
            fn, tp, fp, ae = align_and_stats(gt_current, sig_current, avgrr, sampling_rate=sampling_rate)
            false_negatives[j] += fn
            true_positives[j] += tp
            false_positives[j] += fp
            absolute_error[j] += ae

    print("处理完成。")
    print(f"最后剩余 {count} 条信号")
    peak_types = ['P', 'Q', 'R', 'S', 'T']
    for i, peak_type in enumerate(peak_types):
        precision = true_positives[i] / (true_positives[i] + false_positives[i])
        recall = true_positives[i] / (true_positives[i] + false_negatives[i])
        f1_score = 2 * precision * recall / (precision + recall)
        mean_absolute_error = absolute_error[i] / (true_positives[i] + false_negatives[i])
        mean_absolute_error = 1000 * mean_absolute_error / sampling_rate
        print("\n" + "="*30)
        print(f"{peak_type} Peak:")
        print(f"总心拍数: {true_positives[i] + false_negatives[i]}")
        print(f"总检测数: {true_positives[i] + false_positives[i]}")
        print(f"精确度: {precision}")
        print(f"召回率: {recall}")
        print(f"F1 分数: {f1_score}")
        print(f"平均绝对误差: {mean_absolute_error}")
        print("="*30)

# ==========================================
if __name__ == "__main__":
    # 假设采样率
    SAMPLING_RATE = 360  
    gt = np.load(r".\cleanECG.npy")[:800]
    # signal = np.load(r".\noisyECG.npy")[:800]
    signal = np.load(r".\TFCDiff-10.npy")[:800]
    # signal = np.load(r".\DesCod-10.npy")[:800]
    # with open('./test_results_IIR_nv1.pkl', 'rb') as f:
    #     data_list = pickle.load(f)
    # signal = data_list[2][:800]
    # 运行分析 (请将 fake_signal 和 fake_ann 替换为您的 signal 和 ann)
    # 注意：由于 fake_signal 是随机噪声，NK2 的结果可能与伪造的 GT 差异巨大，这是正常的。
    # 关键在于代码流程是否跑通。
    res = run_analysis(signal, gt, sampling_rate=SAMPLING_RATE)
    print("分析完成。")