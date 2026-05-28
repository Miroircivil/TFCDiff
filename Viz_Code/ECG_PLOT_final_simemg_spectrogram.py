# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:59:50 2026

@author: 李旺
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from scipy import signal

def amplitude_to_db(S, ref=np.max, amin=1e-10, top_db=80.0):
    """
    将幅度谱转换为分贝谱。
    S: 幅度谱矩阵
    ref: 参考值，用于归一化 (0dB 基准)
    amin: 最小阈值，防止 log(0)
    """
    # 确定参考值
    if callable(ref):
        reference = ref(S)
    else:
        reference = ref
    
    # 避免除以零或负数
    reference = np.maximum(reference, amin)
    
    # 计算分贝
    log_spec = 20.0 * np.log10(np.maximum(S, amin))
    log_spec -= 20.0 * np.log10(reference)
    
    # 可选：截断低于 top_db 的值 (librosa 默认行为，此处为了还原原逻辑暂不开启强制截断，依靠 vmin 控制显示)
    return log_spec

# ========== 3. 参数设置 ==========
fs = 360         # 采样频率 (Hz)
n_fft = 64      # FFT 窗口大小 (对应 scipy 的 nperseg)
hop_length = 16 # 步长
win_length = 64 # 窗长 (scipy 中通常 nperseg=win_length)
# 计算重叠点数：noverlap = nperseg - hop_length
noverlap = n_fft - hop_length

# ========== 学术期刊参数设置 ==========
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 1.0
})


# 12个ECG的名称
lead_names = ['Noisy','Denoised', 'Ground Truth']

indices = [8, 19]
ecg_clean = np.load(r".\clean_simemg.npy")[indices]
ecg_noisy = np.load(r".\noisy_simemg.npy")[indices]
ecg_output = np.load(r".\output_simemg.npy")[indices]

all_signals = [ecg_noisy, ecg_output, ecg_clean]

# 生成6条ECG信号
ecg_signals = []
for i in range(3):
    for j in range(2):
        ecg_signals.append(all_signals[i][j,:])

# ========== 创建3x4子图 ==========
fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
fig.subplots_adjust(hspace=0.2, wspace=0.25)

# 颜色方案（使用色盲友好的颜色映射）
# colors = plt.cm.plasma(np.linspace(0.2, 0.8, 12))

colors = ['black']*2 + ['#2E86AB']*2 + ['#2A9D8F']*2 

# ========== 绘制6条ECG曲线 ==========
for idx, ax in enumerate(axes.flat):
    # --- Scipy STFT 计算 ---
    # critical parameters to match librosa center=False:
    # boundary=None: 不进行边界填充
    # padded=False: 不对信号末尾进行零填充
    # nperseg: 窗口长度
    # noverlap: 重叠长度
    f, t, Zxx = signal.stft(
        ecg_signals[idx], 
        fs=fs, 
        nperseg=n_fft, 
        noverlap=noverlap, 
        nfft=n_fft, 
        window='hann',      # librosa 默认也是 hann 窗
        boundary=None,      # 关键：匹配 librosa center=False
        padded=False,       # 关键：匹配 librosa center=False
        axis=-1
    )
    
    # 获取幅度谱
    S_mag = np.abs(Zxx)
    
    # 转换为分贝 (模拟 librosa.amplitude_to_db)
    # ref=np.max 表示将当前信号的最大幅度设为 0dB
    S_db = amplitude_to_db(S_mag, ref=np.max)
    
    # --- Matplotlib 原生绘图 (替代 librosa.display.specshow) ---
    # pcolormesh 需要 X 和 Y 的边缘坐标，或者直接传入中心点坐标 (shading='gouraud' 或 'nearest')
    # 为了与 specshow 效果最接近，我们使用 shading='gouraud' 或直接让 pcolormesh 处理网格
    # 注意：t 和 f 是中心点坐标，pcolormesh 会自动处理网格边界
    
    im = ax.pcolormesh(
        t,          # 时间轴 (seconds)
        f,          # 频率轴 (Hz)
        S_db,       # 强度值 (dB)
        shading='gouraud', # 平滑着色，类似 specshow
        cmap='magma',      # 使用与原代码注释中一致的配色，也可换回默认
        vmin=-60, 
        vmax=0
    )
        # 添加导联名称（左下角）
    if idx%2 ==0:
        ax.text(0.02, 0.95, lead_names[int(idx/2)], transform=ax.transAxes,
            fontsize=8, fontweight='bold', color=colors[idx],
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0, edgecolor='none'))
    # --- 美化修饰 ---
    ax.tick_params(axis='both', which='major', direction='in', length=3, width=0.8)
    
    # 只有最后一行显示 X 轴标签
    if idx < 4:
        ax.set_xlabel('') 
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel('Time (seconds)')
        # ax.set_xlim(t.min(), t.max())
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        
    
    # 限制频率范围 (ECG 重点关注 0-50Hz)
    ax.set_ylim(0, 50)
    if not idx % 2:
        ax.set_ylabel('Frequency (Hz)')
    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False)
    # ax.set_yticks(np.arange(0, 51, 10))
    # ax.yaxis.set_major_locator(MultipleLocator(10))
    
    ax.grid(False)

# ========== 5. 添加统一 Colorbar ==========
# 调整位置以适应布局
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7]) 
cbar = fig.colorbar(im, cax=cbar_ax, format="%+2.0f dB")
cbar.set_label('Relative Power (dB)', rotation=270, labelpad=15)
cbar.ax.tick_params()

# 自动调整布局，防止 title 或 label 被遮挡
# rect 参数为 colorbar 留出右侧空间
plt.tight_layout(rect=[0, 0, 0.84, 1]) 

plt.show()