# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:59:50 2026

@author: 李旺
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from matplotlib.ticker import MultipleLocator

# ========== 学术期刊参数设置 ==========
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 1.0
})


# 12个ECG的名称
lead_names = ['Raw','TFCDiff', 'DesCod', 'TCADE', 'FIR', 'IIR', 'DRNN', 'FCN-DAE', 'DeepFilter']

num = 2545
ecg_clean = np.load(r".\cleanECG.npy")[num]
ecg_noisy = np.load(r".\noisyECG.npy")[num]
ecg_TFCDiff = np.load(r".\TFCDiff-10.npy")[num]
ecg_TFCDiffwo = np.load(r".\TFCDiff-10-1.npy")[num]
ecg_DesCod = np.load(r".\DesCod-10.npy")[num]
with open('./test_results_Transformer_DAE_nv1.pkl', 'rb') as f:
        data_list = pickle.load(f)
ecg_TCADE = data_list[2][num]
with open('./test_results_FIR_nv1.pkl', 'rb') as f:
        data_list = pickle.load(f)
ecg_FIR = data_list[2][num]
ecg_FIR = np.pad(ecg_FIR, (0, 16), mode='constant', constant_values=0)
with open('./test_results_IIR_nv1.pkl', 'rb') as f:
        data_list = pickle.load(f)
ecg_IIR = data_list[2][num]
ecg_IIR = np.pad(ecg_IIR, (0, 16), mode='constant', constant_values=0)
with open('./test_results_DRNN_nv1.pkl', 'rb') as f:
        data_list = pickle.load(f)
ecg_DRNN = data_list[2][num]
ecg_DRNN = np.pad(ecg_DRNN, (0, 16), mode='constant', constant_values=0)
with open('./test_results_FCN-DAE_nv1.pkl', 'rb') as f:
        data_list = pickle.load(f)
ecg_FCN_DAE = data_list[2][num]
ecg_FCN_DAE = np.pad(ecg_FCN_DAE, (0, 16), mode='constant', constant_values=0)
with open('./test_results_Multibranch LANLD_nv1.pkl', 'rb') as f:
        data_list = pickle.load(f)
ecg_DeepFilter = data_list[2][num]
ecg_DeepFilter = np.pad(ecg_DeepFilter, (0, 16), mode='constant', constant_values=0)


ecg_signals = [ecg_noisy, ecg_TFCDiff, ecg_DesCod, ecg_TCADE, ecg_FIR, ecg_IIR, ecg_DRNN, ecg_FCN_DAE, ecg_DeepFilter]
# for i, signal in enumerate(ecg_signals):
#     print(signal.shape)

# ========== 创建3x3子图 ==========
fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
fig.subplots_adjust(hspace=0.2, wspace=0.25)

# 颜色方案（使用色盲友好的颜色映射）
# colors = plt.cm.plasma(np.linspace(0.2, 0.8, 12))
#  ['#2A9D8F']*4

colors = ['#E63946'] + ['#0072B2']*8  
t = np.arange(len(ecg_signals[0]))/360

# ========== 绘制9条ECG曲线 ==========
for idx, ax in enumerate(axes.flat):
    ecg = ecg_signals[idx]
    color = colors[idx]
    # col_idx = idx % 4
    # base_bg_color = "#D3D3D3"
    # alpha_val = 0.05 + col_idx * 0.25
    # ax.set_facecolor(base_bg_color)
    # ax.patch.set_alpha(alpha_val)
    # 绘制ECG信号
    if idx == 0:
        ax.plot(t, ecg, color=color, linewidth=0.6, alpha=0.9, label='Noisy ECG')
    else:
        ax.plot(t, ecg, color=color, linewidth=0.6, alpha=0.9, label='Denoised by '+lead_names[idx])  
    ax.plot(t, ecg_clean, color='#009E60', linewidth=1.2, alpha=0.5, label='Ground Truth')
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.legend(loc='upper left', fontsize=5, frameon=True, framealpha=0.8, edgecolor='none')
    
    # # 添加导联名称（左下角）
    # ax.text(0.02, 0.14, lead_names[idx], transform=ax.transAxes,
    #     fontsize=8, fontweight='bold', color=color,
    #     va='top', ha='left',
    #     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0, edgecolor='none'))
    
    # 添加子图标签 (a), (b), (c)... 按行排列
    # label_num = idx + 1
    # ax.text(0.02, 0.98, f'({chr(96+label_num)})', transform=ax.transAxes,
    #         fontsize=10, fontweight='bold', va='top', ha='left',
    #         bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
    
    # 刻度设置
    ax.tick_params(axis='both', which='major', direction='in', length=2, width=0.4, labelsize=6)
    ax.tick_params(axis='both', which='minor', direction='in', length=0, width=0.5)
    
    
    # ax.set_ylim(-1, 5)
    # # 设置Y轴标签（只显示左右边缘）
    if idx % 3 == 0:  # 第一列
        ax.set_ylabel('Amplitude (mV)', fontsize=7, fontweight='semibold')
    else:
        ax.set_ylabel('')
        # ax.tick_params(axis='y', labelleft=False)
    ax.yaxis.set_major_locator(MultipleLocator(0.4))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    # if idx <= 3:
    #     ax.set_ylim(-1.5, 1.5)
    #     y_ticks = [-1.0, 0.0, 1.0]
    #     ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    # else:
    #     ax.set_ylim(-0.8, 0.8)
    #     y_ticks = [-0.5, 0.0, 0.5]
    #     ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    
    # # 设置Y轴范围
    # y_max = max(abs(np.max(ecg)), abs(np.min(ecg))) * 1.1
    # ax.set_ylim(-y_max, y_max)
    
    # 添加网格线（仅水平方向）
    ax.grid(True, axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 添加心率标注
    # hr = hr_values[idx]
    # ax.text(0.98, 0.92, f'HR: {hr} bpm', transform=ax.transAxes,
    #         fontsize=7, va='top', ha='right',
    #         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # 添加ST段改变提示（如果有）
    # if st_levels[idx] != 0:
    #     st_text = 'some statistics' if st_levels[idx] > 0 else 'some other info'
    #     ax.text(0.98, 0.08, st_text, transform=ax.transAxes,
    #             fontsize=7, va='bottom', ha='right', color='red',
    #             fontweight='bold',
    #             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

# ========== X轴设置（只在最下方一行显示）==========
for ax in axes[-1, :]:
    ax.set_xlabel('Time (seconds)', fontsize=7, fontweight='semibold', labelpad=5)
    ax.set_xlim(t.min(), t.max())
    ax.set_xticks(np.arange(0, 11, 1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))

# 隐藏非底部子图的X轴标签
for ax in axes[:-1, :].flat:
    ax.tick_params(axis='x', labelbottom=False)

# ========== 添加整体标题 ==========
# fig.suptitle('12-Lead Electrocardiogram (ECG) Recording', 
#              fontsize=14, fontweight='bold', y=0.98)

# 添加时间刻度标签（底部）
# fig.text(0.5, 0.02, 'Time (seconds)', fontsize=11, fontweight='semibold', ha='center')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05)

plt.show()

# ========== 保存图片 ==========
# plt.savefig('12_lead_ecg.pdf', format='pdf', dpi=300, bbox_inches='tight')
# plt.savefig('12_lead_ecg.png', dpi=300, bbox_inches='tight', facecolor='white')