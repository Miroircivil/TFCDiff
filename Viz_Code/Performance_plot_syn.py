# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:59:50 2026

@author: 李旺
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import torch
import torch_dct as dct

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
lead_names = ['Noisy','Denoised', 'Ground Truth']

indices = [4, 38, 15, 39]
ecg_clean = torch.FloatTensor(np.load(r".\clean.npy")[indices])
ecg_noisy = torch.FloatTensor(np.load(r".\noisy.npy")[indices])
ecg_output = torch.FloatTensor(np.load(r".\output.npy")[indices])
ecg_clean = dct.dct(ecg_clean, norm='ortho')[:, :1000].cpu().detach().numpy()
ecg_noisy = dct.dct(ecg_noisy, norm='ortho')[:, :1000].cpu().detach().numpy()
ecg_output = dct.dct(ecg_output, norm='ortho')[:, :1000].cpu().detach().numpy()

all_signals = [ecg_noisy, ecg_output, ecg_clean]

# 生成6条ECG信号
ecg_signals = []
for i in range(3):
    for j in range(4):
        ecg_signals.append(all_signals[i][j,:])

# ========== 创建3x4子图 ==========
fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
fig.subplots_adjust(hspace=0.2, wspace=0.25)

# 颜色方案（使用色盲友好的颜色映射）
# colors = plt.cm.plasma(np.linspace(0.2, 0.8, 12))

colors = ['#E63946']*4 + ['#2E86AB']*4 + ['#2A9D8F']*4 
freq = np.arange(len(ecg_signals[0])) / 20

# ========== 绘制6条ECG曲线 ==========
for idx, ax in enumerate(axes.flat):
    ecg = ecg_signals[idx]
    color = colors[idx]
    col_idx = idx % 4
    base_bg_color = "#D3D3D3"
    alpha_val = 0.05 + col_idx * 0.25
    ax.set_facecolor(base_bg_color)
    ax.patch.set_alpha(alpha_val)
    # 绘制ECG信号
    ax.plot(freq, ecg, color=color, linewidth=0.8, alpha=0.9)
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 添加导联名称（左下角）
    if idx%4 ==0:
        ax.text(0.02, 0.14, lead_names[int(idx/4)], transform=ax.transAxes,
            fontsize=8, fontweight='bold', color=color,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0, edgecolor='none'))
    
    # 添加子图标签 (a), (b), (c)... 按行排列
    # label_num = idx + 1
    # ax.text(0.02, 0.98, f'({chr(96+label_num)})', transform=ax.transAxes,
    #         fontsize=10, fontweight='bold', va='top', ha='left',
    #         bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
    
    # 刻度设置
    ax.tick_params(axis='both', which='major', direction='in', length=2, width=0.4)
    ax.tick_params(axis='both', which='minor', direction='in', length=0, width=0.5)
    
    # 设置Y轴标签（只显示左右边缘）
    if idx % 4 == 0:  # 第一列
        ax.set_ylabel('Amplitude (a.u.)', fontsize=9, fontweight='semibold')
    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False)

    if idx <= 3:
        ax.set_ylim(-5, 5)
        ax.yaxis.set_major_locator(MultipleLocator(2.0))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    else:
        ax.set_ylim(-2.5, 3.5)
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    
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
    ax.set_xlabel('Frequency (Hz)', fontsize=9, fontweight='semibold', labelpad=5)
    ax.set_xlim(freq.min(), freq.max())
    ax.set_xticks(np.arange(0, 51, 5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(7.5))

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