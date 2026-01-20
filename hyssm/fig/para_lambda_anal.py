import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 全局字体与样式设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ==========================================
# 2. 数据准备 (补全后的数据)
# ==========================================
# X轴: Lambda Task 值
lambda_values = ['0.01', '0.05', '0.1', '0.2', '0.5']
x_indices = np.arange(len(lambda_values)) 

# MOSI 数据 [ACC2, F1] (ACC7 数据未绘制)
mosi_acc2 = [72.5, 74.8, 76.4, 75.6, 73.1]
mosi_f1   = [72.8, 75.1, 76.3, 75.4, 72.9]

# MOSEI 数据 [ACC2, F1]
mosei_acc2 = [74.8, 76.1, 77.4, 76.5, 75.2]
mosei_f1   = [74.2, 74.8, 75.0, 74.8, 73.9]

# ==========================================
# 3. 绘图逻辑
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5)) 
plt.subplots_adjust(wspace=0.25)

# --- 绘制图 (a) MOSI ---
ax1 = axes[0]
ax1.plot(x_indices, mosi_acc2, label='ACC2', color='#d62728', marker='o', linestyle='-', linewidth=2)
ax1.plot(x_indices, mosi_f1,   label='F1',   color='#1f77b4', marker='s', linestyle='--', linewidth=2)

ax1.set_title('(a) Sensitivity of $\lambda_{task}$ on CMU-MOSI', fontsize=14, y=-0.2)
ax1.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Weight of Task Loss ($\lambda_{task}$)', fontsize=12, fontweight='bold')
ax1.set_xticks(x_indices)
ax1.set_xticklabels(lambda_values)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_ylim(70, 80) # 设置Y轴范围

# --- 绘制图 (b) MOSEI ---
ax2 = axes[1]
ax2.plot(x_indices, mosei_acc2, label='ACC2', color='#d62728', marker='o', linestyle='-', linewidth=2)
ax2.plot(x_indices, mosei_f1,   label='F1',   color='#1f77b4', marker='s', linestyle='--', linewidth=2)

ax2.set_title('(b) Sensitivity of $\lambda_{task}$ on CMU-MOSEI', fontsize=14, y=-0.2)
# ax2.set_ylabel('Performance (%)', fontsize=12) 
ax2.set_xlabel('Weight of Task Loss ($\lambda_{task}$)', fontsize=12, fontweight='bold')
ax2.set_xticks(x_indices)
ax2.set_xticklabels(lambda_values)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_ylim(70, 80) # 保持一致的Y轴范围

# ==========================================
# 4. 全局图例与保存
# ==========================================
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
           ncol=2, fontsize=12, frameon=False)

plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.2) 

# 保存图片
plt.savefig('/root/FoldFish/code/cora-diff/fig/parameter_lambda_analysis.png', dpi=300, bbox_inches='tight')
plt.show()