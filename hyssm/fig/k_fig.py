import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 全局字体与样式设置 (学术风格)
# ==========================================
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ==========================================
# 2. 数据准备
# ==========================================
# X轴: K值
k_values = ['1', '5', '10', '20', '50']
x_indices = np.arange(len(k_values))  # 用于等间距绘图

# MOSI 数据 [ACC2, F1] (移除了 ACC7)
mosi_acc2 = [73.0, 75.3, 76.4, 75.5, 72.8]
mosi_f1   = [73.2, 75.3, 76.3, 75.0, 72.2]

# MOSEI 数据 [ACC2, F1] (移除了 ACC7)
mosei_acc2 = [74.2, 75.7, 77.4, 76.3, 74.5]
mosei_f1   = [73.8, 74.6, 75.0, 74.5, 73.4]

# ==========================================
# 3. 绘图逻辑
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # 1行2列
plt.subplots_adjust(wspace=0.25) # 调整子图间距

# --- 绘制图 (a) MOSI ---
ax1 = axes[0]
ax1.plot(x_indices, mosi_acc2, label='ACC2', color="#4201cec6", marker='o', linestyle='-', linewidth=2)
ax1.plot(x_indices, mosi_f1,   label='F1',   color="#a90269e7", marker='s', linestyle='--', linewidth=2)

ax1.set_title('(a) Parameter Analysis on CMU-MOSI', fontsize=14, y=-0.2)
ax1.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Plans ($K$)', fontsize=12, fontweight='bold')
ax1.set_xticks(x_indices)
ax1.set_xticklabels(k_values)
ax1.grid(True, linestyle='--', alpha=0.5)

# 关键修改：调整Y轴范围以聚焦 ACC2/F1 的变化趋势 (70-80)
# 如果使用原来的 (20, 85)，线条会非常平缓看不出趋势
ax1.set_ylim(70, 80) 

# --- 绘制图 (b) MOSEI ---
ax2 = axes[1]
ax2.plot(x_indices, mosei_acc2, label='ACC2', color='#4201cec6', marker='o', linestyle='-', linewidth=2)
ax2.plot(x_indices, mosei_f1,   label='F1',   color='#a90269e7', marker='s', linestyle='--', linewidth=2)

ax2.set_title('(b) Parameter Analysis on CMU-MOSEI', fontsize=14, y=-0.2)
# ax2.set_ylabel('Performance (%)', fontsize=12) 
ax2.set_xlabel('Number of Plans ($K$)', fontsize=12, fontweight='bold')
ax2.set_xticks(x_indices)
ax2.set_xticklabels(k_values)
ax2.grid(True, linestyle='--', alpha=0.5)

# 同样调整Y轴范围
ax2.set_ylim(71.0, 79.0)

# ==========================================
# 4. 全局图例与保存
# ==========================================
# 获取图例句柄
handles, labels = ax1.get_legend_handles_labels()

# 修改：ncol=2 (因为只有两个指标了)
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
           ncol=2, fontsize=12, frameon=False)

plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.2) 

# 保存图片
plt.savefig('/root/FoldFish/code/cora-diff/fig/parameter_k_analysis.png', dpi=300, bbox_inches='tight')
plt.show()