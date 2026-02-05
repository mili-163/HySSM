import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 全局样式设置 (学术风格)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# ==========================================
# 2. 数据准备
# ==========================================
variants = ['Ours', 'w/o Semantic Planning', 'w/o Re-ranking', 'w/o Evidence Retrieval', 'w/o Sparse Attn']

# CMU-MOSI 数据
mosi_data = {
    'ACC2': [76.4, 73.2, 74.8, 72.8, 73.5],
    'F1':   [76.3, 73.0, 74.3, 73.2, 73.3],
    'ACC7': [36.0, 28.1, 30.3, 28.1, 29.4]
}

# CMU-MOSEI 数据
mosei_data = {
    'ACC2': [77.4, 72.1, 74.3, 74.0, 72.2],
    'F1':   [75.0, 72.5, 73.6, 73.8, 72.2],
    'ACC7': [48.5, 42.2, 44.6, 44.3, 43.5]
}

# 定义颜色：Ours用红色高亮，其他用不同的冷色调
colors = ['#d62728', '#aec7e8', '#1f77b4', '#9467bd', '#c5b0d5']
# 如果你想让其他变体颜色统一，可以用这个：
# colors = ['#d62728', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4'] 

# ==========================================
# 3. 绘图逻辑
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.25) # 调整子图间距

# 指标列表
metrics = ['ACC2', 'F1', 'ACC7']

# --- 第一排：CMU-MOSI ---
for i, metric in enumerate(metrics):
    ax = axes[0, i]
    values = mosi_data[metric]
    
    # 绘制柱状图
    bars = ax.bar(variants, values, color=colors, edgecolor='black', alpha=0.9, width=0.6)
    
    # 设置标题和标签
    ax.set_title(f'(MOSI) {metric}', fontsize=14, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    # 自动调整Y轴范围，让差异更明显
    min_val = min(values)
    max_val = max(values)
    margin = (max_val - min_val) * 0.5
    ax.set_ylim(min_val - margin, max_val + margin * 0.5)
    
    # x轴标签旋转，防止重叠
    ax.set_xticklabels(variants, rotation=30, ha='right', fontsize=10)
    
    # 在柱子上显示数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- 第二排：CMU-MOSEI ---
for i, metric in enumerate(metrics):
    ax = axes[1, i]
    values = mosei_data[metric]
    
    # 绘制柱状图
    bars = ax.bar(variants, values, color=colors, edgecolor='black', alpha=0.9, width=0.9)
    
    # 设置标题和标签
    ax.set_title(f'(MOSEI) {metric}', fontsize=14, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    # 自动调整Y轴范围
    min_val = min(values)
    max_val = max(values)
    margin = (max_val - min_val) * 0.5
    ax.set_ylim(min_val - margin, max_val + margin * 0.5)
    
    # x轴标签旋转
    ax.set_xticklabels(variants, rotation=30, ha='right', fontsize=10)
    
    # 在柱子上显示数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# 添加左侧统一的Y轴标签（可选）
axes[0, 0].set_ylabel('Performance Score', fontsize=12)
axes[1, 0].set_ylabel('Performance Score', fontsize=12)

# ==========================================
# 4. 保存与显示
# ==========================================
# 如果需要图例，可以在最下方添加（虽然x轴已经有标签了，但为了美观可以加一个）
# 这里的 trick 是创建一个不在视野内的图例句柄
handles = [plt.Rectangle((0,0),1,1, color=c) for c in colors]
fig.legend(handles, variants, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
           ncol=5, fontsize=12, frameon=False)

plt.tight_layout()
# 给顶部图例留出空间
plt.subplots_adjust(top=0.88) 

plt.savefig('ablation_study_bars.png', dpi=300, bbox_inches='tight')
plt.show()