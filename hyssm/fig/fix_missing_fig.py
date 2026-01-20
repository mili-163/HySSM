import matplotlib.pyplot as plt
import numpy as np
import re

# =====================
# 1. 基本设置
# =====================
MR = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

def clean_name(name):
    """去掉方法名中的 _数字 后缀"""
    return re.sub(r'_\d+$', '', name)

# 其他方法：浅色
OTHER_ALPHA = 0.35
OTHER_LW = 1.6

# ours：高亮
OURS_COLOR = '#1a9c2c'
OURS_LW = 3.0
OURS_ALPHA = 1.0
OURS_ZORDER = 10

# =====================
# 2. 数据（示例：MOSI）
# =====================
# 格式：
# method: (ACC2_list, F1_list, ACC7_list)

MOSI = {
    "CCA_92": (
        [74.7,71.6,68.8,65.7,62.1,60.4,59.9,53.3],
        [74.1,71.2,68.4,65.0,61.0,59.5,57.4,53.8],
        [29.9,27.6,25.6,26.3,23.0,21.4,21.9,19.2]
    ),
    "AE_06": (
        [85.9,82.3,78.4,74.0,69.8,66.6,60.2,55.6],
        [85.2,83.1,80.3,77.3,72.1,67.9,64.3,58.7],
        [45.8,41.9,39.8,35.7,34.2,33.3,29.0,28.5]
    ),
    "CorrKD_24": (
        [86.0,85.5,83.5,82.0,80.8,77.6,74.4,72.5],
        [86.0,85.5,83.6,82.0,80.2,77.7,74.5,72.7],
        [45.1,45.3,45.0,42.0,39.5,37.7,34.7,30.5]
    ),
    "ours": (
        [89.5,88.5,85.7,84.3,82.2,78.6,77.4,76.4],
        [89.5,88.5,85.7,84.6,82.2,78.8,77.4,76.3],
        [48.2,46.4,45.3,43.0,40.1,39.0,37.8,36.0]
    )
}

# =====================
# 3. MOSEI 数据（示例）
# =====================
MOSEI = {
    "CCA_92": (
        [81.2,79.1,78.1,76.3,74.6,73.2,70.6,56.3],
        [80.9,78.8,76.7,72.8,71.3,70.7,70.8,58.2],
        [47.9,46.9,46.1,45.8,45.3,44.9,44.1,30.8]
    ),
    "CorrKD_24": (
        [85.7,84.2,82.6,80.8,79.5,77.0,76.6,76.0],
        [85.6,84.2,82.7,80.8,79.3,76.1,76.5,75.7],
        [52.0,51.6,51.4,49.8,49.1,47.9,47.8,47.5]
    ),
    "ours": (
        [87.6,86.4,85.0,82.8,80.5,79.2,78.4,77.4],
        [87.8,86.5,85.0,82.6,80.5,78.7,77.7,75.0],
        [56.7,55.4,54.5,53.9,52.5,51.0,50.3,48.5]
    )
}

# =====================
# 4. 绘图函数
# =====================
def plot_block(ax, data, metric_idx, title, ylabel):
    for method, values in data.items():
        name = clean_name(method)
        y = values[metric_idx]

        if name.lower() == "ours":
            ax.plot(
                MR, y,
                color=OURS_COLOR,
                linewidth=OURS_LW,
                alpha=OURS_ALPHA,
                label=name,
                zorder=OURS_ZORDER
            )
        else:
            ax.plot(
                MR, y,
                linewidth=OTHER_LW,
                alpha=OTHER_ALPHA,
                label=name
            )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("MR")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0, 0.7)

# =====================
# 5. 主绘图
# =====================
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

# MOSI
plot_block(axes[0,0], MOSI, 0, "(a) ACC2 on CMU-MOSI", "ACC2 (%)")
plot_block(axes[0,1], MOSI, 1, "(b) F1 on CMU-MOSI", "F1 (%)")
plot_block(axes[0,2], MOSI, 2, "(c) ACC7 on CMU-MOSI", "ACC7 (%)")

# MOSEI
plot_block(axes[1,0], MOSEI, 0, "(d) ACC2 on CMU-MOSEI", "ACC2 (%)")
plot_block(axes[1,1], MOSEI, 1, "(e) F1 on CMU-MOSEI", "F1 (%)")
plot_block(axes[1,2], MOSEI, 2, "(f) ACC7 on CMU-MOSEI", "ACC7 (%)")

# 统一图例（底部）
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=6,
    frameon=False,
    fontsize=10
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("/root/FoldFish/code/cora-diff/fig/fix_missing_fig.png", dpi=300)
