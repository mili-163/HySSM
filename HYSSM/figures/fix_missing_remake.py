import matplotlib.pyplot as plt
import numpy as np

raw_mosi = """
CCA	74.7/74.1/29.9	71.6/71.2/27.6	68.8/68.4/25.6	65.7/65/26.3	62.1/61/23	60.4/59.5/21.4	59.9/57.4/21.9	53.3/53.8/19.2
AE	85.9/85.2/45.8	82.3/83.1/41.9	78.4/80.3/39.8	74/77.3/35.7	69.8/72.1/34.2	66.6/67.9/33.3	60.2/64.3/29	55.6/58.7/28.5
DCCA	75.3/75.4/30.5	72.1/72.8/28	69.3/69.1/25.8	65.4/65.2/25.7	62.8/62/24.2	60.9/59.9/21.6	58.6/57.3/21.2	57.4/56/20.4
DCCAE	77.3/77.4/31.2	74.5/74.7/28.1	71.8/71.9/27.6	67/66.7/25.8	63.6/62.8/24.2	62/61.3/23	59.6/58.5/20.9	58.1/57.4/20.6
CRA	85.7/85/45.1	82.6/81.9/42.2	78.5/79.2/40.2	75.1/76.2/35.6	70.2/71.4/34.4	67.4/67.8/33.7	62.4/64.7/30.3	59.4/59.2/27.1
MCTN	81.4/81.5/43.4	78.4/78.5/39.8	75.6/75.7/38.5	71.3/71.2/35.5	68/67.6/32.9	65.4/64.8/31.2	63.8/62.5/29.7	61.2/59.9/27.5
MMIN	84.6/84.4/44.8	81.8/81.8/41.2	79/79.1/38.9	76.1/76.2/36.9	71.7/71.6/34.9	67.2/66.5/32.2	64.9/64/29.1	62.8/61/28.4
TATE	83.3/83/44.7	83.2/82.6/42.4	80.8/81.7/38.6	79.9/80.6/36.5	77.8/79.3/34.7	73/73.7/32.7	68.3/64.1/31.1	61.9/62.4/28.8
MPMM	83.2/83.8/43.2	81/80.8/41.3	79/78.9/39.7	77.5/77.2/38.9	76/74.8/38	74/74/37.5	73.2/73.4/36.8	71.3/71.9/35.6
IMDer	85.7/85.6/45.3	84.8/84.8/44.8	83.5/83.4/44.3	81.2/81/42.5	78.6/78.1/39.7	76.2/75.9/37.9	74.7/74/35.8	71.9/71.2/33.4
MPLMM	83.6/84.1/44.9	81.7/82.2/42.1	80.2/80.2/40.2	78.4/78.4/39.4	77.1/77/38.3	76.8/77.2/37.1	76.1/76.6/36.7	74.9/76.3/35.1
SMCSMA	85.8/86.1/39.2	85.1/85.4/38	83.6/83.3/37.2	81.4/81.1/36.3	79.7/79.3/35.3	77.4/77.5/33.7	75.0/73.5/27.6	72.2/71.5/28.0
CorrKD	86.0/86.0/45.1	85.5/85.5/45.3	83.5/83.6/45.0	82.0/82.0/42.0	80.8/80.2/39.5	77.6/77.7/37.7	74.4/74.5/34.7	72.5/72.7/30.5
PMSM	86.0/86.0/44.5	85.4/85.2/43.2	84.7/84.8/39.1	83.5/83.4/37.6	78.5/77.8/36.6	76.4/76.2/28.0	75.8/74.6/25.2	72.0/69.6/23.2
Ours	89.5/89.5/48.2	88.5/88.5/46.4	85.7/85.7/45.3	84.3/84.6/43.0	82.2/82.2/40.1	78.6/78.8/39.0	77.4/77.4/37.8	76.4/76.5/36.0
"""

raw_mosei = """
CCA	81.2/80.9/47.9	79.1/78.8/46.9	78.1/76.7/46.1	76.3/72.8/45.8	74.6/71.3/45.3	73.2/70.7/44.9	70.6/70.8/44.1	56.3/58.2/30.8
AE	86.7/87.5/53.3	84.4/84.8/52.7	82.6/83.1/52.3	80.6/80.3/51.9	78.8/78.7/50.8	76.4/76.5/47	74.3/74.4/45.9	72.8/72.7/42.9
DCCA	85.7/85.6/45.3	84.9/84.8/44.8	83.5/83.4/44.3	81.2/81/42.5	78.6/78.1/39.7	76.2/75.9/37.9	74.7/74/35.8	71.9/71.2/33.4
DCCAE	81.2/81.2/48.2	78.4/78.3/46.9	75.5/75.4/46.3	72.3/72.2/45.6	70.3/70/44	69.2/66.4/43.3	67.6/63.2/42.9	66.6/62.6/42.5
CRA	86.5/86.3/53.9	84.2/84.5/52.9	82.3/82.7/52.5	80.1/79.9/50.4	78.6/78.5/50.3	75.9/75.7/48	74.1/74.3/44.7	72.5/72.4/43.1
MCTN	84.2/84.2/51.2	81.8/81.6/49.8	79/78.7/48.6	76.9/76.2/47.4	74.3/74.1/45.6	73.6/72.6/45.1	73.2/71.1/43.8	72.7/70.5/43.6
MMIN	84.3/84.2/52.4	81.9/81.3/50.6	79.8/78.8/49.6	77.2/75.5/48.1	75.2/72.6/47.5	73.9/70.7/46.7	73.2/70.3/45.6	73.1/69.5/44.8
TATE	82.9/84.1/50.1	83.4/83.9/49.8	81.8/81.1/49.4	80.4/79.5/48.8	78.1/77.8/47.8	75.7/75.6/45.1	68.5/64.6/33	62.3/62.9/30.2
MPMM	83.8/82.5/47.6	79/79.5/46.7	78.1/77.3/45.9	76.7/75.3/47.7	74.3/73.9/43.1	73.6/73.5/42.1	72.6/72.4/35.8	72.4/71.5/33.7
IMDer	85.1/85.1/53.4	84.8/84.6/53.1	82.7/82.4/52.0	81.3/80.7/51.3	79.3/78.1/50.0	79/77.4/49.2	78/75.5/48.5	77.3/74.6/47.6
MPLMM	85.1/84.5/53.6	83.5/82.8/51.5	82.1/81.6/48.7	78.5/78.8/46.8	77.6/77.6/45.7	76.6/76.7/44.8	75.4/75.6/43	74.3/74.1/42.2
SMCSMA	80.6/80.9/50.7	79.0/79.3/50.4	77.0/77.3/50.3	76.2/76.5/50.0	74.5/74.7/49.2	72.1/72.2/49.1	71.2/71.2/48.7	70.3/70.4/48.0
CorrKD	85.7/85.6/52.0	84.2/84.2/51.6	82.6/82.7/51.4	80.8/80.8/49.8	79.5/79.3/49.1	77.0/76.1/47.9	76.6/76.5/47.8	76.0/75.7/47.5
PMSM	85.1/84.8/53.4	84.0/83.9/51.7	82.5/82.5/49.2	80.8/80.8/48.2	79.4/79.6/47.9	77.8/78.1/47.3	76.5/77.0/43.3	72.0/72.7/40.7
Ours	87.6/87.8/56.7	86.4/86.5/55.4	85.0/85.0/54.5	82.8/82.6/53.9	80.5/80.5/52.5	79.2/78.7/51.0	78.4/77.7/50.3	77.4/75.0/48.5
"""

def parse_data(raw_text):
    data_dict = {}
    lines = raw_text.strip().split('\n')
    for line in lines:
        line_clean = line.replace(' /', '/').replace('/ ', '/').replace(' / ', '/')
        
        parts = line_clean.split()
        if not parts: continue
        name = parts[0]
        
        values = []
        for p in parts[1:]:
            try:
                v = [float(x) for x in p.split('/') if x.strip() != '']
                if len(v) > 0:
                    values.append(v)
            except ValueError:
                print(f"警告: 无法解析数据块 '{p}'，已跳过。")
                continue
                
        if values:
            data_dict[name] = np.array(values)
            
    return data_dict

mosi_data = parse_data(raw_mosi)
mosei_data = parse_data(raw_mosei)


mr_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
method_list = list(mosi_data.keys())

markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '1']
colors = plt.cm.tab20(np.linspace(0, 1, len(method_list)))

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

plt.subplots_adjust(bottom=0.17, wspace=0.15, hspace=0.15) 

titles_row1 = ['(a) ACC2 on CMU-MOSI', '(b) F1 on CMU-MOSI', '(c) ACC7 on CMU-MOSI']
titles_row2 = ['(d) ACC2 on CMU-MOSEI', '(e) F1 on CMU-MOSEI', '(f) ACC7 on CMU-MOSEI']
y_labels = ['ACC2(%)', 'F1(%)', 'ACC7(%)']

def plot_row(row_idx, dataset_data, titles):
    for col_idx in range(3): 
        ax = axes[row_idx, col_idx]
        for idx, method in enumerate(method_list):
            current_method_key = method
            if dataset_data is mosei_data and method == 'MPMM':
                 if 'MPMTM' in dataset_data: current_method_key = 'MPMTM'
            
            if current_method_key not in dataset_data: continue

            if col_idx < dataset_data[current_method_key].shape[1]:
                y_values = dataset_data[current_method_key][:, col_idx]
                
                if method.lower() == 'ours':
                    ax.plot(mr_rates, y_values, marker=markers[idx], label=method,
                            color='green', linewidth=2.5, markersize=7, zorder=10)
                else:
                    ax.plot(mr_rates, y_values, marker=markers[idx], label=method,
                            color=colors[idx], linewidth=1.2, markersize=5, alpha=0.8)

        ax.set_title(titles[col_idx], fontsize=14, pad=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        ax.set_ylabel(y_labels[col_idx], fontsize=12, fontweight='bold')

plot_row(0, mosi_data, titles_row1)
plot_row(1, mosei_data, titles_row2)

for ax in axes[1, :]:
    ax.set_xlabel("MR", fontsize=12, fontweight='bold')
    ax.set_xticks(mr_rates)


handles, labels = axes[0, 0].get_legend_handles_labels()

lgd = fig.legend(handles, labels, loc='lower center', 
                 bbox_to_anchor=(0.5, 0.02), 
                 ncol=8, 
                 fontsize=17, 
                 frameon=True, 
                 edgecolor='black')


for handle in lgd.get_lines():
    handle.set_markersize(10)

plt.savefig('/root/FoldFish/code/cora-diff/fig/result_chart.png', dpi=300, bbox_inches='tight')