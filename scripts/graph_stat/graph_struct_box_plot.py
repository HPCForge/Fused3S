import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to remove trailing zeros from a pandas Series
def remove_trailing_zeros(s):
    s = s.dropna()
    arr = s.values
    idx = np.where(arr != 0)[0]
    if len(idx) == 0:
        return []
    last_non_zero_idx = idx[-1]
    return arr[:last_non_zero_idx+1]

# List of graphs
graphs = ['citeseer', 'cora', 'YeastH', 'OVCAR-8H', 'Yeast', 'DD', 'soc-BlogCatalog', 'web-BerkStan', 'reddit', 'ddi', 'protein']

# List of BLK_H and BLK_W combinations
block_sizes = [(8,32), (16,16), (8,8)]  # Added (8,8)

# Data dictionaries to hold dataframes
RO_nnzPerBlock_dict = {}
RO_nBlockPerRowWindow_dict = {}
OG_nnzPerBlock_dict = {}
OG_nBlockPerRowWindow_dict = {}

for BLK_H, BLK_W in block_sizes:
    suffix = f"{BLK_H}x{BLK_W}"
    # Read the CSV files and drop the first column
    try:
        RO_nnzPerBlock_dict[suffix] = pd.read_csv(f"RO_{suffix}_nnzPerBlock_pd.csv").iloc[:, 1:]
        RO_nBlockPerRowWindow_dict[suffix] = pd.read_csv(f"RO_{suffix}_nBlockPerRowWindow_pd.csv").iloc[:, 1:]
        OG_nnzPerBlock_dict[suffix] = pd.read_csv(f"OG_{suffix}_nnzPerBlock_pd.csv").iloc[:, 1:]
        OG_nBlockPerRowWindow_dict[suffix] = pd.read_csv(f"OG_{suffix}_nBlockPerRowWindow_pd.csv").iloc[:, 1:]
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        continue

# Series list: (label, size_label, dataframe, color, legend_label)
series_list = [
    ('RO', '16x16', RO_nnzPerBlock_dict.get('16x16'), 'lightcoral', 'RO 16x16'),
    ('OG', '16x16', OG_nnzPerBlock_dict.get('16x16'), 'lightgray', 'OG 16x16'),
    ('RO', '8x32', RO_nnzPerBlock_dict.get('8x32'), 'lightblue', 'RO 8x32'),
    ('OG', '8x32', OG_nnzPerBlock_dict.get('8x32'), 'lightgreen', 'OG 8x32'),
    ('RO', '8x8', RO_nnzPerBlock_dict.get('8x8'), 'gold', 'RO 8x8'),
    ('OG', '8x8', OG_nnzPerBlock_dict.get('8x8'), 'violet', 'OG 8x8'),
]

# Remove any None entries due to missing data
series_list = [s for s in series_list if s[2] is not None]

# Function to prepare data
def prepare_data(series_list):
    data = []
    positions = []
    n_bars_per_group = len(series_list)  # Number of series per group
    for i, graph in enumerate(graphs):
        group_pos = i * (n_bars_per_group + 1)
        for j, (label, size_label, df, color, legend_label) in enumerate(series_list):
            #break up size_label into BLK_H and BLK_W
            BLK_H, BLK_W = size_label.split('x')
            BLK_H = int(BLK_H)
            BLK_W = int(BLK_W)
            # Determine the column name
            if label == 'RO':
                col_name = graph + '.reorder'
            else:
                col_name = graph
            if col_name in df.columns:
                s = df[col_name]
            else:
                print(f"Column {col_name} not found in DataFrame for {label} {size_label}.")
                continue
            data_series = remove_trailing_zeros(s)
            data_series = data_series/(BLK_H*BLK_W)
            data.append(data_series)
            pos = group_pos + j
            positions.append(pos)
    return data, positions

# Prepare data for nnzPerBlock
data_nnzRatioPerBlock, positions_nnzPerBlock = prepare_data(series_list)

# Prepare data for nBlockPerRowWindow
# Update series_list with appropriate dataframes
series_list_nBlockPerRowWindow = [
    ('RO', '16x16', RO_nBlockPerRowWindow_dict.get('16x16'), 'lightcoral', 'RO 16x16'),
    ('OG', '16x16', OG_nBlockPerRowWindow_dict.get('16x16'), 'lightgray', 'OG 16x16'),
    ('RO', '8x32', RO_nBlockPerRowWindow_dict.get('8x32'), 'lightblue', 'RO 8x32'),
    ('OG', '8x32', OG_nBlockPerRowWindow_dict.get('8x32'), 'lightgreen', 'OG 8x32'),
    ('RO', '8x8', RO_nBlockPerRowWindow_dict.get('8x8'), 'gold', 'RO 8x8'),
    ('OG', '8x8', OG_nBlockPerRowWindow_dict.get('8x8'), 'violet', 'OG 8x8'),
]
# Remove any None entries due to missing data
series_list_nBlockPerRowWindow = [s for s in series_list_nBlockPerRowWindow if s[2] is not None]
data_nBlockPerRowWindow, positions_nBlockPerRowWindow = prepare_data(series_list_nBlockPerRowWindow)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(16, 14))

# Colors and legend labels
colors_per_group = [item[3] for item in series_list]
legend_labels = [item[4] for item in series_list]

n_bars_per_group = len(series_list)
xtick_positions = [i * (n_bars_per_group + 1) + (n_bars_per_group - 1)/2 for i in range(len(graphs))]

# Function to plot each boxplot
def plot_boxplot(ax, data, positions, title, ylabel, colors, legend_labels, log_y):
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        whis=1.5,         # Adjust whisker length
        showfliers=False  # Hide outliers
    )
    colors_full = colors * len(graphs)
    for patch, color in zip(box['boxes'], colors_full):
        patch.set_facecolor(color)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(graphs, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_xlabel('Graphs')
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.6)
    # Use logarithmic 2 scale
    if log_y:
        ax.set_yscale('log', base=2)
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=label) for color, label in zip(colors, legend_labels)]
    ax.legend(handles=legend_elements, loc='upper right')

# Plot nnzRatioPerBlock
plot_boxplot(axs[0], data_nnzRatioPerBlock, positions_nnzPerBlock, 'nnzRatioPerBlock', 'nnzRatioPerBlock', colors_per_group, legend_labels, log_y=False)

# Plot nBlockPerRowWindow
plot_boxplot(axs[1], data_nBlockPerRowWindow, positions_nBlockPerRowWindow, 'nBlockPerRowWindow', 'nBlockPerRowWindow', colors_per_group, legend_labels, log_y=True)

plt.tight_layout()
plt.savefig("graph_struct_box_plot.pdf")
plt.show()