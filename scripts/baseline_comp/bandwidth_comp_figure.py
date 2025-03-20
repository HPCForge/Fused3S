import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Datasets ordered by size
datasets = ["citeseer", "cora", "pubmed", "Ell", "github", 
            "Artist", "com-amazon.ungraph", "Blog", 
            "amazon0505", "igb_small", "yelp", "reddit", 
            "igb_medium", "ogbn-products", "amazonProducts"]

# Original names for data reading
algs_original = ['f3s_1tb1tcb', 'f3s_1tb1rw', 
        'f3s_1tb1rw_scheduled', 'f3s_1tb1rw_scheduled_permuteV',
        'flashSparse_naive_softmax', 'flashSparse_stable_softmax', 
        'dfgnn_tiling', 'dfgnn_hyper', 
        'pyg_gtconv']

# Display names for the legend
algs_display = ['F3S_splitR', 'F3S_splitC', 
        'F3S_reorderRW', 'F3S_permuteQKV',
        'FlashSparse_naive_softmax', 'FlashSparse_stable_softmax', 
        'DF-GNN_tiling', 'DF-GNN_hyper', 
        'PyG']

# Create figure and axis with larger size
plt.figure(figsize=(24, 10))

# Dictionary to store all data
all_data = {}

gpu_name = "A30"
results_path = "kernel_only_comp_results_2"

# Read the bandwidth CSV file
file_name = f"{results_path}/baseline_comp_kernel_only_bandwidth_all_all_all_{gpu_name}.csv"
try:
    df = pd.read_csv(file_name)
    
    # For each dataset
    for dataset in datasets:
        print(f"--------------------------------")
        print(f"dataset: {dataset}")
        # Get the row for this dataset
        dataset_row = df[df['dataset'] == dataset].iloc[0] if not df[df['dataset'] == dataset].empty else None
        
        # Store the bandwidth values for each algorithm
        for alg in algs_original:
            if alg not in all_data:
                all_data[alg] = []
            value = dataset_row[alg] if dataset_row is not None and alg in dataset_row else np.nan
            print(f"alg: {alg} value: {value}")
            # Convert to GB/s for better readability
            if not np.isnan(value):
                value = value / 1e9
            # Store bandwidth values in GB/s
            all_data[alg].append(value)
except FileNotFoundError:
    print(f"Error: Could not find file {file_name}")
    exit(1)

# Custom colors for each group - keep the same color scheme
f3s_blues = plt.cm.Blues([0.4, 0.6, 0.8, 1])  # Increasingly dark blues
flash_reds = plt.cm.Reds([0.5, 0.8])  # Two shades of red
dfgnn_greens = plt.cm.Greens([0.5, 0.8])  # Two shades of green
grey = '#808080'  # Grey for Pyg

# Create color map
colors = {}
for i, alg in enumerate(algs_original):
    if 'f3s' in alg:
        colors[alg] = f3s_blues[len(colors)]
    elif 'flashSparse' in alg:
        colors[alg] = flash_reds[len(colors) - 4]  # Offset by number of f3s algorithms
    elif 'dfgnn' in alg:
        colors[alg] = dfgnn_greens[len(colors) - 6]  # Offset by number of f3s + flashSparse algorithms
    else:  # PyG
        colors[alg] = grey

# Create display names for x-axis labels
display_datasets = datasets.copy()
display_datasets[datasets.index("com-amazon.ungraph")] = "com-amazon"
# Add line breaks to specific dataset names
display_datasets[datasets.index("amazonProducts")] = "Amazon\nProducts"
display_datasets[datasets.index("ogbn-products")] = "ogbn\nproducts"
display_datasets[datasets.index("igb_small")] = "IGB\nsmall"
display_datasets[datasets.index("igb_medium")] = "IGB\nmedium"

# Set font size
font_size = 16
plt.rcParams.update({'font.size': font_size})

# This will be the x-axis
x = np.arange(len(datasets))

# Plot line for each algorithm
handles = []
labels = []
f3s_handles = []
f3s_labels = []
other_handles = []
other_labels = []

for i, (orig_alg, display_alg) in enumerate(zip(algs_original, algs_display)):
    values = all_data[orig_alg]
    
    # Plot lines with markers
    line, = plt.plot(x, values, 
                   marker='o',
                   markersize=8,
                   linewidth=3,
                   label=display_alg,
                   color=colors[orig_alg],
                   alpha=0.9)
    
    # Collect handles and labels separately to reorder them
    if 'f3s' in display_alg:
        f3s_handles.append(line)
        f3s_labels.append(display_alg)
    else:
        other_handles.append(line)
        other_labels.append(display_alg)

# Combine handles and labels with F3S first, then others
handles = f3s_handles + other_handles
labels = f3s_labels + other_labels

# Customize the plot
plt.ylabel('Memory Bandwidth (GB/s)', fontsize=font_size)
plt.xlabel('Dataset', fontsize=font_size)
plt.xticks(x, display_datasets, rotation=0, ha='center', fontsize=font_size, fontweight='bold')
plt.yticks(fontsize=font_size)

# Create legend
plt.legend(handles, labels, loc='upper left', fontsize=font_size, ncol=2)

# Add grid for better readability
plt.grid(True, which="major", ls="-", alpha=0.7)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Add data labels for the highest bandwidth algorithm at each dataset
for i, dataset in enumerate(datasets):
    # Find the maximum bandwidth for this dataset
    max_value = -float('inf')
    max_alg = None
    for alg in algs_original:
        if i < len(all_data[alg]) and not np.isnan(all_data[alg][i]) and all_data[alg][i] > max_value:
            max_value = all_data[alg][i]
            max_alg = alg
    
    if max_alg:
        # Add annotation above the highest point
        plt.annotate(f'{max_value:.1f} GB/s',
                    xy=(i, max_value),
                    xytext=(0, 10),  # 10 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

# Set title for the plot
plt.title(f'Memory Bandwidth Comparison on {gpu_name} GPU', fontsize=font_size + 4)

# Save the figure
plt.savefig(f'{results_path}/bandwidth_comparison_{gpu_name}.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary stats
print("\nBandwidth Comparisons - Geometric Means:")

# Calculate geometric means for each algorithm (across all datasets)
geomeans = {}
for alg in algs_original:
    values = [v for v in all_data[alg] if not np.isnan(v) and v > 0]
    if values:
        geomeans[alg] = stats.gmean(values)
    else:
        geomeans[alg] = np.nan

# Print the geometric means in descending order
sorted_algs = sorted([(alg, geomeans[alg]) for alg in geomeans], key=lambda x: x[1], reverse=True)
for alg, geomean in sorted_algs:
    display_name = algs_display[algs_original.index(alg)]
    print(f"{display_name}: {geomean:.2f} GB/s")

# Calculate and print the relative performance compared to the baseline
baseline_alg = 'dfgnn_tiling'
baseline_geomean = geomeans.get(baseline_alg, np.nan)

if not np.isnan(baseline_geomean) and baseline_geomean > 0:
    print(f"\nRelative Performance Compared to {algs_display[algs_original.index(baseline_alg)]}:")
    for alg in algs_original:
        if alg != baseline_alg and not np.isnan(geomeans[alg]):
            relative = geomeans[alg] / baseline_geomean
            display_name = algs_display[algs_original.index(alg)]
            print(f"{display_name}: {relative:.2f}x") 