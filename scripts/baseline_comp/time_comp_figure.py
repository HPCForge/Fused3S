import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

datasets = ["citeseer", "cora", "pubmed", "Ell", "github", 
            "Artist", "com-amazon.ungraph", "Blog", 
            "amazon0505", "igb_small", "yelp", "reddit", 
            "igb_medium", "ogbn-products", "amazonProducts"]

# Original names for data reading
# algs_original = ['f3s_1tb1tcb', 'f3s_1tb1rw', 
#         'f3s_1tb1rw_scheduled', 'f3s_1tb1rw_scheduled_permuteV',
#         'flashSparse_naive_softmax', 'flashSparse_stable_softmax', 
#         'GTConvFuse_inference_tiling', 'GTConvFuse_inference_hyper', 
#         'propagate']
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
plt.figure(figsize=(24, 6))

# Width of each bar and positions of the bars
bar_width = 0.08
index = np.arange(len(datasets))

# Dictionary to store all data
all_data = {}
baseline_data = []

gpu_name = "A30"

results_path = "kernel_only_comp_results_2"
# Read the consolidated CSV file
file_name = f"{results_path}/baseline_comp_kernel_only_runtime_all_all_all_{gpu_name}.csv"
try:
    df = pd.read_csv(file_name)
    
    # For each dataset
    for dataset in datasets:
        print(f"--------------------------------")
        print(f"dataset: {dataset}")
        # Get the row for this dataset
        dataset_row = df[df['dataset'] == dataset].iloc[0] if not df[df['dataset'] == dataset].empty else None
        
        # Get baseline (f3s_1tb1rw_scheduled_permuteV) value
        baseline = dataset_row['f3s_1tb1rw_scheduled_permuteV'] if dataset_row is not None else np.nan
        baseline_data.append(baseline)
        
        # Store the speedup values for each algorithm
        for alg in algs_original:
            if alg not in all_data:
                all_data[alg] = []
            value = dataset_row[alg] if dataset_row is not None else np.nan
            print(f"alg: {alg} value: {value}")
            # Calculate relative runtime (value/baseline), handle special cases
            if pd.isna(value) or pd.isna(baseline) or baseline == 0:
                relative_runtime = np.nan
            else:
                relative_runtime = value/baseline
            all_data[alg].append(relative_runtime)
except FileNotFoundError:
    print(f"Error: Could not find file {file_name}")
    exit(1)

# Calculate geometric means for the requested comparisons
# 1. f3s_1tb1rw_scheduled_permuteV over df-gnn_tiling (GTConvFuse_inference_tiling)
f3s_vs_dfgnn = all_data['f3s_1tb1rw_scheduled_permuteV'].copy()
# Filter out NaN values for geometric mean calculation
f3s_vs_dfgnn_filtered = [x for x in f3s_vs_dfgnn if not np.isnan(x)]
geomean_f3s_vs_dfgnn = stats.gmean(f3s_vs_dfgnn_filtered) if f3s_vs_dfgnn_filtered else np.nan

# 2. f3s_1tb1rw_scheduled_permuteV over flashSparse_naive_softmax
# First calculate speedups of flashSparse_naive_softmax over baseline
flash_speedups = all_data['flashSparse_naive_softmax'].copy()
f3s_speedups = all_data['f3s_1tb1rw_scheduled_permuteV'].copy()

# Calculate relative speedup for each dataset
f3s_vs_flash = []
for i in range(len(datasets)):
    if not np.isnan(flash_speedups[i]) and not np.isnan(f3s_speedups[i]) and flash_speedups[i] != 0:
        # This gives f3s speedup relative to flashSparse
        relative_speedup = f3s_speedups[i] / flash_speedups[i]
        f3s_vs_flash.append(relative_speedup)

geomean_f3s_vs_flash = stats.gmean(f3s_vs_flash) if f3s_vs_flash else np.nan

print(f"Geometric mean of f3s_1tb1rw_scheduled_permuteV over df-gnn_tiling: {geomean_f3s_vs_dfgnn}")
print(f"Geometric mean of f3s_1tb1rw_scheduled_permuteV over flashSparse_naive_softmax: {geomean_f3s_vs_flash}")

# Calculate the requested geometric mean speedups

# 1. f3s_1tb1rw over f3s_1tb1tcb
f3s_1tb1rw_speedups = all_data['f3s_1tb1rw'].copy()
f3s_1tb1tcb_speedups = all_data['f3s_1tb1tcb'].copy()

f3s_1tb1rw_vs_f3s_1tb1tcb = []
for i in range(len(datasets)):
    if not np.isnan(f3s_1tb1rw_speedups[i]) and not np.isnan(f3s_1tb1tcb_speedups[i]) and f3s_1tb1tcb_speedups[i] != 0:
        # This gives f3s_1tb1rw speedup relative to f3s_1tb1tcb
        relative_speedup = f3s_1tb1rw_speedups[i] / f3s_1tb1tcb_speedups[i]
        f3s_1tb1rw_vs_f3s_1tb1tcb.append(relative_speedup)

geomean_f3s_1tb1rw_vs_f3s_1tb1tcb = stats.gmean(f3s_1tb1rw_vs_f3s_1tb1tcb) if f3s_1tb1rw_vs_f3s_1tb1tcb else np.nan

# 2. f3s_1tb1rw_scheduled over f3s_1tb1rw
f3s_1tb1rw_scheduled_speedups = all_data['f3s_1tb1rw_scheduled'].copy()

f3s_1tb1rw_scheduled_vs_f3s_1tb1rw = []
for i in range(len(datasets)):
    if not np.isnan(f3s_1tb1rw_scheduled_speedups[i]) and not np.isnan(f3s_1tb1rw_speedups[i]) and f3s_1tb1rw_speedups[i] != 0:
        # This gives f3s_1tb1rw_scheduled speedup relative to f3s_1tb1rw
        relative_speedup = f3s_1tb1rw_scheduled_speedups[i] / f3s_1tb1rw_speedups[i]
        f3s_1tb1rw_scheduled_vs_f3s_1tb1rw.append(relative_speedup)

geomean_f3s_1tb1rw_scheduled_vs_f3s_1tb1rw = stats.gmean(f3s_1tb1rw_scheduled_vs_f3s_1tb1rw) if f3s_1tb1rw_scheduled_vs_f3s_1tb1rw else np.nan

# 3. f3s_1tb1rw_scheduled_permuteV over f3s_1tb1rw_scheduled
f3s_1tb1rw_scheduled_permuteV_speedups = all_data['f3s_1tb1rw_scheduled_permuteV'].copy()

f3s_1tb1rw_scheduled_permuteV_vs_f3s_1tb1rw_scheduled = []
for i in range(len(datasets)):
    if not np.isnan(f3s_1tb1rw_scheduled_permuteV_speedups[i]) and not np.isnan(f3s_1tb1rw_scheduled_speedups[i]) and f3s_1tb1rw_scheduled_speedups[i] != 0:
        # This gives f3s_1tb1rw_scheduled_permuteV speedup relative to f3s_1tb1rw_scheduled
        relative_speedup = f3s_1tb1rw_scheduled_permuteV_speedups[i] / f3s_1tb1rw_scheduled_speedups[i]
        f3s_1tb1rw_scheduled_permuteV_vs_f3s_1tb1rw_scheduled.append(relative_speedup)

geomean_f3s_1tb1rw_scheduled_permuteV_vs_f3s_1tb1rw_scheduled = stats.gmean(f3s_1tb1rw_scheduled_permuteV_vs_f3s_1tb1rw_scheduled) if f3s_1tb1rw_scheduled_permuteV_vs_f3s_1tb1rw_scheduled else np.nan

# NEW: Calculate geometric mean for datasets where f3s_1tb1rw_scheduled is faster than f3s_1tb1rw
f3s_1tb1rw_scheduled_faster = []
faster_datasets = []
for i, dataset in enumerate(datasets):
    if not np.isnan(f3s_1tb1rw_scheduled_speedups[i]) and not np.isnan(f3s_1tb1rw_speedups[i]) and f3s_1tb1rw_speedups[i] != 0:
        # Check if scheduled is faster (relative_speedup > 1)
        relative_speedup = f3s_1tb1rw_scheduled_speedups[i] / f3s_1tb1rw_speedups[i]
        if relative_speedup > 1:
            f3s_1tb1rw_scheduled_faster.append(relative_speedup)
            faster_datasets.append(dataset)

geomean_f3s_1tb1rw_scheduled_faster = stats.gmean(f3s_1tb1rw_scheduled_faster) if f3s_1tb1rw_scheduled_faster else np.nan

# Print the requested geometric mean speedups
print(f"\nRequested Geometric Mean Speedups:")
print(f"1. f3s_1tb1rw over f3s_1tb1tcb: {geomean_f3s_1tb1rw_vs_f3s_1tb1tcb:.4f}x")
print(f"2. f3s_1tb1rw_scheduled over f3s_1tb1rw: {geomean_f3s_1tb1rw_scheduled_vs_f3s_1tb1rw:.4f}x")
print(f"3. f3s_1tb1rw_scheduled_permuteV over f3s_1tb1rw_scheduled: {geomean_f3s_1tb1rw_scheduled_permuteV_vs_f3s_1tb1rw_scheduled:.4f}x")

# Print the new geometric mean calculation
print(f"\nGeometric mean of f3s_1tb1rw_scheduled over f3s_1tb1rw (only for datasets where scheduled is faster): {geomean_f3s_1tb1rw_scheduled_faster:.4f}x")
print(f"Datasets where f3s_1tb1rw_scheduled is faster than f3s_1tb1rw: {faster_datasets}")

# Custom colors for each group
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
    else:  # propagate
        colors[alg] = grey

# Plot bars for each algorithm
handles = []
labels = []
other_handles = []
other_labels = []

for i, (orig_alg, display_alg) in enumerate(zip(algs_original, algs_display)):
    # For f3s_1tb1rw_scheduled_permuteV, always plot 1.0 since it's the baseline
    if orig_alg == 'f3s_1tb1rw_scheduled_permuteV':
        values = [1.0] * len(datasets)
    else:
        # Get the original values
        orig_values = all_data[orig_alg].copy()
        # Cap the values at 8.0 for display
        values = [min(v, 8.0) if not np.isnan(v) else np.nan for v in orig_values]
    
    bar = plt.bar(index + i * bar_width, 
            values,
            bar_width,
            label=display_alg,
            color=colors[orig_alg],
            alpha=0.8)
    
    # Collect handles and labels separately to reorder them
    if 'f3s' in display_alg:
        handles.append(bar)
        labels.append(display_alg)
    else:
        other_handles.append(bar)
        other_labels.append(display_alg)

# Combine handles and labels with F3S first, then others
handles.extend(other_handles)
labels.extend(other_labels)

# Add a horizontal line at y=1.0 to show baseline
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

# Set font size
font_size = 16
plt.rcParams.update({'font.size': font_size})

# Create display names for x-axis labels
display_datasets = datasets.copy()
display_datasets[datasets.index("com-amazon.ungraph")] = "com-amazon"
# Add line breaks to specific dataset names
display_datasets[datasets.index("amazonProducts")] = "Amazon\nProducts"
display_datasets[datasets.index("ogbn-products")] = "ogbn\nproducts"
display_datasets[datasets.index("igb_small")] = "IGB\nsmall"
display_datasets[datasets.index("igb_medium")] = "IGB\nmedium"

# Customize the plot
plt.ylabel('Relative runtime (lower is better)', fontsize=font_size)
# Make x-tick labels bold and centered with no rotation
plt.xticks(index + bar_width * len(algs_original)/2, display_datasets, rotation=0, ha='center', fontsize=font_size, fontweight='bold')
plt.yticks(fontsize=font_size)

# Create legend with two columns
if gpu_name == "GH200":
    plt.legend(handles, labels, loc='upper left', fontsize=font_size, ncol=3)

# Set x-axis limits to reduce padding
plt.xlim(-0.2, len(datasets) - 0.3)

# Set y-axis to log2 scale and customize ticks
plt.yscale('log', base=2)

# Set a fixed max for y-axis at 4.0
max_runtime = 8.0

# Generate tick positions for powers of 2 up to near the max
max_power = int(np.floor(np.log2(max_runtime)))
yticks = [2**i for i in range(-1, max_power + 1)]  # Start from 2^-1

# Create custom tick labels, replacing the lowest value with "0.5"
ytick_labels = [f'{tick:.2f}' for tick in yticks]
ytick_labels[0] = "0.5"  # Replace the first label (2^-1 = 0.5) with "0.5"

plt.yticks(yticks, ytick_labels)
plt.ylim(2**-1, max_runtime)

# Add text labels for values that exceed the cap
for i, (orig_alg, display_alg) in enumerate(zip(algs_original, algs_display)):
    if orig_alg != 'f3s_1tb1rw_scheduled_permuteV':  # Skip baseline
        for j, dataset in enumerate(datasets):
            if not np.isnan(all_data[orig_alg][j]) and all_data[orig_alg][j] > max_runtime:
                # Round to 1 decimal place for display
                value = round(all_data[orig_alg][j], 1)
                plt.text(j + i * bar_width, max_runtime * 0.95, 
                         f'{value}x', 
                         ha='center', va='top', rotation=90, 
                         fontsize=font_size-6, color='black',
                         fontweight='bold')

# Add grid for better readability with more emphasis on horizontal lines
plt.grid(True, which="major", ls="-", alpha=0.7)
plt.grid(True, which="minor", ls=":", alpha=0.4)
plt.gca().yaxis.grid(True, which='both')  # Ensure horizontal grid lines are shown
plt.gca().xaxis.grid(False)  # Turn off vertical grid lines

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig(f'{results_path}/speedup_comparison_{gpu_name}.png', dpi=300, bbox_inches='tight')
plt.close()
