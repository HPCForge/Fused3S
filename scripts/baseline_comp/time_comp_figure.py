import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datasets = ["reddit", "amazonProducts", "yelp", "amazon0505", 
            "Artist", "Blog", "com-amazon.ungraph", "github", 
            "Ell", "ogbn-products", "citeseer", "pubmed", "cora",
            "igb_small", "igb_medium"]

# Original names for data reading
algs_original = ['f3s_1tb1tcb', 'f3s_1tb1rw', 
        'f3s_1tb1rw_scheduled', 'f3s_1tb1rw_scheduled_permuteV',
        'flashSparse_naive_softmax', 'flashSparse_stable_softmax', 
        'GTConvFuse_inference_tiling', 'GTConvFuse_inference_hyper', 
        'propagate']

# Display names for the legend
algs_display = ['f3s_1tb1tcb', 'f3s_1tb1rw', 
        'f3s_1tb1rw_scheduled', 'f3s_1tb1rw_scheduled_permuteV',
        'flashSparse_naive_softmax', 'flashSparse_stable_softmax', 
        'df-gnn_tiling', 'df-gnn_hyper', 
        'Pyg']

# Create figure and axis with larger size
plt.figure(figsize=(24, 6))

# Width of each bar and positions of the bars
bar_width = 0.08
index = np.arange(len(datasets))

# Dictionary to store all data
all_data = {}
baseline_data = []

gpu_name = "GH200"

# Read the consolidated CSV file
file_name = f"csv/baseline_comp_kernel_only_all_all_all_{gpu_name}.csv"
try:
    df = pd.read_csv(file_name)
    
    # For each dataset
    for dataset in datasets:
        print(f"--------------------------------")
        print(f"dataset: {dataset}")
        # Get the row for this dataset
        dataset_row = df[df['dataset'] == dataset].iloc[0] if not df[df['dataset'] == dataset].empty else None
        
        # Get baseline (GTConvFuse_inference_tiling) value
        baseline = dataset_row['GTConvFuse_inference_tiling'] if dataset_row is not None else np.nan
        baseline_data.append(baseline)
        
        # Store the speedup values for each algorithm
        for alg in algs_original:
            if alg not in all_data:
                all_data[alg] = []
            value = dataset_row[alg] if dataset_row is not None else np.nan
            print(f"alg: {alg} value: {value}")
            # Calculate speedup (baseline/value), handle special cases
            if pd.isna(value) or pd.isna(baseline) or baseline == 0:
                speedup = np.nan
            else:
                speedup = baseline/value
            all_data[alg].append(speedup)
except FileNotFoundError:
    print(f"Error: Could not find file {file_name}")
    exit(1)

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
    elif 'GTConvFuse' in alg:
        colors[alg] = dfgnn_greens[len(colors) - 6]  # Offset by number of f3s + flashSparse algorithms
    else:  # propagate
        colors[alg] = grey

# Plot bars for each algorithm
handles = []
labels = []
other_handles = []
other_labels = []

for i, (orig_alg, display_alg) in enumerate(zip(algs_original, algs_display)):
    # For GTConvFuse_inference_tiling (now df-gnn_tiling), always plot 1.0 since it's the baseline
    if orig_alg == 'GTConvFuse_inference_tiling':
        values = [1.0] * len(datasets)
    else:
        values = all_data[orig_alg]
    
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

# Customize the plot
plt.ylabel('Speedup over df-gnn_tiling\n(higher is better)', fontsize=font_size)
plt.xticks(index + bar_width * len(algs_original)/2, display_datasets, rotation=30, ha='right', fontsize=font_size)
plt.yticks(fontsize=font_size)

# Create legend with two columns
if gpu_name == "A30":
    plt.legend(handles, labels, loc='lower left', fontsize=font_size, ncol=3)

# Set x-axis limits to reduce padding
plt.xlim(-0.2, len(datasets) - 0.3)

# Set y-axis to log2 scale and customize ticks
plt.yscale('log', base=2)

# Find the maximum speedup value
max_speedup = max(max(values) for values in all_data.values() if len(values) > 0)

# Generate tick positions for powers of 2 up to near the max
max_power = int(np.floor(np.log2(max_speedup)))
yticks = [2**i for i in range(-3, max_power + 1)]
plt.yticks(yticks, [f'{tick:.2f}x' for tick in yticks])
plt.ylim(2**-3, max_speedup)

# Add grid for better readability with more emphasis on horizontal lines
plt.grid(True, which="major", ls="-", alpha=0.7)
plt.grid(True, which="minor", ls=":", alpha=0.4)
plt.gca().yaxis.grid(True, which='both')  # Ensure horizontal grid lines are shown
plt.gca().xaxis.grid(False)  # Turn off vertical grid lines

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig(f'speedup_comparison_{gpu_name}.png', dpi=300, bbox_inches='tight')
plt.close()
