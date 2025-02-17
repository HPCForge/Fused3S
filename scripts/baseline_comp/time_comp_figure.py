import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datasets = ["reddit", "amazonProducts", "yelp", "amazon0505", 
            "Artist", "Blog", "com-amazon.ungraph", "github", 
            "Ell", "ogbn-products", "citeseer", "pubmed", "cora",
            "igb_small", "igb_medium"]

algs = ['f3s_1tb1tcb', 'f3s_1tb1rw', 
        'f3s_1tb1rw_scheduled', 'f3s_1tb1rw_scheduled_permuteV',
        'flashSparse_naive_softmax', 'flashSparse_stable_softmax', 
        'GTConvFuse_inference_tiling', 'GTConvFuse_inference_hyper', 
        'propagate']

# Create figure and axis with larger size
plt.figure(figsize=(20, 10))

# Width of each bar and positions of the bars
bar_width = 0.08
index = np.arange(len(datasets))

# Dictionary to store all data
all_data = {}
baseline_data = []

# Read the consolidated CSV file
file_name = "csv/baseline_comp_kernel_only_all_all_all.csv"
try:
    df = pd.read_csv(file_name)
    
    # For each dataset
    for dataset in datasets:
        # Get the row for this dataset
        dataset_row = df[df['dataset'] == dataset].iloc[0] if not df[df['dataset'] == dataset].empty else None
        
        # Get baseline (GTConvFuse_inference_tiling) value
        baseline = dataset_row['GTConvFuse_inference_tiling'] if dataset_row is not None else np.nan
        baseline_data.append(baseline)
        
        # Store the speedup values for each algorithm
        for alg in algs:
            if alg not in all_data:
                all_data[alg] = []
            value = dataset_row[alg] if dataset_row is not None else np.nan
            # Calculate speedup (baseline/value), handle special cases
            if pd.isna(value) or pd.isna(baseline) or baseline == 0:
                speedup = np.nan
            else:
                speedup = baseline/value
            all_data[alg].append(speedup)
except FileNotFoundError:
    print(f"Error: Could not find file {file_name}")
    exit(1)

# Plot bars for each algorithm
colors = plt.cm.tab20(np.linspace(0, 1, len(algs)))  # Generate distinct colors
for i, (alg, color) in enumerate(zip(algs, colors)):
    # For GTConvFuse_inference_tiling, always plot 1.0 since it's the baseline
    if alg == 'GTConvFuse_inference_tiling':
        values = [1.0] * len(datasets)
    else:
        values = all_data[alg]
    
    plt.bar(index + i * bar_width, 
            values,
            bar_width,
            label=alg,
            color=color,
            alpha=0.8)

# Add a horizontal line at y=1.0 to show baseline
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

# Customize the plot
plt.xlabel('Datasets')
plt.ylabel('Speedup over GTConvFuse_inference_tiling\n(higher is better)')
plt.title('Performance Speedup Comparison Across Different Datasets and Algorithms')
plt.xticks(index + bar_width * len(algs)/2, datasets, rotation=45, ha='right')
plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))

# Set y-axis to log2 scale and customize ticks
plt.yscale('log', base=2)
# Generate tick positions for powers of 2 from 2^-3 to 2^3
yticks = [2**i for i in range(-3, 4)]
plt.yticks(yticks, [f'{tick:.2f}x' if tick != 1 else '1.00x (baseline)' for tick in yticks])

# Add grid for better readability with more emphasis on horizontal lines
plt.grid(True, which="major", ls="-", alpha=0.7)
plt.grid(True, which="minor", ls=":", alpha=0.4)
plt.gca().yaxis.grid(True, which='both')  # Ensure horizontal grid lines are shown
plt.gca().xaxis.grid(False)  # Turn off vertical grid lines

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('speedup_comparison.png', dpi=300, bbox_inches='tight')
plt.close()


