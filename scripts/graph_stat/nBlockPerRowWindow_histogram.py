import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('OG_16x8_nBlockPerRowWindow_pd.csv')
df = df.iloc[:, -4:]  # Remove first column (index)

# Calculate number of rows and columns for subplots
n_graphs = len(df.columns)
n_rows = int(np.ceil(n_graphs / 2))
n_cols = 2

# Create figure
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
axes = axes.flatten()  # Flatten axes array for easier indexing

# For each graph (column)
for idx, (col_name, data) in enumerate(df.items()):
    # Remove trailing zeros and zeros
    data = data[data != 0]
    
    if len(data) > 0:
        # Calculate 90th percentile
        p90 = np.percentile(data, 90)
        
        # Create modified data where values above 90th percentile are set to p90
        plot_data = np.copy(data)
        n_outliers = np.sum(plot_data > p90)
        plot_data[plot_data > p90] = p90

        # Create histogram
        counts, bins, patches = axes[idx].hist(plot_data, bins='auto')
        axes[idx].set_xticks(bins)
        axes[idx].set_xticklabels([f'{x:.1f}' for x in bins], rotation=60)

        # Modify the last bin to show it contains outliers
        if n_outliers > 0:
            patches[-1].set_facecolor('red')  # Make the last bin red
            axes[idx].text(p90, counts[-1], f'+{n_outliers}\noutliers', 
                        ha='left', va='bottom')
        
        # Add text showing the range of outliers
        if n_outliers > 0:
            max_val = data.max()
        
        axes[idx].set_title(f'{col_name}\n(total number of row windows: {len(data)})')
    else:
        axes[idx].text(0.5, 0.5, 'No non-zero data', 
                      horizontalalignment='center',
                      verticalalignment='center')
        axes[idx].set_title(col_name)

# Remove empty subplots
for idx in range(n_graphs, len(axes)):
    fig.delaxes(axes[idx])

# Add figure caption at the bottom
fig.text(0.5, 0.05, 
         'Distribution of blocks per row window for different graphs.\n' +
         'X-axis: Number of blocks in each row window\n' +
         'Y-axis: Frequency count of row windows' +
         'Note the outliers bin account for 10% of the total number of row windows.\n',
         ha='center', va='center', fontsize=14)

# Adjust layout to make room for caption

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('OG_16x8_nBlockPerRowWindow_histogram.png')
plt.close() 