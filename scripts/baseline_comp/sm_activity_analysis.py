import numpy as np
import matplotlib.pyplot as plt
import argparse

# Visualization parameters
VIZ_PARAMS = {
    'figure_size': (15, 6),
    'font_sizes': {
        'xlabel': 18,
        'ylabel': 18,
        'title': 18,
        'legend': 18,
        'ticks': 18
    },
    'title_padding': 15,
    'grid_alpha': 0.6,
    'mean_line_color': 'r',
    'mean_line_style': '--'
}

def read_and_normalize_data(filename):
    # Read the CSV file
    with open(filename, 'r') as f:
        # each line is 2 numbers, separated by a space
        # compute the difference between the two numbers
        data = [float(line.strip().split()[1]) - float(line.strip().split()[0]) for line in f if line.strip()]
    
    # Convert to numpy array and normalize by max
    data_array = np.array(data)
    max_time = np.max(data_array)
    normalized_data = data_array / max_time * 100  # Convert to percentage
    
    return normalized_data, max_time

def read_data(filename):
    # Read the CSV file
    with open(filename, 'r') as f:
        # each line is 2 numbers, separated by a space
        # compute the difference between the two numbers
        data = [float(line.strip().split()[1]) - float(line.strip().split()[0]) for line in f if line.strip()]
    
    # Convert to numpy array and normalize by max
    data_array = np.array(data)
    max_time = np.max(data_array)

    return data_array, max_time

def plot_sm_activities(scheduled_data, unscheduled_data, scheduled_max, unscheduled_max):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ_PARAMS['figure_size'], sharey=True)
    x = np.arange(len(scheduled_data))
    
    # Plot scheduled data
    ax1.bar(x, scheduled_data)
    ax1.set_xlabel('SM Index', fontsize=VIZ_PARAMS['font_sizes']['xlabel'])
    ax1.set_ylabel('SM Active Time (ms)', fontsize=VIZ_PARAMS['font_sizes']['ylabel'])
    ax1.set_title('Row Window Reordering', fontsize=VIZ_PARAMS['font_sizes']['title'], 
                 pad=VIZ_PARAMS['title_padding'])
    ax1.grid(True, axis='y', alpha=VIZ_PARAMS['grid_alpha'])
    mean_val = np.mean(scheduled_data)
    ax1.axhline(y=mean_val, 
                color=VIZ_PARAMS['mean_line_color'], 
                linestyle=VIZ_PARAMS['mean_line_style'], 
                label=f'Mean: {mean_val:.3f} ms')
    ax1.legend(fontsize=VIZ_PARAMS['font_sizes']['legend'],
              loc='center right')
    ax1.tick_params(axis='both', which='major', labelsize=VIZ_PARAMS['font_sizes']['ticks'])
    ax1.set_xlim(-0.9, len(scheduled_data) - 0.1)  # Reduce whitespace
    
    # Plot unscheduled data
    ax2.bar(x, unscheduled_data)
    ax2.set_xlabel('SM Index', fontsize=VIZ_PARAMS['font_sizes']['xlabel'])
    ax2.set_title('Default', fontsize=VIZ_PARAMS['font_sizes']['title'], 
                 pad=VIZ_PARAMS['title_padding'])
    ax2.grid(True, axis='y', alpha=VIZ_PARAMS['grid_alpha'])
    mean_val = np.mean(unscheduled_data)
    ax2.axhline(y=mean_val, 
                color=VIZ_PARAMS['mean_line_color'], 
                linestyle=VIZ_PARAMS['mean_line_style'], 
                label=f'Mean: {mean_val:.3f} ms')
    ax2.legend(fontsize=VIZ_PARAMS['font_sizes']['legend'],
              loc='center right')
    ax2.tick_params(axis='both', which='major', labelsize=VIZ_PARAMS['font_sizes']['ticks'])
    ax2.set_xlim(-0.9, len(unscheduled_data) - 0.1)  # Reduce whitespace
    
    # Set y-axis limits and ticks
    max_val = max(scheduled_max, unscheduled_max)
    ax1.set_ylim(0, max_val)
    
    # Create custom y-ticks with max value
    yticks = ax1.get_yticks()
    if max_val not in yticks:
        yticks = np.append(yticks[:-1], max_val)  # Replace last tick with max value
        ax1.set_yticks(yticks)
    
    # Format y-tick labels
    ax1.set_yticklabels([f'{y:.2f}' for y in yticks], fontsize=VIZ_PARAMS['font_sizes']['ticks'])
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

def main(args):
    # File paths
    scheduled_file = f'SM_active_time_results/SM_time_f3s1tb1rw_scheduled_{args.dataset}.csv'
    unscheduled_file = f'SM_active_time_results/SM_time_f3s1tb1rw_{args.dataset}.csv'
    
    # Process data
    scheduled_data, scheduled_max = read_data(scheduled_file)
    unscheduled_data, unscheduled_max = read_data(unscheduled_file)
    
    # Create plots
    plot_sm_activities(scheduled_data, unscheduled_data, scheduled_max, unscheduled_max)
    plt.savefig(f'sm_activity_comparison_{args.dataset}.png')
    
    # Print some statistics
    print(f"Scheduled Version:")
    print(f"Max SM Active Time: {scheduled_max:.4f} ms")
    print(f"Mean SM Active Time: {np.mean(scheduled_data):.4f} ms")
    print(f"Std Dev: {np.std(scheduled_data):.4f} ms\n")
    
    print(f"Unscheduled Version:")
    print(f"Max SM Active Time: {unscheduled_max:.4f} ms")
    print(f"Mean SM Active Time: {np.mean(unscheduled_data):.4f} ms")
    print(f"Std Dev: {np.std(unscheduled_data):.4f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SM Activity Analysis')
    parser.add_argument('--dataset', '-d', type=str, default='reddit', help='Dataset name')
    args = parser.parse_args()
    main(args) 