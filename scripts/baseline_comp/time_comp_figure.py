import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import argparse

datasets_raw_display_name_dict = {
    "citeseer": "Citeseer",
    "cora": "Cora",
    "pubmed": "PubMed",
    "Ell": "Ell",
    "github": "GitHub",
    "Artist": "Artist",
    "com-amazon.ungraph": "Amazon",
    "Blog": "Blog",
    "amazon0505": "Amazon0505",
    "igb_small": "IGB \nsmall",
    "igb_medium": "IGB \nmedium",
    "yelp": "Yelp",
    "reddit": "Reddit",
    "ogbn-products": "OGB \nproducts",
    "amazonProducts": "Amazon \nproducts",
    "ogbg-molhiv": "OGB \nmolhiv",
    "ogbg-ppa": "OGB \nppa",
    "ogbg-molpcba": "OGB \nmolpcba",
    "ogbg-code2": "OGB \ncode2",
    "ZINC": "ZINC",
    "PascalVOC-SP": "PascalVOC-SP",
    "COCO-SP": "COCO-SP",
    "Peptides-func": "Peptides-func",
    "Peptides-struct": "Peptides-struct",
}

algs_original_display_name_dict = {
    'f3s_1tb1tcb': 'F3S_splitC',
    'f3s_1tb1rw': 'F3S_splitR',
    'f3s_1tb1rw_scheduled': 'F3S_reorderRW',
    'f3s_1tb1rw_scheduled_permuteV': 'F3S_permuteQKV',
    'flashSparse_naive_softmax': 'FlashSparse_naive_softmax',
    'flashSparse_stable_softmax': 'FlashSparse_stable_softmax',
    'dfgnn_tiling': 'DF-GNN_tiling',
    'dfgnn_hyper': 'DF-GNN_hyper',
    'pyg_gtconv': 'PyG',
}

algs_colors = {
    'f3s_1tb1tcb': plt.cm.Blues(0.3),
    'f3s_1tb1rw': plt.cm.Blues(0.5),
    'f3s_1tb1rw_scheduled': plt.cm.Blues(0.7),
    'f3s_1tb1rw_scheduled_permuteV': plt.cm.Blues(0.9),
    'flashSparse_naive_softmax': plt.cm.Reds(0.5),
    'flashSparse_stable_softmax': plt.cm.Reds(0.8),
    'dfgnn_tiling': plt.cm.Greens(0.5),
    'dfgnn_hyper': plt.cm.Greens(0.8),
    'pyg_gtconv': '#808080',
}

def report_speedup(all_data, datasets):
    # Calculate geometric mean speedups
    # 1. f3s_1tb1rw_scheduled_permuteV over df-gnn_tiling
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

    f3s_1tb1rw_speedups = all_data['f3s_1tb1rw'].copy()
    f3s_1tb1tcb_speedups = all_data['f3s_1tb1tcb'].copy()
    f3s_1tb1rw_scheduled_speedups = all_data['f3s_1tb1rw_scheduled'].copy()
    f3s_1tb1rw_scheduled_permuteV_speedups = all_data['f3s_1tb1rw_scheduled_permuteV'].copy()

    f3s_1tb1rw_vs_f3s_1tb1tcb = []
    f3s_1tb1rw_scheduled_vs_f3s_1tb1rw = []
    f3s_1tb1rw_scheduled_permuteV_vs_f3s_1tb1rw_scheduled = []
    f3s_1tb1rw_scheduled_faster = []
    faster_datasets = []
    for i, dataset in enumerate(datasets):
        if not np.isnan(f3s_1tb1rw_speedups[i]) and not np.isnan(f3s_1tb1tcb_speedups[i]) and f3s_1tb1tcb_speedups[i] != 0:
            # This gives f3s_1tb1rw speedup relative to f3s_1tb1tcb
            relative_speedup = f3s_1tb1rw_speedups[i] / f3s_1tb1tcb_speedups[i]
            f3s_1tb1rw_vs_f3s_1tb1tcb.append(relative_speedup)
        if not np.isnan(f3s_1tb1rw_scheduled_speedups[i]) and not np.isnan(f3s_1tb1rw_speedups[i]) and f3s_1tb1rw_speedups[i] != 0:
            # This gives f3s_1tb1rw_scheduled speedup relative to f3s_1tb1rw
            relative_speedup = f3s_1tb1rw_scheduled_speedups[i] / f3s_1tb1rw_speedups[i]
            if relative_speedup > 1:
                f3s_1tb1rw_scheduled_faster.append(relative_speedup)
                faster_datasets.append(dataset)
            f3s_1tb1rw_scheduled_vs_f3s_1tb1rw.append(relative_speedup)
        if not np.isnan(f3s_1tb1rw_scheduled_permuteV_speedups[i]) and not np.isnan(f3s_1tb1rw_scheduled_speedups[i]) and f3s_1tb1rw_scheduled_speedups[i] != 0:
            relative_speedup = f3s_1tb1rw_scheduled_permuteV_speedups[i] / f3s_1tb1rw_scheduled_speedups[i]
            f3s_1tb1rw_scheduled_permuteV_vs_f3s_1tb1rw_scheduled.append(relative_speedup)

    geomean_f3s_1tb1rw_vs_f3s_1tb1tcb = stats.gmean(f3s_1tb1rw_vs_f3s_1tb1tcb) if f3s_1tb1rw_vs_f3s_1tb1tcb else np.nan
    geomean_f3s_1tb1rw_scheduled_vs_f3s_1tb1rw = stats.gmean(f3s_1tb1rw_scheduled_vs_f3s_1tb1rw) if f3s_1tb1rw_scheduled_vs_f3s_1tb1rw else np.nan
    geomean_f3s_1tb1rw_scheduled_permuteV_vs_f3s_1tb1rw_scheduled = stats.gmean(f3s_1tb1rw_scheduled_permuteV_vs_f3s_1tb1rw_scheduled) if f3s_1tb1rw_scheduled_permuteV_vs_f3s_1tb1rw_scheduled else np.nan
    # Calculate geometric mean for datasets where f3s_1tb1rw_scheduled is faster than f3s_1tb1rw
    geomean_f3s_1tb1rw_scheduled_faster = stats.gmean(f3s_1tb1rw_scheduled_faster) if f3s_1tb1rw_scheduled_faster else np.nan

    # Print the requested geometric mean speedups
    print(f"Geometric Mean Speedups:")
    print(f"1. f3s_1tb1rw over f3s_1tb1tcb: {geomean_f3s_1tb1rw_vs_f3s_1tb1tcb:.4f}x")
    print(f"2. f3s_1tb1rw_scheduled over f3s_1tb1rw: {geomean_f3s_1tb1rw_scheduled_vs_f3s_1tb1rw:.4f}x")
    print(f"3. f3s_1tb1rw_scheduled_permuteV over f3s_1tb1rw_scheduled: {geomean_f3s_1tb1rw_scheduled_permuteV_vs_f3s_1tb1rw_scheduled:.4f}x")
    print(f"4. Geometric mean of f3s_1tb1rw_scheduled over f3s_1tb1rw (only for datasets where scheduled is faster): {geomean_f3s_1tb1rw_scheduled_faster:.4f}x")
    print(f"Datasets where f3s_1tb1rw_scheduled is faster than f3s_1tb1rw: {faster_datasets}")

# def choose_algs(algs_set):
#     if algs_set == "all":
#         baseline_alg = 'dfgnn_tiling'
#     elif algs_set == "internal":
#         baseline_alg = 'f3s_1tb1tcb'
#         algs_original_display_name_dict.pop('dfgnn_tiling')
#         algs_original_display_name_dict.pop('dfgnn_hyper')
#         algs_original_display_name_dict.pop('pyg_gtconv')
#         algs_original_display_name_dict.pop('flashSparse_naive_softmax')
#         algs_original_display_name_dict.pop('flashSparse_stable_softmax')
#     elif algs_set == "external":
#         baseline_alg = 'dfgnn_tiling'
#         algs_original_display_name_dict.pop('f3s_1tb1rw')
#         algs_original_display_name_dict.pop('f3s_1tb1tcb')
#         algs_original_display_name_dict.pop('f3s_1tb1rw_scheduled')
#     baseline_pair = (baseline_alg, algs_original_display_name_dict.pop(baseline_alg))
#     return algs_original_display_name_dict, baseline_pair

def choose_datasets(args):
    if args.batched:
        datasets = ["ZINC", "PascalVOC-SP", "COCO-SP", 
                    "Peptides-func", "Peptides-struct",
                    "ogbg-molhiv", "ogbg-ppa", "ogbg-molpcba", "ogbg-code2"]
    else:
        datasets = ["citeseer", "cora", "pubmed", "Ell", "github", 
                    "Artist", "com-amazon.ungraph", "Blog", 
                    "amazon0505", "igb_small", "yelp", "reddit", 
                    "igb_medium", "ogbn-products", "amazonProducts"]
    return datasets


def main(args):
    datasets = choose_datasets(args)
    # Set font size
    font_size = 16
    plt.rcParams.update({'font.size': font_size})
    # Width of each bar and positions of the bars
    bar_width = 0.08
    dataset_offset = np.arange(len(datasets))
    
    # Read the data
    file_name = f"{args.data_path}/baseline_comp_kernel_only_runtime_all_all_all_{args.gpu_name}.csv"
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: Could not find file {file_name}")
        exit(1)
    
    # Generate series of plots
    if args.series_type == "external":
        # External comparison - use dfgnn_tiling as baseline
        algorithm_groups = [
            ['f3s_1tb1rw_scheduled_permuteV'],  # F3S alone
            ['f3s_1tb1rw_scheduled_permuteV', 'dfgnn_hyper'],  # F3S + DF-GNN
            ['f3s_1tb1rw_scheduled_permuteV', 'dfgnn_hyper', 
             'flashSparse_naive_softmax'],  # F3S + DF-GNN + FlashSparse (naive)
            ['f3s_1tb1rw_scheduled_permuteV', 'dfgnn_hyper', 
             'flashSparse_naive_softmax', 'flashSparse_stable_softmax'],  # F3S + DF-GNN + FlashSparse (naive + stable)
            ['f3s_1tb1rw_scheduled_permuteV', 'dfgnn_hyper', 
             'flashSparse_naive_softmax', 'flashSparse_stable_softmax',
             'pyg_gtconv'],  # All algorithms
        ]
        series_names = ["f3s_only", "f3s_dfgnn", "f3s_dfgnn_flash_naive", "f3s_dfgnn_flash_naive_stable", "all"]
        baseline_alg = 'dfgnn_tiling'
    elif args.series_type == "internal":
        # Internal comparison - use f3s_1tb1tcb as baseline
        algorithm_groups = [
            ['f3s_1tb1tcb', 'f3s_1tb1rw'],  # + Row-wise
            ['f3s_1tb1tcb', 'f3s_1tb1rw', 'f3s_1tb1rw_scheduled'],  # + Scheduled
            ['f3s_1tb1tcb', 'f3s_1tb1rw', 'f3s_1tb1rw_scheduled', 'f3s_1tb1rw_scheduled_permuteV'],  # + PermuteV
        ]
        series_names = ["row_wise", "scheduled", "permuted"]
        baseline_alg = 'f3s_1tb1tcb'
    elif args.series_type == "all":
        algorithm_groups = [
            ['f3s_1tb1tcb', 'f3s_1tb1rw', 'f3s_1tb1rw_scheduled', 'f3s_1tb1rw_scheduled_permuteV',
             'flashSparse_naive_softmax', 'flashSparse_stable_softmax',
             'dfgnn_hyper', 'dfgnn_tiling', 'pyg_gtconv'],  # All algorithms
        ]
        series_names = ["all"]
        baseline_alg = 'dfgnn_tiling'
    else:
        raise ValueError(f"Unknown series type: {args.series_type}")
    
    baseline_display = algs_original_display_name_dict[baseline_alg]
    baseline_pair = (baseline_alg, baseline_display)
    
    for group_idx, group in enumerate(algorithm_groups):
        # Skip baseline in the algorithm list if it's included
        current_group = [alg for alg in group if alg != baseline_alg]
        generate_plot(df, current_group, baseline_pair, datasets, dataset_offset, bar_width, font_size, 
                     series_names[group_idx], args)

def generate_plot(df, alg_list, baseline_pair, datasets, dataset_offset, bar_width, font_size, series_name, args):
    # Create a copy of the original display name dict to work with
    current_algs_dict = {alg: algs_original_display_name_dict[alg] for alg in alg_list if alg in algs_original_display_name_dict}
    
    # Generate the plot with the specified baseline
    generate_plot_with_algs(df, current_algs_dict, baseline_pair, datasets, 
                           dataset_offset, bar_width, font_size, args, series_name)

def generate_plot_with_algs(df, algs_dict, baseline_pair, datasets, dataset_offset, 
                          bar_width, font_size, args, series_name=None):
    # Create figure and axis with larger size
    plt.figure(figsize=(24, 6))
    
    # Dictionary to store all data
    all_data = {}
    baseline_data = []
    
    for dataset in datasets:
        print(f"--------------------------------")
        print(f"dataset: {dataset}")
        # Get the row for this dataset
        dataset_row = df[df['dataset'] == dataset].iloc[0] if not df[df['dataset'] == dataset].empty else None
        
        # Get baseline value
        baseline = dataset_row[baseline_pair[0]] if dataset_row is not None else np.nan
        baseline_data.append(baseline)
        if pd.isna(baseline):
            print(f"Warning: Baseline value for {dataset} is NaN")
            continue
        if baseline == 0:
            print(f"Warning: Baseline value for {dataset} is 0")
            continue

        # Store the speedup values for each algorithm
        for alg, display_alg in algs_dict.items():
            if alg not in all_data:
                all_data[alg] = []
            value = dataset_row[alg] if dataset_row is not None else np.nan
            # Calculate speedup (baseline/value), handle special cases
            if pd.isna(value):
                speedup = np.nan
            else:
                speedup = baseline/value
            all_data[alg].append(speedup)
    
    if args.series_type == "all":
        report_speedup(all_data, datasets)

    # Plot bars for each algorithm
    handles = []
    labels = []
    other_handles = []
    other_labels = []

    for i, (orig_alg, display_alg) in enumerate(algs_dict.items()):
        values = all_data[orig_alg]
        print(f"orig_alg: {orig_alg}, display_alg: {display_alg}, color: {algs_colors[orig_alg]}, values: {values}, i: {i}")
        
        bar = plt.bar(dataset_offset + i * bar_width, 
                    values,
                    bar_width,
                    label=display_alg,
                    color=algs_colors[orig_alg])
        
        # Add "OOM" text for NaN values
        for j, value in enumerate(values):
            if np.isnan(value):
                plt.text(dataset_offset[j] + i * bar_width, 
                         0.1,  # Position near bottom of graph
                         "OOM", 
                         ha='center',
                         va='bottom',
                         rotation=90,
                         fontsize=font_size-2,
                         color=algs_colors[orig_alg])
        
        # Collect handles and labels separately to reorder them
        if 'f3s' in display_alg.lower():
            handles.append(bar)
            labels.append(display_alg)
        else:
            other_handles.append(bar)
            other_labels.append(display_alg)

    # Combine handles and labels with F3S first, then others
    handles.extend(other_handles)
    labels.extend(other_labels)

    # Add a horizontal line at y=1.0 to show baseline
    baseline_line = plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label=f'{baseline_pair[1]} (baseline)')
    handles.insert(0, baseline_line)
    labels.insert(0, f'{baseline_pair[1]} (baseline)')

    # Create display names for x-axis labels
    display_datasets = datasets.copy()
    for i, dataset in enumerate(datasets):
        display_datasets[datasets.index(dataset)] = datasets_raw_display_name_dict[dataset]

    # Customize the plot
    plt.ylabel(f'Speedup over {baseline_pair[1]}\n', fontsize=font_size)
    plt.xticks(dataset_offset + bar_width * len(algs_dict)/2, display_datasets, rotation=0, ha='center', fontsize=font_size, fontweight='bold')
    plt.yticks(fontsize=font_size)

    # Create legend with two columns
    if not args.no_legend:
        plt.legend(handles, labels, loc='upper left', 
                   bbox_to_anchor=(0, 1.4),
                   fontsize=font_size, 
                   ncol=3)

    # Set x-axis limits to reduce padding
    plt.xlim(-0.2, len(datasets) - 0.3)
    # Set y-axis to log2 scale and customize ticks
    plt.yscale('log', base=2)

    # Find the maximum speedup value
    max_speedup = max([max(values) for alg, values in all_data.items() if values and not all(np.isnan(v) for v in values)], default=2.0)

    # Generate tick positions for powers of 2 up to near the max
    max_power = int(np.floor(np.log2(max_speedup)))
    yticks = [2**i for i in range(-3, max_power + 1)]
    plt.yticks(yticks, [f'{tick:.2f}' for tick in yticks])
    plt.ylim(2**-4, max_speedup * 1.1)  # Add 10% margin at the top

    # Add grid for better readability with more emphasis on horizontal lines
    plt.grid(True, which="major", ls="-", alpha=0.7)
    plt.grid(True, which="minor", ls=":", alpha=0.4)
    plt.gca().yaxis.grid(True, which='both')  # Ensure horizontal grid lines are shown
    plt.gca().xaxis.grid(False)  # Turn off vertical grid lines

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    filename_suffix = f"_{series_name}" if series_name else ""
    
    if args.batched:
        plt.savefig(f'{args.data_path}/speedup_batched_{args.series_type}_comp_{args.gpu_name}/speedup_{args.series_type}_batched_{args.gpu_name}{filename_suffix}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'{args.data_path}/speedup_full_graph_{args.series_type}_comp_{args.gpu_name}/speedup_{args.series_type}_full_graph_{args.gpu_name}{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_legend", action='store_true')
    parser.add_argument("--gpu_name", type=str, default="GH200")
    parser.add_argument("--batched", action='store_true')
    parser.add_argument("--data_path", type=str, default="kernel_only_comp_results")
    parser.add_argument("--series_type", type=str, default="external", 
                        choices=["external", "internal", "all"],
                        help="Type of series comparison to perform")
    args = parser.parse_args()
    main(args)