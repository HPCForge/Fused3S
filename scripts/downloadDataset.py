#!/usr/bin/env python3
import numpy as np
import os

def save_dataset(name, num_nodes, src_li, dst_li, output_file):
    np.savez(
        output_file,
        src_li=src_li,
        dst_li=dst_li,
        num_nodes=num_nodes
    )
    print("-"*10 + f" {name} " + "-"*10)
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {len(src_li)}")
    print(f"Dataset saved as {output_file}")
    print("-"*10)

def download_pyg_dataset(path, output_path, name='Github'):
    from torch_geometric.datasets import GitHub, Yelp, AmazonProducts, \
      Planetoid, Reddit, EllipticBitcoinDataset
    temp_path = path+'/'+name
    os.makedirs(temp_path, exist_ok=True)
    if name == 'GitHub':
        dataset = GitHub(root=temp_path)
    elif name == 'Yelp':
        dataset = Yelp(root=temp_path)
    elif name == 'AmazonProducts':
        dataset = AmazonProducts(root=temp_path)
    elif name == 'Reddit':
        dataset = Reddit(root=temp_path)
    elif name == 'EllipticBitcoinDataset':
        dataset = EllipticBitcoinDataset(root=temp_path)
    elif name in ['Pubmed', 'Cora', 'Citeseer']:
        dataset = Planetoid(root=temp_path, name=name)
    else:
        import shutil
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        raise ValueError(f"Dataset {name} not supported")
    data = dataset[0]

    # Extract edge indices
    edge_index = data.edge_index.numpy()
    src_li = edge_index[0]
    dst_li = edge_index[1]
    num_nodes = data.num_nodes
    
    # Save to npz format
    output_file = f"{output_path}/{name}.npz"
    save_dataset(name, num_nodes, src_li, dst_li, output_file)

def download_ogb_dataset(path, output_path, name='ogbn-products'):
    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(name=name, root=path)
    data = dataset[0]
    edge_index = data.edge_index.numpy()
    src_li = edge_index[0]
    dst_li = edge_index[1]
    num_nodes = data.num_nodes
    output_file = f"{output_path}/{name}.npz"
    save_dataset(name, num_nodes, src_li, dst_li, output_file)

def download_igb_dataset(path, output_path, size='medium'):
    from igb.dataloader import IGB260M
    if size in ['medium', 'small']:
      data = IGB260M(root=path+'/igb', size=size, in_memory=True, \
              classes=1, synthetic=0)

      # Extract edge indices
      edge_index = data.paper_edge
      src_li = edge_index[:, 0]
      dst_li = edge_index[:, 1]
      num_nodes = data.num_nodes()
      
      # Save to npz format
      output_file = f"{output_path}/igb_{size}.npz"
      save_dataset(f"igb_{size}", num_nodes, src_li, dst_li, output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--output_path', type=str, default='Fused3S/dataset')
    args = parser.parse_args()
    for size in ['medium', 'small']:
        download_igb_dataset(args.root, args.output_path, size=size)
    for name in ['AmazonProducts', 'GitHub', 'Yelp', 'Pubmed', 'Cora', 'Citeseer']:
        download_pyg_dataset(args.root, args.output_path, name=name)
    for name in ['ogbn-products']:
        download_ogb_dataset(args.root, args.output_path, name=name)
