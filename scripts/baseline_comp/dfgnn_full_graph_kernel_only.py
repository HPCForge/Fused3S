import torch
from DFGNN.layers import load_prepfunc
from DFGNN.layers.util import preprocess_softmax, preprocess_CSR
from DFGNN.operators.fused_gtconv import GTConvFuse_inference_softmax, GTConvFuse_inference_csr, GTConvFuse_inference_hyper
from dgl.data import (CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset)
import argparse

def main(args):
  num_heads = 1
  dataset_dir = "/share/crsp/lab/amowli/share/Fused3S/dfgnn/"
  dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  if args.dataset == "citeseer":
    dataset = CiteseerGraphDataset(raw_dir=dataset_dir)
  elif args.dataset == "pubmed":
    dataset = PubmedGraphDataset(raw_dir=dataset_dir)
  elif args.dataset == "cora":
    dataset = CoraGraphDataset(raw_dir=dataset_dir)
  else:
    raise ValueError(f"Invalid dataset: {args.dataset}")

  g = dataset[0].to(dev)
  num_nodes = g.num_nodes()
  num_edges = g.num_edges()
  Q = torch.rand(num_nodes, num_heads, args.embedding_dim, dtype=torch.float32, device=dev)
  K = torch.rand(num_nodes, num_heads, args.embedding_dim, dtype=torch.float32, device=dev)
  V = torch.rand(num_nodes, num_heads, args.embedding_dim, dtype=torch.float32, device=dev)
  print(g.ndata["feat"].shape)
  preprocess_func = load_prepfunc(args)
  if args.format == "softmax":
    indptr, indices, rows, val, smem_consume = preprocess_func(g)
    out = GTConvFuse_inference_softmax(indptr, indices, rows, val, smem_consume, Q, K, V)
  elif args.format == "csr":
    indptr, indices, val, smem_consume = preprocess_func(g)
    out = GTConvFuse_inference_csr(indptr, indices, val, smem_consume, Q, K, V)
  elif args.format == "hyper":
    indptr, indices, rows, val, smem_consume = preprocess_func(g)
    out = GTConvFuse_inference_hyper(indptr, indices, rows, val, smem_consume, Q, K, V)
  else:
    raise ValueError(f"Invalid format: {args.format}")
  torch.cuda.synchronize()
  print(out.shape)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', '-emb', type=int, default=128,
                       help='Dimension of node embeddings')
    parser.add_argument('--format', '-f', type=str, default="softmax",
                        choices=["softmax", "csr", "hyper"],
                        help='Format to use')
    parser.add_argument('--dataset', '-d', type=str, default="pubmed",
                       choices=["citeseer", "pubmed", "cora"],
                       help='Dataset to use')
    args = parser.parse_args()
    main(args)
