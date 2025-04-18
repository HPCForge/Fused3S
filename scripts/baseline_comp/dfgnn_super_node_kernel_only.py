# adapted from DF-GNN/DFGNN/script/test/test_full_graph.py and test_full_graph_super_node.sh
# only test the kernel instead of full forward pass of a GT model.
import argparse
import dgl.sparse as dglsp
import torch
from DFGNN.layers import load_prepfunc
from dgl.data import RedditDataset
from ogb.linkproppred import DglLinkPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from DFGNN.operators.fused_gtconv import GTConvFuse_inference_csr_gm, GTConvFuse_inference_tiling, GTConvFuse_inference_softmax_gm

def preprocess_dglsp(g):
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))
    return A

def PrintGraphStruct(g):
    print("----------graph statics -----------")
    print(f"# of nodes {g.num_nodes()}")
    print(f"# of edges {g.num_edges()}")
    print(f"avg. degree {torch.mean(g.out_degrees().float()):.2f}")
    print(f"max. degree {max(g.out_degrees())}")


def test_format(args, dev, g):
    X = g.ndata["feat"]
    num_attention_heads = 1
    Q = torch.rand(X.shape[0], num_attention_heads, args.embedding_dim, dtype=torch.float32, device=dev)
    K = torch.rand(X.shape[0], num_attention_heads, args.embedding_dim, dtype=torch.float32, device=dev)
    V = torch.rand(X.shape[0], num_attention_heads, args.embedding_dim, dtype=torch.float32, device=dev)
    preprocess_func = load_prepfunc(args)
    if args.format == "csr_gm":
        indptr, indices, val, _ = preprocess_func(g)
        out = GTConvFuse_inference_csr_gm(indptr, indices, val, Q, K, V)
    elif args.format == "tiling":
        indptr, indices, val, smem_consume = preprocess_func(g)
        out = GTConvFuse_inference_tiling(indptr, indices, val, smem_consume, Q, K, V)
    elif args.format == "softmax_gm":
        indptr, indices, rows, val, _ = preprocess_func(g)
        out = GTConvFuse_inference_softmax_gm(indptr, indices, rows, val, Q, K, V)
    else:
        raise ValueError(f"format: {args.format} not supported in this test")

    torch.cuda.synchronize()
    print(out.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str, default="csr_gm", choices=["csr_gm", "tiling", "softmax_gm"])
    parser.add_argument("--embedding_dim", "-d", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="reddit", choices=["reddit", "ppa", "protein"])
    parser.add_argument("--dataDir", type=str, default="/share/crsp/lab/amowli/share/Fused3S/dfgnn")
    args = parser.parse_args()
    assert args.embedding_dim < 320, "dim should be less than 320"
    assert args.embedding_dim % 32 == 0, "dim should be divisible by 32"

    print("format:", args.format)
    print("dataset:", args.dataset)
    print("hidden dim", args.embedding_dim)

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    if args.dataset == "reddit":
        dataset = RedditDataset(raw_dir=args.dataDir)
    elif args.dataset == "ppa":
        dataset = DglLinkPropPredDataset(name="ogbl-ppa", root=args.dataDir)
    elif args.dataset == "protein":
        dataset = DglNodePropPredDataset(name="ogbn-proteins", root=args.dataDir)
    else:
        raise ValueError(f"dataset: {args.dataset} not supported in this test")
    g = dataset[0].to(dev)
    PrintGraphStruct(g)

    test_format(args, dev, g)
