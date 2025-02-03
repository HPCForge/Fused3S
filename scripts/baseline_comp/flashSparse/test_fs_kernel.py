from torch_geometric.utils import softmax
import argparse
import time
import torch
from agnn_dataset import *
import FS_SDDMM
import FS_SpMM

def test(data, args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MAGNN_dataset(data, num_features=3, num_classes=10)
    inputInfo.to(device)

    edge_att_rand = torch.rand(inputInfo.num_edges).half().to(device)
    
    X_prime = torch.rand(inputInfo.num_nodes, args.hidden).half().to(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    n_iter = 10
    if(args.softmax_type == 0):
      for i in range(n_iter):
        torch.cuda.synchronize()
        start_event.record()
      
        att = FS_SDDMM.forward_gen_fp16_gnn(   
                X_prime.size(1),                                      
                inputInfo.row_pointers, 
                inputInfo.column_index, 
                inputInfo.degrees, 
                inputInfo.t_window_rowTensor,
                X_prime,X_prime,inputInfo.max)[0] 
        h_prime = FS_SpMM.forward_fp16_gnn(   
                    inputInfo.row_pointers, 
                    inputInfo.column_index, 
                    att, 
                    inputInfo.t_window_rowTensor,
                    inputInfo.t_atomicTensor,
                    X_prime, 
                    inputInfo.num_nodes, 
                    X_prime.size(1), 
                    inputInfo.num_nodes_ori)[0].half()

        end_event.record()
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        print(f"execution time: {execution_time} ms")
    elif(args.softmax_type == 1):
      for i in range(n_iter):
        torch.cuda.synchronize()
        start_event.record()
      
        att = FS_SDDMM.forward_gen_fp16_gnn(   
                X_prime.size(1),                                      
                inputInfo.row_pointers, 
                inputInfo.column_index, 
                inputInfo.degrees, 
                inputInfo.t_window_rowTensor,
                X_prime,X_prime,inputInfo.max)[0] 
        
        att = torch.exp(att) # softmax
        rows_sum = FS_SpMM.forward_fp16_gnn_ones(   
                    inputInfo.row_pointers, 
                    inputInfo.column_index, 
                    att, 
                    inputInfo.t_window_rowTensor,
                    inputInfo.t_atomicTensor,
                    inputInfo.ones, 
                    inputInfo.num_nodes, 
                    inputInfo.ones.size(1), 
                    inputInfo.num_nodes_ori)[0]
        h_prime = FS_SpMM.forward_fp16_gnn(   
                    inputInfo.row_pointers, 
                    inputInfo.column_index, 
                    att, 
                    inputInfo.t_window_rowTensor,
                    inputInfo.t_atomicTensor,
                    X_prime, 
                    inputInfo.num_nodes, 
                    X_prime.size(1), 
                    inputInfo.num_nodes_ori)[0].half()
        h_prime = h_prime.div(rows_sum) # softmax

        end_event.record()
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        print(f"execution time: {execution_time} ms")
    else:
      for i in range(n_iter):
        torch.cuda.synchronize()
        start_event.record()
      
        att = FS_SDDMM.forward_gen_fp16_gnn(   
                X_prime.size(1),                                      
                inputInfo.row_pointers, 
                inputInfo.column_index, 
                inputInfo.degrees, 
                inputInfo.t_window_rowTensor,
                X_prime,X_prime,inputInfo.max)[0] 
        softmax(edge_att_rand, ptr=inputInfo.orig_row_pointers.to(device))
        h_prime = FS_SpMM.forward_fp16_gnn(   
                    inputInfo.row_pointers, 
                    inputInfo.column_index, 
                    att, 
                    inputInfo.t_window_rowTensor,
                    inputInfo.t_atomicTensor,
                    X_prime, 
                    inputInfo.num_nodes, 
                    X_prime.size(1), 
                    inputInfo.num_nodes_ori)[0].half()

        end_event.record()
        torch.cuda.synchronize()
        execution_time = start_event.elapsed_time(end_event)
        print(f"execution time: {execution_time} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--softmax_type', '-s', type=int, default=0, help='0: no softmax, 1: naive softmax, 2: stable softmax')
    parser.add_argument('--dataset', '-d', type=str, default='reddit', help='dataset name')
    parser.add_argument('--hidden', type=int, default=128, help='hidden dimension')
    args = parser.parse_args()
    test(f'/pub/zitongl5/DTC-SpMM-Datasets/{args.dataset}.npz', args)
