"""
本脚本用于对单个口袋和多个小分子（smiles或带初始构象的mol文件）进行对接
"""
import torch
import os
import numpy as np
from dist_to_coords_utils import modify_conformer
import sys
import torch.nn.functional as F
import pydock


def savetensors(seq, values, init_coord, torsions, masks, pocket, pred_cross_dist, pred_holo_dist, loss):
    path = os.path.join("/home/jgq/src/docking/tensors", f'{seq}')
    os.makedirs(path, exist_ok=True)
    target = {
        "values": values,
        "init_coord": init_coord,
        "torsions": torsions,
        "masks": masks,
        "pocket": pocket,
        "pred_cross_dist": pred_cross_dist,
        "pred_holo_dist": pred_holo_dist,
        "loss": loss
    }
    for k, v in target.items():
        torch.save(v, os.path.join(path, k))
def loadtensors(seq):
    target = [
        "values",
        "init_coord",
        "torsions",
        "masks",
        "pocket",
        "pred_cross_dist",
        "pred_holo_dist",
        "loss"
    ]
    path = "/home/forrest/project/dockcpp/python/examples/tensors"
    return (torch.load(os.path.join(path, str(seq), name)) for name in target)
        
        
def dumpcpp(seq, values, init_coord, torsions, masks, pocket, pred_cross_dist, pred_holo_dist, loss):
    path = os.path.join("/home/jgq/src/docking/cases", f'{seq}.h')
    target = {
        "values": values,
        "init_coord": init_coord,
        "torsions": torsions,
        "masks": masks,
        "pocket": pocket,
        "pred_cross_dist": pred_cross_dist,
        "pred_holo_dist": pred_holo_dist,
        "loss": loss.item()
    }

    with open(path, 'w') as fp:
        for k, v in target.items():
            print(f'dump {k} type {type(v)}')
            isBool = False
            if isinstance(v, list):
                if isinstance(v[0], torch.Tensor):
                    print(f'v0.dtype {v[0].dtype}')
                    if v[0].dtype == torch.bool:
                        vs = [t.int().tolist() for t in v]
                        isBool = True
                    else:
                        vs = [t.tolist() for t in v]
                    v = np.array(vs)
                else:
                    v = np.array(v)
            if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                val = v.reshape(-1).tolist()
                sval = [str(i) for i in val]
                s = '{' + ','.join(sval) + '}'
                name = k + '[]'
                if isBool:
                    tp = 'uint8_t'
                elif str(v.dtype).startswith('torch.float') or v.dtype.kind == 'f':
                    tp = 'float'
                elif str(v.dtype).startswith('torch.int') or v.dtype.kind == 'i':
                    tp = 'int'
                else:
                    assert False, f'unknown {k} type {v.dtype}'
            elif isinstance(v, float):
                s = str(v)
                tp = 'float'
                name = k
            else:
                assert False, f'unknown {k} type {type(v)}'
            fp.write(f'{tp} {name} = {s};\n')
            
                
        
seq = 0

DEVICE = torch.device('cuda:1')
use_cuda = False

def load(file):
    arr = torch.load( "/home/jgq/src/docking/data/" + file)
    if isinstance(arr, list):
        print(f'{file} len: {len(arr)} value {arr}')
        for v in arr:
            print(f'shape {v.shape}')
    else:
        print(f'{file} shape: {arr.shape}')
    if use_cuda:
        return arr.to(DEVICE)
    return arr

def single_SF_loss(
        predict_coords,
        pocket_coords,
        distance_predict,
        holo_distance_predict,
        dist_threshold=6,
        # dist_threshold=10,

):
    # dist = dist.unsqueeze(0)
    dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    t = dist.reshape(-1)[0:80*8].reshape((80,8))
    print(f'dist {t}')
    holo_dist = torch.norm(
        predict_coords.unsqueeze(1) - predict_coords.unsqueeze(0), dim=-1
    )
    distance_mask = distance_predict < dist_threshold
    cross_dist_score = F.smooth_l1_loss(distance_predict[distance_mask], dist[distance_mask])
    sum_cross_dist_score = F.smooth_l1_loss(distance_predict[distance_mask], dist[distance_mask], reduction='sum')
    print(f'sum {sum_cross_dist_score}')
    dist_score = F.smooth_l1_loss(holo_distance_predict, holo_dist)
    loss = cross_dist_score * 1.0 + dist_score * 5.0
    return loss

def test_py(seq):
    vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist, gt = loadtensors(seq)
    new_pos = modify_conformer(init_coord, vt, torsions, masks)
    loss = single_SF_loss(new_pos, pocket_coords, pred_cross_dist, pred_holo_dist)
    print(f'gt {gt} loss {loss}')

def test_grad_seq(seq, ctx):
    vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist, gt = loadtensors(seq)
    t = ctx.dock_grad(vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist)
    print(f't={t}')
    
def test_seq(seq, ctx):
    vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist, gt = loadtensors(seq)
    ctx = pydock.CudaContext(0) # cuda context on device 0
    t = ctx.dock(vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist)
    print(f't={t}')


if __name__ == '__main__':
    assert len(sys.argv) >= 3
    action = sys.argv[1]
    start = int(sys.argv[2])
    if len(sys.argv) > 3:
        end = int(sys.argv[3]) + 1
    else:
        end = start+1
    ctx = pydock.CudaContext(0)
    for seq in range(start, end):
        print(f'\n\n==== test {seq} ====')
        if action == 'py':
            # python version of conformance test
            test_py(seq)
        elif action == 'one':
            # Cuda calculates a single loss
            test_seq(seq, ctx)
        elif action == 'grad':
            # Cuda calculates loss and grads
            test_grad_seq(seq, ctx)