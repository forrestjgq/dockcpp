"""
本脚本用于对单个口袋和多个小分子（smiles或带初始构象的mol文件）进行对接
"""
import torch
import copy
import math
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
def loadtensors1(seq):
    target = [
        "values",
        "init_coord",
        "torsions",
        "masks",
        "pocket",
        "pred_cross_dist",
        "pred_holo_dist",
        # "loss"
    ]
    path = f"/home/jgq/src/docking-master/failed/{seq}"
    # path = "/home/forrest/project/dockcpp/python/examples/tensors"
    return (torch.load(os.path.join(path, f'{name}')) for name in target)
def loadtensors(seq):
    target = [
        "values",
        "init_coord",
        "torsions",
        "masks",
        "pocket_coords",
        "pred_cross_dist",
        "pred_holo_dist",
        # "loss"
    ]
    path = f"/home/jgq/src/docking-master/failed"
    # path = "/home/forrest/project/dockcpp/python/examples/tensors"
    return (torch.load(os.path.join(path, f'{seq}_{name}')) for name in target)
        
        
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
    print(f'dist shape {dist.shape}:')
    for i in range(dist.shape[0]):
        print(f'{i}: {str(dist[i])}')
    holo_dist = torch.norm(
        predict_coords.unsqueeze(1) - predict_coords.unsqueeze(0), dim=-1
    )
    print(f'holo dist shape {holo_dist.shape}:')
    for i in range(holo_dist.shape[0]):
        print(f'{i}: {str(holo_dist[i])}')
    distance_mask = distance_predict < dist_threshold
    cross_dist_score = F.smooth_l1_loss(distance_predict[distance_mask], dist[distance_mask])
    sum_cross_dist_score = F.smooth_l1_loss(distance_predict[distance_mask], dist[distance_mask], reduction='sum')
    dist_score = F.smooth_l1_loss(holo_distance_predict, holo_dist)
    print(f'cross score {cross_dist_score} dist score {dist_score}')
    loss = cross_dist_score * 1.0 + dist_score * 1.0
    return loss

def test_py(seq):
    vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist = loadtensors(seq)
    new_pos = modify_conformer(init_coord, vt, torsions, masks)
    loss = single_SF_loss(new_pos, pocket_coords, pred_cross_dist, pred_holo_dist)
    print(f'loss {loss}')

def test_grad_py(seq):
    vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist = loadtensors(seq)
    new_pos = modify_conformer(init_coord, vt, torsions, masks)
    loss = single_SF_loss(new_pos, pocket_coords, pred_cross_dist, pred_holo_dist)

    t = [loss.item()]
    for i in range(len(vt)):
        l = copy.deepcopy(vt.tolist())
        l[i] += 0.05
        v = torch.tensor(l)
        new_pos = modify_conformer(init_coord, v, torsions, masks)
        loss1 = single_SF_loss(new_pos, pocket_coords, pred_cross_dist, pred_holo_dist)
        t.append((loss1.item() - loss.item())/0.05)
    print(f't={t}')


def test_grad_seq(seq, ctx):
    vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist, gt = loadtensors(seq)
    t, ok = ctx.dock_grad(vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist)
    assert ok
    print(f't={t}')
    
def test_seq(seq, ctx):
    vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist, gt = loadtensors(seq)
    t, ok = ctx.dock(vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist)
    assert ok
    print(f't={t}')

def test_session_seq1(seq, ctx):
    vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist, gt = loadtensors(seq)
    s = ctx.new_session(init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist, len(vt))
    # gt = [1.7734, 1.7735, 1.7734, 1.7738, 1.7739, 1.7743, 1.7733, 1.7734, 2.5332]
    # gt = [1.7734, 1.7734, 1.7735, 1.7734, 1.7738, 1.7739, 1.7743, 1.7733, 1.7734]
    gt = [ 1.7734,  0.0138,  0.0484, -0.0246,  0.4230,  0.5088,  0.8847, -0.0858, -0.0420]
    for i in range(1):
        t, ok = ctx.session_submit(s, vt)
        assert ok
        # print(f'i={i} t={t}')
        failed = False
        for i in range(len(gt)):
            if math.isnan(t[i]) or abs(gt[i] - t[i]) > 0.0001:
                print(f'i: {i}, gt {gt[i]} t {t[i]}')
                failed = True
        assert not failed, (i, t)

def test_session_seq(seq, ctx):
    vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist = loadtensors1(seq)
    vt = torch.tensor([0.117664,-0.482597,-0.596182,-0.782030,-1.486720,0.675532,0.307211])
    s = ctx.new_session(init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist, len(vt))
    t, ok = ctx.session_submit(s, vt)
    print(f't={t}')
    assert ok

def test_lbfgsb_seq(seq):
    vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist = loadtensors1(seq)
    values = torch.zeros(vt.shape[0], device=init_coord.device, requires_grad=False)
    t, best, ok = pydock.lbfgsb(6, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist, values)
    assert ok
    print(f'best = {best} t={t}')

def test_lbfgsb_srv_seq(seq):
    vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist = loadtensors1(seq)
    print(f'torsions {torsions}')
    values = torch.zeros(vt.shape[0], device=init_coord.device, requires_grad=False)
    lb = pydock.LBFGSBServer(1, 6)
    stub = lb.dock_optimize(init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist, values)
    print(f'stub {stub}')
    rsp = lb.poll()
    print(f'rsp {rsp}')

if __name__ == '__main__':
    assert len(sys.argv) >= 3
    action = sys.argv[1]
    start = int(sys.argv[2])
    if len(sys.argv) > 3:
        end = int(sys.argv[3]) + 1
    else:
        end = start+1
    ctx = pydock.CudaContext(6)
    for seq in range(start, end):
        print(f'\n\n==== test {seq} ====')
        if action == 'py':
            # python version of conformance test
            test_py(seq)
        elif action == 'gpy':
            # Cuda calculates a single loss
            test_grad_py(seq)
        elif action == 'one':
            # Cuda calculates a single loss
            test_seq(seq, ctx)
        elif action == 'grad':
            # Cuda calculates loss and grads
            test_grad_seq(seq, ctx)
        elif action == 'session':
            # Cuda calculates loss and grads
            test_session_seq(seq, ctx)
        elif action == 'lb':
            test_lbfgsb_seq(seq)
        elif action == 'lbs':
            test_lbfgsb_srv_seq(seq)