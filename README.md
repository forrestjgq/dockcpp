# Requirement
- python >= 3.6
- pytorch
- cuda development environment

# Build & Install
```sh
cd dockcpp
mkdir build
cd build
cmake ..
make install-python-package
```

# Verify installation
```python
import pydock
```

# Use
step 1: create a cuda context with specified CUDA device id
```python
device = 0
ctx = pydock.CudaContext(device)
```

step 2: prepare CPU data
- vt (torch.Tensor or type acceptable for torch.tensor()): x for f(x), 1 dimension
- init_coord (torch.Tensor): initial ligand position, dim: (N, 3)
- torsions (list): M x 2 int list for torsion angle, dim 0 for start node index, dim 1 for end node index
- masks (list of torch.Tensor(dtype=bool)): torsion masks, length should be M, sub length should be N
- pocket_coords (torch.Tensor): pocket positions, K x 3 float tensors
- pred_cross_dist (torch.Tensor): predicted distance from ligand to pociekts, N x K
- pred_holo_dist (torch.Tensor): predicted ligand holo distance, N X N

step3: calculate loss and grads
```python
    t = ctx.dock_grad(vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist)
    # t will be one dimention float tensor, size len(vt) + 1
    # the first float will be loss value, and rest will be grad for each value in vt
```


