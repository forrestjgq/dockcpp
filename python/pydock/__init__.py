import torch

from pydock._build_config import _build_config  # NOQA: E402
__pydock_lib_version__ = _build_config["PYDOCK_VERSION"]
__pydock_lib_build_type__ = _build_config["PYDOCK_BUILD_TYPE"]

import pydock.cudock as _cu

class CudaContext:
    """CudaContext provide APIs for docking to run on specified GPU CUDA device
    
    CudaContext will start a thread internally and allocate GPU resources for tasks.
    """
    def __init__(self, device):
        self._device = device
        self._cu_ctx = _cu.create_cuda_context(device)
        assert self._cu_ctx is not None, f"Create cuda context failed, make sure cuda device {device} is available"

    def new_session(self, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist, nval, eps=0.01):
        """create a session for one optimization

        Args:
            init_coord (torch.Tensor): initial ligand position, dim: (N, 3)
            torsions (list): M x 2 int list for torsion angle, dim 0 for start node index, dim 1 for end node index
            masks (list of torch.Tensor(dtype=bool)): torsion masks, length should be M, sub length should be N
            pocket_coords (torch.Tensor): pocket positions, K x 3 float tensors
            pred_cross_dist (torch.Tensor): predicted distance from ligand to pociekts, N x K
            pred_holo_dist (torch.Tensor): predicted ligand holo distance, N X N
            nval (int): length of values (x in f(x))
            eps (float): delta x for dy/dx

        Returns:
            _cu.Request: session handle
        """
        masks = [[1 if m else 0 for m in item] for item in masks]
        masks = torch.tensor(masks,dtype=torch.uint8)
        torsions = torch.tensor(torsions, dtype=torch.int)
        return _cu.create_dock_session(self._cu_ctx, init_coord, pocket_coords, pred_cross_dist, pred_holo_dist, torsions, masks, eps, nval)

    def session_submit(self, session, vt):
        """_summary_

        Args:
            session (_cu.Request): return value of new_session()
            vt (torch.Tensor or type acceptable for torch.tensor()): x for f(x), 1 dimension

        Returns:
            tuple of (torch.Tensor, bool)
            torch.Tensor: 1 dimension tensor, length should be vt.shape[0]+1 the first element should be the loss, 
                          the rest will be the grad value for each value in vt 
        """
        return _cu.dock_submit(self._cu_ctx, session, vt);
        