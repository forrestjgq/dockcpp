import os as _os
import glob as _glob
from ctypes import CDLL as _CDLL
import ctypes
import torch

from pydock._build_config import _build_config  # NOQA: E402
__pydock_lib_version__ = _build_config["PYDOCK_VERSION"]
__pydock_lib_build_type__ = _build_config["PYDOCK_BUILD_TYPE"]

import pydock.cudock as _cu

class CudaContext:
    def __init__(self, device):
        self._device = device
        self._cu_ctx = _cu.create_cuda_context(device)
        assert self._cu_ctx is not None
    def dock_grad(self, vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist):
        if not isinstance(vt, torch.Tensor):
            vt = torch.tensor(vt, dtype=torch.float)
        masks = [[1 if m else 0 for m in item] for item in masks]
        masks = torch.tensor(masks,dtype=torch.uint8)
        torsions = torch.tensor(torsions, dtype=torch.int)
        return _cu.dock_grad(self._cu_ctx, init_coord, pocket_coords, pred_cross_dist, pred_holo_dist, vt, torsions, masks)


    def dock(self, vt, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist):
        if not isinstance(vt, torch.Tensor):
            vt = torch.tensor(vt, dtype=torch.float)
        masks = [[1 if m else 0 for m in item] for item in masks]
        masks = torch.tensor(masks,dtype=torch.uint8)
        torsions = torch.tensor(torsions, dtype=torch.int)
        return _cu.dock(self._cu_ctx, init_coord, pocket_coords, pred_cross_dist, pred_holo_dist, vt, torsions, masks)