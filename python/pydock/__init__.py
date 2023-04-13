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
        
def lbfgsb(device, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist, init_values, eps=0.01):
    """create a session for one optimization

    Args:
        device (int): cuda device id
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
    # print(f'pocket type {type(pocket_coords)} shape {pocket_coords.shape}')
    # print(f'pred cross dist type {type(pred_cross_dist)} shape {pred_cross_dist.shape}')
    # print(f'pred holo dist type {type(pred_holo_dist)} shape {pred_holo_dist.shape}')
    return _cu.lbfgsb(device, init_coord, pocket_coords, pred_cross_dist, pred_holo_dist, torsions, masks, init_values, eps)

class LBFGSBServer:
    def __init__(self, n, device) -> None:
        """create an LBFGSB optimizer server

        Args:
            n (int): number of instances of LBFGSB optimizer
            device (int): cuda device id, device >= 0
        """
        assert n > 0 and device >= 0
        self._srv = _cu.create_lbfgsb_server(device, n)
        assert self._srv is not None, "lbfgsb server creation failure"
        self._running = 0
    
    def in_run(self):
        """get how many request are sent but not responded
        """
        return self._running
    

    def dock_optimize(self, init_coord, torsions, masks, pocket_coords, pred_cross_dist, pred_holo_dist, init_values, eps=0.01):
        """post optimize request to server

        optimize with LBFGSB server is an async operation, user should call dock_optimize to 
        submit a request, or more requests, and then call poll() to get response.

        Args:
            init_coord (torch.Tensor): initial ligand position, dim: (N, 3)
            torsions (list): M x 2 int list for torsion angle, dim 0 for start node index, dim 1 for end node index
            masks (list of torch.Tensor(dtype=bool)): torsion masks, length should be M, sub length should be N
            pocket_coords (torch.Tensor): pocket positions, K x 3 float tensors
            pred_cross_dist (torch.Tensor): predicted distance from ligand to pociekts, N x K
            pred_holo_dist (torch.Tensor): predicted ligand holo distance, N X N
            nval (int): length of values (x in f(x))
            init_values (_type_): initial values
            eps (float): delta x for dy/dx

        Returns:
            tuple of (int, bool): int for a sequence number, bool for post result, True for success
        """
        masks = [[1 if m else 0 for m in item] for item in masks]
        masks = torch.tensor(masks,dtype=torch.uint8)
        torsions = torch.tensor(torsions, dtype=torch.int)
        req = _cu.create_lbfgsb_dock_request(init_coord, pocket_coords, pred_cross_dist, pred_holo_dist, torsions, masks, init_values, eps)
        seq, ok = _cu.post_lbfgsb_request(self._srv, req)
        if ok:
            self._running += 1
        return seq, ok
    
    def poll(self, n):
        """poll optimize response, it will NOT return until all expected responses are received

        Args:
            n (int, optional): how many responses are expected. If n <= 0 or n > running(in_run()), responses for all running requests
                               are expected

        Returns:
            list of(tuple of(torch.Tensor, float, int, bool)): torch.Tensor will be best values, float will be the best loss,
                                                      int is the sequence number created by dock_optimize(), and bool
                                                      will be the result, other args are valid only when this value is
                                                      True
                                                      
                                                      if no running requests, None will be returned
        """
        sz = self._running
        if sz == 0:
            return None
        if n > sz or n <= 0:
            n = sz
        ret = []
        # print(f'sz {sz} n {n} running {self._running}')
        while n > 0:
            val, loss, seq, ok = _cu.poll_lbfgsb_response(self._srv)
            ret.append((val, loss, seq, ok,))
            n-=1
            self._running -= 1
        return ret

    def poll(self):
        """poll one optimize response

        Returns:
            tuple of(torch.Tensor, float, int, bool) or None: torch.Tensor will be best values, float will be the best loss,
                                                      int is the sequence number created by dock_optimize(), and bool
                                                      will be the result, other args are valid only when this value is
                                                      True.
                                                      
                                                      if no running requests, None will be returned
        """
        sz = self._running
        if sz == 0:
            return None
        return _cu.poll_lbfgsb_response(self._srv)
            