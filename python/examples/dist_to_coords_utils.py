import copy
# from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import math
from torch.autograd import Variable


def axis_angle_to_quaternion(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    print(f'axis {axis_angle} angles {angles} half {half_angles}')
    small_angles = angles.abs() < eps
    print(f'small angles {small_angles}')
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    print(f'sins 1 {sin_half_angles_over_angles}')
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    print(f'sins 2 {sin_half_angles_over_angles}')
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    print(f'qu: {quaternions}')
    return quaternions


def quaternion_to_matrix(quaternions):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to rotation matrices.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def rigid_transform_Kabsch_3D_torch(A, B):
    # R = 3x3 rotation matrix, t = 3x1 column vector
    # This already takes residue identity into account.
    # print(f'SVD A shape {A.shape} B shape {B.shape}')

    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    print(f'A {A}\nB {B}')
    # find mean column wise: 3 x 1
    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    print(f'Center A {centroid_A} B {centroid_B}')
    print(f'Am {Am}\nBm {Bm}')
    print(f'H={H}')
    # find rotation
    U, S, Vt = torch.linalg.svd(H)
    # print(f'shape U {U.shape} S {S.shape} Vt {Vt.shape}')
    print(f'SVD: \nU={U}\nS={S}\nVt={Vt}')

    R = Vt.T @ U.T
    print(f'r {R}')
    d = torch.linalg.det(R)
    print(f'det {d}')
    # special reflection case
    if d < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = torch.diag(torch.tensor([1., 1., -1.], device=A.device))
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(torch.linalg.det(R) - 1) < 3e-3  # note I had to change this error bound to be higher

    t = -R @ centroid_A + centroid_B
    print(f'R {R}')
    print(f't {t}')
    return R, t


def modify_conformer(coords, values, edge_index, mask_rotate):
    print(f'coords {coords}')
    print(f'values {values}')
    # print(f'values {values.shape}')
    # print(f'edge {edge_index}')
    # print(f'mask {len(mask_rotate)} {[s.shape for s in mask_rotate]}')
    tr_update = (torch.sigmoid(values[:3]) - 0.5) * 10
    rot_update = (torch.sigmoid(values[3:6]) - 0.5) * np.pi * 2
    # torsion_updates = (torch.sigmoid(values[6:]) - 0.5) * np.pi * 2
    # rot_update = values[3:6]
    torsion_updates = values[6:]


    # print(f'rot update {rot_update} sq {rot_update.squeeze()}')
    lig_center = torch.mean(coords, dim=0, keepdim=True)
    # print(f'lig center {lig_center.shape} {lig_center}')
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    rigid_new_pos = (coords - lig_center) @ rot_mat.T + tr_update + lig_center
    print(f'tr update {tr_update} + center {tr_update + lig_center}')
    print(f'rot update {rot_update}')
    print(f'lig center {lig_center}')
    print(f'rot_mat {rot_mat}')
    print(f'rigid new pos {rigid_new_pos}')

    if values.shape[0] > 6:
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,
                                                           edge_index,
                                                           mask_rotate,
                                                           torsion_updates).to(rigid_new_pos.device)
        try:
            R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
            aligned_flexible_pos = flexible_new_pos @ R.T + t.T
            print(f'aligned {aligned_flexible_pos}')
            return aligned_flexible_pos
        except:
            return flexible_new_pos
    else:
        return rigid_new_pos


def modify_conformer2(coords, values, edge_index, mask_rotate):
    tr_update = (torch.sigmoid(values[:3]) - 0.5) * 10
    rot_update = (torch.sigmoid(values[3:6]) - 0.5) * np.pi * 2
    # torsion_updates = (torch.sigmoid(values[6:]) - 0.5) * np.pi * 2
    # rot_update = values[3:6]
    torsion_updates = values[6:]

    lig_center = torch.mean(coords, dim=0, keepdim=True)
    rigid_new_pos = coords - lig_center

    if values.shape[0] > 6:
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,
                                                           edge_index,
                                                           mask_rotate,
                                                           torsion_updates).to(rigid_new_pos.device)
        try:
            R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
            holo_pos = flexible_new_pos @ R.T + t.T
        except:
            holo_pos = flexible_new_pos
    else:
        holo_pos = rigid_new_pos
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    out_pos = holo_pos @ rot_mat.T + tr_update + lig_center
    return out_pos


def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates):
    pos = copy.deepcopy(pos)

    for idx_edge, e in enumerate(edge_index):
        # if torsion_updates[idx_edge] == 0:
        #     continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        # assert not mask_rotate[idx_edge, u]
        # assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        print(f'idx edge {idx_edge} u {u} v {v} rot_vec {rot_vec}')
        rot_vec = rot_vec / torch.norm(rot_vec)  # idx_edge!
        print(f'rot_vec norm {rot_vec}')
        print(f'torsion updates {torsion_updates}')
        rot_mat = gen_matrix_from_rot_vec(rot_vec, torsion_updates[idx_edge])
        print(f'rot_mat {rot_mat}')

        print(f'idx {idx_edge} before {pos}\nmask {mask_rotate[idx_edge]}')

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]
        print(f'idx {idx_edge} after {pos}')

    print(f'modify conformer ret pos {pos}')
    return pos


def gen_matrix_from_rot_vec(k, theta):
    K = torch.zeros((3, 3), device=k.device)
    K[[1, 2, 0], [2, 0, 1]] = -k
    K[[2, 0, 1], [1, 2, 0]] = k
    print(f"theta {theta} K: {K}")
    print(f'eye {torch.eye(3)} sin {torch.sin(theta)} cos {torch.cos(theta)} 1-cos {1 - torch.cos(theta)}')
    print(f"KK {torch.matmul(K, K)}")
    R = torch.eye(3) + K * torch.sin(theta) + (1 - torch.cos(theta)) * torch.matmul(K, K)
    print(f"R {R}")
    return R


