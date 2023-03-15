#include <algorithm>

#include <cuda_runtime.h>

#include "math.h"

namespace dock {

#include "svd3_cuda.h"
// calc 1/x in round-to-nearest-even mode
#define reciprocal(x) __frcp_rn(x)

__device__ __forceinline__ float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ void sub3p(float *a, float *b, float *out) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}
__device__ __forceinline__ void add3p(float *a, float *b, float *out) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
}
__device__ __forceinline__ void sub3p(float *a, float *b) {
    a[0] = a[0] - b[0];
    a[1] = a[1] - b[1];
    a[2] = a[2] - b[2];
}
__device__ __forceinline__ void add3p(float *a, float *b) {
    a[0] = a[0] + b[0];
    a[1] = a[1] + b[1];
    a[2] = a[2] + b[2];
}
// https://zhuanlan.zhihu.com/p/543841297
// sigmoid(x) = 1/(1+e^(-x))
template <typename T>
__device__ __forceinline__ T SigmoidForward(T x) {
    return T(1) / (T(1) + expf(-x));
}
template <typename T>
__device__ __forceinline__ void SigmoidBackward(T dy, T y) {
    return dy * y * (T(1) - y);
}
// sum(abs(x)**ord)**(1./ord) where ord = 2
template <int n>
__device__ __forceinline__ float Norm2(float *d) {
    float f = 0;
    for (int i = 0; i < n; i++) {
        float x = d[i];  // abs is ignored for ^2
        f += x * x;
    }
    return sqrtf(f);
}

#if 0
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
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
// for x small, sin(x / 2) is about x / 2 - (x / 2) ^ 3 / 6
// so sin(x / 2) / x is about 1 / 2 - (x * x) / 48
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions
#endif
__device__ __forceinline__ float4 axis_angle_to_quaternion(float3 axis) {
    float angle = norm3df(axis.x, axis.y, axis.z);  // sqrt(x^2 + y^2 + z^2)
    float half  = angle * 0.5f;
    float sins;
    if (angle < 1e-6) {
        sins = 0.5f - (angle * angle) * 0.020833333333333332;  // 1/48 = 0.020833333333333332
    } else {
        sins = sinf(angle) * reciprocal(angle);
    }
    float4 ret;
    ret.x = cosf(half);
    ret.y = axis.x * sins;
    ret.z = axis.y * sins;
    ret.w = axis.z * sins;
    return ret;
}
#if 0

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
#endif
// m: 3*3
__device__ __forceinline__ void quaternion_to_matrix(float4 q, float *m) {
    float4 q2 = make_float4(q.x * q.x, q.y * q.y, q.z * q.z, q.w * q.w);
    float s   = reciprocal(q2.x + q2.y + q2.z + q2.w) * 2.0;
    m[0] = 1 - s * (q2.z + q2.w), m[1] = s * (q.y * q.z - q.w * q.x),
    m[2] = s * (q.y * q.w + q.z * q.x), m[3] = s * (q.y * q.z + q.w * q.x),
    m[4] = 1 - s * (q2.y + q2.w), m[5] = s * (q.z * q.w - q.y * q.x),
    m[6] = s * (q.y * q.w - q.z * q.x), m[7] = s * (q.z * q.w + q.y * q.x),
    m[8] = 1 - s * (q2.y + q2.z);
}
// p: source point, m: rotate matrix(3x3), n: n-th dim after rotate, value [0,2]
__device__ __forceinline__ float rotate(float3 p, float *m, int n) {
    return p.x * m[n * 3 + 0] + p.y * m[n * 3 + 1] + p.z * m[n * 3 + 2];
}
// calc m1[m,n] @ m2[n,k]
// if trans, calc m1[m,n] @ m2[k,n].T
template <int m, int n, int k, bool trans>
__device__ __forceinline__ void matmul(float *m1, float *m2, float *out) {
    int s = 0;
    for (int r = 0; r < m; r++) {  // rows
        int row = r * m;
        for (int c = 0; c < k; c++) {  // columns
            float *rp, *cp;
            if (trans) {
                rp        = &m1[row];
                cp        = &m2[c * k];
                float sum = 0.;
                for (int i = 0; i < n; i++) {
                    sum += rp[i] * cp[i];
                }
                out[s++] = sum;

            } else {
                rp        = &m1[row];
                cp        = &m2[c];
                float sum = 0.;
                for (int i = 0; i < n; i++) {
                    sum += rp[i] * cp[i * n];
                }
                out[s++] = sum;
            }
        }
    }
}
template <bool trans>
__device__ __forceinline__ void matmul1(float *m1, float *m2, float *out, int m, int n, int k) {
    int s = 0;
    for (int r = 0; r < m; r++) {  // rows
        int row = r * m;
        for (int c = 0; c < k; c++) {  // columns
            float *rp, *cp;
            if (trans) {
                rp        = &m1[row];
                cp        = &m2[c * k];
                float sum = 0.;
                for (int i = 0; i < n; i++) {
                    sum += rp[i] * cp[i];
                }
                out[s++] = sum;

            } else {
                rp        = &m1[row];
                cp        = &m2[c];
                float sum = 0.;
                for (int i = 0; i < n; i++) {
                    sum += rp[i] * cp[i * n];
                }
                out[s++] = sum;
            }
        }
    }
}

#if 0
def gen_matrix_from_rot_vec(k, theta):
    K = torch.zeros((3, 3), device=k.device)
    K[[1, 2, 0], [2, 0, 1]] = -k
    K[[2, 0, 1], [1, 2, 0]] = k
    R = torch.eye(3) + K * torch.sin(theta) + (1 - torch.cos(theta)) * torch.matmul(K, K)
    return R
#endif
template <int n>
__device__ __forceinline__ void gen_matrix_from_rot_vec(float *k, float *out, float theta) {
    float sin = sinf(theta);
    float cos = cosf(theta);
    matmul<n, n, n, false>(k, k, out);
    int idx = 0;
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            out[idx] = out[idx] * cos + sin * k[idx] + (r == c ? 1. : 0.);
            idx++;
        }
    }
}
__device__ __forceinline__ void gen_matrix_from_rot_vec3(float *k, float *out, float theta) {
    float K[3][3] = { { 0., -k[2], k[1] }, { k[2], 0., -k[0] }, { -k[1], k[0], 0. } };
    gen_matrix_from_rot_vec<3>((float *)K, out, theta);
}
__device__ __forceinline__ void modify_conformer_torsion_angles_single(float *pos,
                                                                       float *out,
                                                                       uint8_t *mask,
                                                                       int u,
                                                                       int v,
                                                                       float theta,
                                                                       int n,
                                                                       float *tmp) {
    float *pu  = pos + 3 * u;
    float *pv  = pos + 3 * v;
    float *rot = tmp;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float rot_vec[3];
        sub3p(pu, pv, rot_vec);
        float norm = rnorm3df(rot_vec[0], rot_vec[1], rot_vec[2]);
        rot_vec[0] *= norm;
        rot_vec[1] *= norm;
        rot_vec[2] *= norm;
        gen_matrix_from_rot_vec3(rot_vec, rot, theta);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (mask[i]) {
            float tp[3];
            float *p    = pos + 3 * i;
            float *outp = out + 3 * i;
            sub3p(p, pv, tp);
            matmul<1, 3, 3, true>(tp, rot, outp);
            add3p(outp, pv);
        }
    }
}
// nedge: edge_index should be 2D array of <edge_size>*2
// pos: n x 3 2D array
// out: output, nx3
// mask_rotate: edge_size * n bool mask
// torsion_updates: 1D array, size <edge_size>
// require tmp: 3 * nedge floats
__device__ __forceinline__ void modify_conformer_torsion_angles(float *pos,
                                                                float *out,
                                                                float *tmp,
                                                                uint8_t *mask_rotate,
                                                                int *edge_index,
                                                                float *torsion_updates,
                                                                int n,
                                                                int nedge) {
    for (int idx_edge = threadIdx.y; idx_edge < nedge; idx_edge += blockDim.y) {
        int u = edge_index[idx_edge * 2];
        int v = edge_index[idx_edge * 2 + 1];
        modify_conformer_torsion_angles_single(
          pos, out, mask_rotate + n * idx_edge, u, v, torsion_updates[idx_edge], n, tmp + idx_edge);
    }
    __syncthreads();
}
#if 0
// pos: (.., 3)
// edge_index: torsion, 2xtupleof 2
// mask_rotate(2x20 bools)
// torsion_updates: shape (2)
def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates):
  // pos = copy.deepcopy(pos)

    for idx_edge, e in enumerate(edge_index):
      // if torsion_updates[idx_edge] == 0:
      //  continue
        u, v = e[0], e[1]

    //  check if need to reverse the edge, v should be connected to the part that gets rotated
    //  assert not mask_rotate[idx_edge, u]
    //  assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec / torch.norm(rot_vec)  # idx_edge!
        rot_mat = gen_matrix_from_rot_vec(rot_vec, torsion_updates[idx_edge])

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    return pos
#endif

template <typename T>
__device__ __forceinline__ T det3x3(T *v) {
    T a = v[0] * v[4] * v[8] + v[1] * v[5] * v[6] + v[2] * v[3] * v[7];
    T b = v[0] * v[5] * v[7] + v[1] * v[3] * v[8] + v[2] * v[4] * v[6];
    return a - b;
}
template <typename T>
__device__ __forceinline__ void trans(T *in, T *out, int m, int n) {
    for (int inr = 0; inr < m; inr++) {
        T *instart  = in + inr * n;
        T *outstart = out + inr;
        for (int inc = 0; inc < n; inc++, outstart += m) {
            *outstart = instart[inc];
        }
    }
}
#if 0
def rigid_transform_Kabsch_3D_torch(A, B):
// R = 3x3 rotation matrix, t = 3x1 column vector
// This already takes residue identity into account.

    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

// find mean column wise : 3 x 1
    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)

// subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

// find rotation
// shape U torch.Size([3, 3]) S torch.Size([3]) Vt torch.Size([3, 3])
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T
// special reflection case
    if torch.linalg.det(R) < 0:
// print("det(R) < R, reflection detected!, correcting for it ...")
        SS = torch.diag(torch.tensor([1., 1., -1.], device=A.device))
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(torch.linalg.det(R) - 1) < 3e-3  # note I had to change this error bound to be higher

    t = -R @ centroid_A + centroid_B
    return R, t
#endif
// a, b: nx3, input
// at, bt: 3xn, tmp mem
// r: 3x3 output
// t: 3x1 output
// n: num of input coords a and b
__device__ __forceinline__ void rigid_transform_Kabsch_3D_torch(float *a,
                                                                float *b,
                                                                float *at,
                                                                float *bt,
                                                                float *r,
                                                                float *t,
                                                                int n) {
    // mean of a, b
    float ca[3] = { 0., 0., 0. };
    float cb[3] = { 0., 0., 0. };
    float *tpa = a, *tpb = b;
    for (int i = 0; i < n * 3;) {
        ca[0] += *tpa++;
        ca[1] += *tpa++;
        ca[2] += *tpa++;
        cb[0] += *tpb++;
        cb[1] += *tpb++;
        cb[2] += *tpb++;
    }
    for (int i = 0; i < 3; i++) {
        float r = reciprocal((float)n);
        ca[i] *= r;
        cb[i] *= r;
    }
    // substract mean and turn nx3 -> 3xn
    for (int i = 0; i < 3; i++) {
        tpa = at + i * n, tpb = bt + i * n;
        for (int j = 0; j < n; j++) {
            // at[i, j] = a[j, i] - ca[i];
            tpa[j] = a[j * 3 + i] - ca[i];
            tpb[j] = b[j * 3 + i] - cb[i];
        }
    }

    float h[9];
    matmul1<true>(at, bt, h, 3, n, 3);

    float u[9], s[3], v[9], vt[9];
    svd(h, u, s, v);
    trans<float>(v, vt, 3, 3);

    matmul<3, 3, 3, true>(vt, u, r);

    float dt = det3x3<float>(r);
    if (dt < 0) {
        float ss[9] = { 1, 0, 0, 0, 1, 0, 0, 0, -1 };
        float st[9];
        matmul<3, 3, 3, false>(vt, ss, st);
        matmul<3, 3, 3, true>(st, u, r);
    }
    float r1[9];
    for (int i = 0; i < 9; i++) {
        r1[i] = -r[i];
    }
    matmul<3, 3, 1, false>(r1, ca, t);
    t[0] += cb[0];
    t[1] += cb[1];
    t[2] += cb[2];
}

#define PI 3.141592653589793

// pos: npos x 3
// newpos: npos x 3, output
// values: size nval
// edge_index: nedge x 2
// mask_rotate: npos
// tmp: tmp memory, at least 12 + npos * 3 * 3 + nedge * 3 floats
__device__ __forceinline__ void modify_conformer(float *pos,
                                                 float *newpos,
                                                 float *values,
                                                 int *edge_index,
                                                 uint8_t *mask_rotate,
                                                 int npos,
                                                 int nval,
                                                 int nedge,
                                                 float *tmp) {
    float *center, *rot_mat, *tr, *rot;
    center         = tmp, tmp += 3;
    rot_mat        = tmp, tmp += 9;
    tr             = tmp, tmp += 3;
    rot            = tmp, tmp += 3;
    bool exec_main = threadIdx.x == 0 && threadIdx.y == 0;

    int start = threadIdx.x, block = blockDim.x;
    bool run = threadIdx.y == 0;
    if (blockDim.y > blockDim.x) {
        start = threadIdx.y, block = blockDim.y;
        bool run = threadIdx.x == 0;
    }

    for (int i = start; i < 3 && run; i += block) {
        tr[i]  = (SigmoidForward<float>(values[i]) - 0.5) * 10;
        rot[i] = (SigmoidForward<float>(values[3 + i]) - 0.5) * 2 * PI;
    }
    __syncthreads();

    if (exec_main) {
        float3 rot1 = make_float3(rot[0], rot[1], rot[2]);
        float4 qu   = axis_angle_to_quaternion(rot1);
        quaternion_to_matrix(qu, rot_mat);
        for (int i = 0; i < npos; i++) {
            center[0] += pos[i * 3];
            center[1] += pos[i * 3 + 1];
            center[2] += pos[i * 3 + 2];
        }
        float renpos = reciprocal(((float)npos));
        center[0] *= renpos, center[1] *= renpos, center[2] *= renpos;
        tr[0] += center[0], tr[1] += center[1], tr[2] += center[2];
    }
    // todo: should we sync?
    __syncthreads();

    // calc new pos
    for (int n = start; n < npos && run; n += block) {
        float *p    = pos + 3 * n;
        float *newp = newpos + 3 * n;
        float np[3];
        sub3p(p, center, np);
        float rotp[3];
        matmul<1, 3, 3, true>(np, rot_mat, rotp);
        add3p(rotp, tr, newp);
    }
    __syncthreads();

    if (nval > 6) {
        float *flexpos, *ctmp, *at, *bt, *r, *t;
        flexpos = tmp, tmp += npos * 3;
        ctmp    = tmp, tmp += nedge * 3;
        at      = tmp, tmp += npos * 3;
        bt      = tmp, tmp += npos * 3;
        r       = tmp, tmp += 9;
        t       = tmp, tmp += 3;
        // require nedge * 3 floats
        modify_conformer_torsion_angles(
          newpos, flexpos, ctmp, mask_rotate, edge_index, values + 6, npos, nedge);
        if (exec_main) {
            rigid_transform_Kabsch_3D_torch(flexpos, newpos, at, bt, r, t, npos);
        }
        __syncthreads();
        for (int n = start; n < npos; n += block) {
            float *p    = flexpos + n * 3;
            float *outp = newpos + n * 3;
            matmul<1, 3, 3, true>(p, r, outp);
            add3p(outp, t);
        }
    }
}
#if 0
// values : 8
// edge_index : torsion, 2xtupleof 2
// mask_rotate(2x20 bools)
def modify_conformer(coords, values, edge_index, mask_rotate):
    tr_update = (torch.sigmoid(values[:3]) - 0.5) * 10
    rot_update = (torch.sigmoid(values[3:6]) - 0.5) * np.pi * 2
//  torsion_updates = (torch.sigmoid(values [6:]) - 0.5) * np.pi * 2
//  rot_update      = values[3 : 6]
    torsion_updates = values[6:]

    lig_center = torch.mean(coords, dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    rigid_new_pos = (coords - lig_center) @ rot_mat.T + tr_update + lig_center

    if values.shape[0] > 6:
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,
                                                           edge_index,
                                                           mask_rotate,
                                                           torsion_updates).to(rigid_new_pos.device)
        try:
            R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
            aligned_flexible_pos = flexible_new_pos @ R.T + t.T
            return aligned_flexible_pos
        except:
            return flexible_new_pos
    else:
        return rigid_new_pos
#endif

// rebeta = 1/beta
__device__ __forceinline__ float smooth_l1_loss(float a, float b, float beta, float rebeta) {
    float z = abs(a - b);
    return z < beta ? 0.5 * z * z * rebeta : z - 0.5 * beta;
}
// a, b: array of m x n
// beta: smooth beta
// tmp: at least m x n floats required
// flags, nullptr or flag indicates if corresponding a and b should be calculated
// mean: true for smooth with mean, or false for sum
template <bool mean>
__device__ __forceinline__ void smooth_l1_loss(float *a,
                                               float *b,
                                               float *tmp,
                                               float beta,
                                               int m,
                                               int n,
                                               uint8_t *flags,
                                               float *out) {
    float rebeta = reciprocal(beta);
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        for (int j = threadIdx.y; j < n; j += blockDim.y) {
            int seq = i * m + j;
            if (flags == nullptr || flags[seq] != 0) {
                tmp[seq] = smooth_l1_loss(a[seq], b[seq], beta, rebeta);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0;
        int total = 0;
        for (int i = 0; i < m * n; i++) {
            sum += tmp[i];
            if (flags == nullptr || flags[i] != 0) {
                total++;
            }
        }
        if (mean) {
            sum *= reciprocal(total);
        }
        *out = sum;
    }
}

#if 0
def single_SF_loss(
        predict_coords,
        pocket_coords,
        distance_predict,
        holo_distance_predict,
        dist_threshold=6,
// dist_threshold = 10,

):
// dist           = dist.unsqueeze(0)
    dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    holo_dist = torch.norm(
        predict_coords.unsqueeze(1) - predict_coords.unsqueeze(0), dim=-1
    )
    distance_mask = distance_predict < dist_threshold
    cross_dist_score = F.smooth_l1_loss(distance_predict[distance_mask], dist[distance_mask])
    dist_score = F.smooth_l1_loss(holo_distance_predict, holo_dist)
    loss = cross_dist_score * 1.0 + dist_score * 5.0
    return loss
#endif
// require tmp npred *(npocket + npred + max(npred, npocket) + 2) * float + npred*npocket
__device__ __forceinline__ void single_SF_loss(float *predict,            // npredx3
                                               float *pocket,             // npocketx3
                                               float *dist_predict,       // npred x npocket
                                               float *holo_dist_predict,  // npred x npred
                                               float dist_threshold,
                                               int npred,
                                               int npocket,
                                               float *tmp,
                                               float *out) {
    int ncross = npred * npocket;
    float *dist, *holo_dist, *cross_dist_score, *dist_score;
    dist             = tmp, tmp += ncross;
    holo_dist        = tmp, tmp += npred * npred;
    cross_dist_score = tmp, tmp++;
    dist_score       = tmp, tmp++;

    for (int i = threadIdx.x; i < npred; i += blockDim.x) {
        float *p1 = predict + 3 * i;
        for (int j = threadIdx.y; j < npocket; j += blockDim.y) {
            float *p2           = pocket + 3 * j;
            dist[i * npred + j] = norm3df(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
        }
        for (int j = threadIdx.y; j < npred; j += blockDim.y) {
            float *p2                = predict + 3 * j;
            holo_dist[i * npred + j] = norm3df(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
        }
    }
    __syncthreads();
    uint8_t *flags = (uint8_t *)(tmp + max(ncross, npred * npred));
    for (int i = threadIdx.x; i < npred; i += blockDim.x) {
        for (int j = threadIdx.y; j < npocket; j += blockDim.y) {
            int seq    = i * npred + j;
            flags[seq] = dist_predict[seq] < dist_threshold ? 1 : 0;
        }
    }
    smooth_l1_loss<true>(dist_predict, dist, tmp, 0.5, npred, npocket, flags, cross_dist_score);
    smooth_l1_loss<true>(holo_dist_predict, holo_dist, tmp, 0.5, npred, npred, nullptr, dist_score);

    if (threadIdx.x == 0) {
        *out = *cross_dist_score + *dist_score * 5.0;
    }
}

__global__ void dock_kernel(float *init_coord,
                            float *pocket,
                            float *pred_cross_dist,
                            float *pred_holo_dist,
                            float *values,
                            int *torsions,
                            uint8_t *masks,
                            int npred,
                            int npocket,
                            int nval,
                            int ntorsion,
                            float *loss) {
    extern __shared__ float sm[];
    float *new_pos = sm;  // require float * npred * 3
    float *tmp     = &sm[npred * 3];
    // require: tmp memory, at least 12 + npos * 3 * 3 + nedge * 3 floats
    modify_conformer(init_coord, new_pos, values, torsions, masks, npred, nval, ntorsion, tmp);
    // require tmp npred *(npocket + npred + max(npred, npocket)+ 2) * float + npred*npocket
    single_SF_loss(new_pos, pocket, pred_cross_dist, pred_holo_dist, 6, npred, npocket, tmp, loss);
}

void dock_gpu(float *init_coord,
              float *pocket,
              float *pred_cross_dist,
              float *pred_holo_dist,
              float *values,
              int *torsions,
              uint8_t *masks,
              int npred,
              int npocket,
              int nval,
              int ntorsion,
              float *loss,
              cudaStream_t stream) {
    int smsize = npred * 3 * sizeof(float);
    int extra  = std::max(
      (npred * 9 + ntorsion * 3 + 12) * sizeof(float),
      npred * (npocket + npred + std::max(npred, npocket) + 2) * sizeof(float) + npred * npocket);
    smsize += extra;
    dim3 block(npred, npocket, 1);
    dim3 grid(1, 1, 1);
    dock_kernel<<<grid, block, smsize, stream>>>(init_coord,
                                                 pocket,
                                                 pred_cross_dist,
                                                 pred_holo_dist,
                                                 values,
                                                 torsions,
                                                 masks,
                                                 npred,
                                                 npocket,
                                                 nval,
                                                 ntorsion,
                                                 loss);
}
}  // namespace dock