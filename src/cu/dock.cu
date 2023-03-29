#include <algorithm>
#include <iostream>
#include <assert.h>

#include <cuda_runtime.h>

#include "math.h"
#define REST __restrict__
#define IN
#define OUT
namespace dock {
#define DOCKDBG 0
#define GRIDIDX 0
#define INGRID  (blockIdx.x == GRIDIDX)
#if DOCKDBG
# define SDBG(x1, y1, ...) \
  do { \
   if (INGRID && threadIdx.x == x1 && threadIdx.y == y1) { \
    printf(__VA_ARGS__); \
   } \
  } while (0)
# define DBG(...) \
  do { \
   if (INGRID) { \
    printf(__VA_ARGS__); \
   } \
  } while (0)
#else
#define SDBG(x1, y1, ...) 
#define DBG(...)
#endif

#include "svd3_cuda.h"
// calc 1/x in round-to-nearest-even mode
#define reciprocal(x) __frcp_rn(x)
#define FOR_LOOP(i, end) for (int i = threadIdx.x; i < end; i += blockDim.x)
#define IS_THREAD(n)     (threadIdx.x == (n))
#define IS_MAIN_THREAD() IS_THREAD(0)
#define PI               3.141592653589793

__device__ void dumparr(int m, int n, const uint8_t * REST p) {
    printf(" ====== t[%d,%d] (%dx%d) ======\n", threadIdx.x, threadIdx.y, m, n);
    for (int i = 0; i < m; i++) {
        printf("%d: ", i);
        const uint8_t *p1 = p + i * n;
        for (int j = 0; j < n; j++) {
            printf("%d ", int(p1[j]));
        }
        printf("\n");
    }
}
__device__ void dumparr(int m, int n, const float * REST p) {
    printf(" ====== t[%d,%d] (%dx%d) ======\n", threadIdx.x, threadIdx.y, m, n);
    for (int i = 0; i < m; i++) {
        printf("%d: ", i);
        const float *p1 = p + i * n;
        for (int j = 0; j < n; j++) {
            printf("%f ", p1[j]);
        }
        printf("\n");
    }
}
#if DOCKDBG
# define DUMP(hdr, m, n, p) \
  do { \
   if (INGRID) { \
    printf(hdr); \
    dumparr(m, n, p); \
   } \
  } while (0)

# define DUMPARR(x1, y1, hdr, m, n, p) \
  do { \
   if (INGRID && threadIdx.x == x1 && threadIdx.y == y1) { \
    printf(hdr); \
    dumparr(m, n, p); \
   } \
  } while (0)
#else
#define DUMP(hdr, m, n, p) 
#define DUMPARR(x1, y1, hdr, m, n, p) 
#endif
# define DUMPARR1(hdr, m, n, p) \
  do { \
   if (blockIdx.x == 3 && threadIdx.x == 0 && threadIdx.y == 0) { \
    printf(hdr); \
    dumparr(m, n, p); \
   } \
  } while (0)

__device__ __forceinline__ float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ void sub3p(const float * REST a, const float * REST b, float * REST out) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}
__device__ __forceinline__ void add3p(const float * REST a, const float * REST b, float * REST out) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
}
__device__ __forceinline__ void sub3p(IN OUT float * REST a, const float * REST b) {
    a[0] = a[0] - b[0];
    a[1] = a[1] - b[1];
    a[2] = a[2] - b[2];
}
__device__ __forceinline__ void add3p(IN OUT float * REST a, const float * REST b) {
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
__device__ __forceinline__ float Norm2(const float * REST d) {
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
    DBG("axis %f %f %f angle %f half %f\n", axis.x, axis.y, axis.z, angle, half);
    float sins;
    if (angle < 1e-6) {
        sins = 0.5f - (angle * angle) * 0.020833333333333332;  // 1/48 = 0.020833333333333332
    } else {
        sins = sinf(half) * reciprocal(angle);
    }
    DBG("sins %f \n", sins);
    float4 ret;
    ret.x = cosf(half);
    ret.y = axis.x * sins;
    ret.z = axis.y * sins;
    ret.w = axis.z * sins;
    DBG("qu %f %f %f %f\n", ret.x, ret.y, ret.z, ret.w);
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
__device__ __forceinline__ void quaternion_to_matrix(float4 q, OUT float * REST m) {
    float4 q2 = make_float4(q.x * q.x, q.y * q.y, q.z * q.z, q.w * q.w);
    float s   = reciprocal(q2.x + q2.y + q2.z + q2.w) * 2.0;
    m[0] = 1 - s * (q2.z + q2.w), m[1] = s * (q.y * q.z - q.w * q.x),
    m[2] = s * (q.y * q.w + q.z * q.x), m[3] = s * (q.y * q.z + q.w * q.x),
    m[4] = 1 - s * (q2.y + q2.w), m[5] = s * (q.z * q.w - q.y * q.x),
    m[6] = s * (q.y * q.w - q.z * q.x), m[7] = s * (q.z * q.w + q.y * q.x),
    m[8] = 1 - s * (q2.y + q2.z);
}
// p: source point, m: rotate matrix(3x3), n: n-th dim after rotate, value [0,2]
__device__ __forceinline__ float rotate(float3 p, const float * REST m, int n) {
    return p.x * m[n * 3 + 0] + p.y * m[n * 3 + 1] + p.z * m[n * 3 + 2];
}
// calc m1[m,n] @ m2[n,k]
// if trans, calc m1[m,n] @ m2[k,n].T
template <int m, int n, int k, bool trans>
__device__ __forceinline__ void matmul(const float * REST m1, const float * REST m2, float * REST out) {
    int s = 0;
    for (int r = 0; r < m; r++) {  // rows
        int row = r * n;
        for (int c = 0; c < k; c++) {  // columns
            const float *rp, *cp;
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
                    sum += rp[i] * cp[i * k];
                }
                out[s++] = sum;
            }
        }
    }
}
template <bool trans>
__device__ __forceinline__ void matmul1(const float * REST m1, const float * REST m2, float * REST out, int m, int n, int k) {
    int s = 0;
    for (int r = 0; r < m; r++) {  // rows
        int row = r * n;
        for (int c = 0; c < k; c++) {  // columns
            const float *rp, *cp;
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
                    sum += rp[i] * cp[i * k];
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
__device__ __forceinline__ void gen_matrix_from_rot_vec(const float * REST k, float * REST out, float theta) {
    float sin = sinf(theta);
    float cos = 1.0 - cosf(theta);
    DUMPARR(1, 0, "K", 3, 3, k);
    matmul<n, n, n, false>(k, k, out);
    DUMPARR(1, 0, "sin", 1, 1, &sin);
    DUMPARR(1, 0, "cos", 1, 1, &cos);
    DUMPARR(1, 0, "KK", 3, 3, out);
    int idx = 0;
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            out[idx] = out[idx] * cos + sin * k[idx] + (r == c ? 1. : 0.);
            idx++;
        }
    }
    DUMPARR(1, 0, "R", 3, 3, out);
}
__device__ __forceinline__ void gen_matrix_from_rot_vec3(const float * REST k, float * REST out, float theta) {
    float K[3][3] = { { 0., -k[2], k[1] }, { k[2], 0., -k[0] }, { -k[1], k[0], 0. } };
    gen_matrix_from_rot_vec<3>((float *)K, out, theta);
}
#if 0
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
    if (threadIdx.x == 0) {
        float rot_vec[3];
        sub3p(pu, pv, rot_vec);
        DUMP("rot vec", 1, 3, rot_vec);
        float norm = rnorm3df(rot_vec[0], rot_vec[1], rot_vec[2]);
        rot_vec[0] *= norm;
        rot_vec[1] *= norm;
        rot_vec[2] *= norm;
        DUMP("rot vec norm", 1, 3, rot_vec);
        gen_matrix_from_rot_vec3(rot_vec, rot, theta);
        DUMP("rot mat", 3, 3, rot);
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
#endif
// nedge: edge_index should be 2D array of <edge_size>*2
// pos: n x 3 2D array
// out: output, nx3
// mask_rotate: edge_size * n bool mask
// torsion_updates: 1D array, size <edge_size>
// require tmp: 9 * nedge floats
__device__ __forceinline__ void modify_conformer_torsion_angles_concurrent(
    const float *REST pos, OUT float *REST out, float *REST tmp, const uint8_t *REST mask_rotate,
    const int *REST edge_index, const float *REST torsion_updates, int n, int nedge) {
    float *rot_mats = tmp;
    tmp += nedge * 9;
    FOR_LOOP(i, n) {
        int seq  = i * 3;
        out[seq] = pos[seq], out[seq + 1] = pos[seq + 1], out[seq + 2] = pos[seq + 2];
        if (i < nedge) {
            int idx_edge = i;
            int u        = edge_index[idx_edge * 2];
            int v        = edge_index[idx_edge * 2 + 1];

            float theta   = torsion_updates[idx_edge];
            const float *pu     = pos + 3 * u;
            const float *pv     = pos + 3 * v;
            float *rot    = rot_mats + idx_edge * 9;
            float rot_vec[3];
            sub3p(pu, pv, rot_vec);
            DUMPARR(1, 0, "rot vec", 1, 3, rot_vec);
            float norm = rnorm3df(rot_vec[0], rot_vec[1], rot_vec[2]);
            rot_vec[0] *= norm, rot_vec[1] *= norm, rot_vec[2] *= norm;
            DUMPARR(1, 0, "rot vec norm", 1, 3, rot_vec);
            gen_matrix_from_rot_vec3(rot_vec, rot, theta);
            DUMPARR(1, 0, "rot mat", 3, 3, rot);
        }
    }

    __syncthreads();
    DUMPARR(0, 0, "mask 0", 1, n, mask_rotate);
    DUMPARR(0, 0, "mask 1", 1, n, mask_rotate + n);
    FOR_LOOP(i, n) {
        const float *p    = pos + 3 * i;
        float *outp = out + 3 * i;
        for (int k = 0; k < nedge; k++) {
            const uint8_t *mask = mask_rotate + n * k;
            if (mask[i] != 0) {
                int u      = edge_index[k * 2];
                int v      = edge_index[k * 2 + 1];
                const float *pu  = pos + 3 * u;
                const float *pv  = pos + 3 * v;
                const float *rot = rot_mats + k * 9;
                float tp[3];
                sub3p(p, pv, tp);
                matmul<1, 3, 3, true>(tp, rot, outp);
                add3p(outp, pv);
            }
        }
    }
    __syncthreads();
    DUMPARR(0, 0, "modify conformer ret pos", n, 3, out);
}
__device__ __forceinline__ void modify_conformer_torsion_angles(
    const float *REST pos, OUT float *REST out, float *REST tmp, const uint8_t *REST mask_rotate,
    const int *REST edge_index, const float *REST torsion_updates, int n, int nedge) {
    float *rot_mats = tmp;
    tmp += nedge * 9;
    // copy to output
    FOR_LOOP(i, n * 3) {
        out[i] = pos[i];
    }
    __syncthreads();
    if (IS_MAIN_THREAD()) {
        for (int k = 0; k < nedge; k++ ) {
            int idx_edge = k;
            int u        = edge_index[idx_edge * 2];
            int v        = edge_index[idx_edge * 2 + 1];

            float theta     = torsion_updates[idx_edge];
            const float *pu = pos + 3 * u;
            const float *pv = pos + 3 * v;
            float rot_vec[3];
            float rot[9];
            sub3p(pu, pv, rot_vec);
            DUMP("rot vec", 1, 3, rot_vec);
            float norm = rnorm3df(rot_vec[0], rot_vec[1], rot_vec[2]);
            rot_vec[0] *= norm, rot_vec[1] *= norm, rot_vec[2] *= norm;
            DUMP("rot vec norm", 1, 3, rot_vec);
            gen_matrix_from_rot_vec3(rot_vec, rot, theta);
            DUMP("rot mat", 3, 3, rot);
            const uint8_t *mask = mask_rotate + n * k;
            for (int i = 0; i < n; i++) {
                if (mask[i] != 0) {
                    float *outp = out + 3 * i;
                    float tp[3];
                    sub3p(outp, pv, tp);
                    matmul<1, 3, 3, true>(tp, rot, outp);
                    add3p(outp, pv);
                }
            }
        }
    }

    __syncthreads();
    DUMPARR(0, 0, "modify conformer ret pos", n, 3, out);
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
__device__ __forceinline__ T det3x3(const T * REST v) {
    T a = v[0] * v[4] * v[8] + v[1] * v[5] * v[6] + v[2] * v[3] * v[7];
    T b = v[0] * v[5] * v[7] + v[1] * v[3] * v[8] + v[2] * v[4] * v[6];
    return a - b;
}
template <typename T>
__device__ __forceinline__ void trans(const T * REST in, OUT T * REST out, int m, int n) {
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
// called in main thread, require memory 2xnx3
__device__ __forceinline__ void rigid_transform_Kabsch_3D_torch(const float *REST a,
                                                                const float *REST b,
                                                                float *REST tmp, OUT float *REST r,
                                                                OUT float *REST t, int n) {
    // mean of a, b

    float *at, *bt;
    at = tmp, tmp += n * 3;
    bt = tmp, tmp += n * 3;
    DUMP("A", n, 3, a);
    DUMP("B", n, 3, b);
    float ca[3] = { 0., 0., 0. };
    float cb[3] = { 0., 0., 0. };
    const float * REST tpa = a;
    const float *REST tpb = b;
    for (int i = 0; i < n * 3; i += 3) {
        add3p(ca, tpa + i);
        add3p(cb, tpb + i);
    }
    float re = reciprocal((float)n);
    for (int i = 0; i < 3; i++) {
        ca[i] *= re;
        cb[i] *= re;
    }
    DUMP("Center A", 1, 3, ca);
    DUMP("Center B", 1, 3, cb);
    // substract mean and turn nx3 -> 3xn
    for (int i = 0; i < 3; i++) {
        float * REST tp = at + i * n;
        for (int j = 0; j < n; j++) {
            // at[i, j] = a[j, i] - ca[i];
            tp[j] = a[j * 3 + i] - ca[i];
        }
    }
    for (int i = 0; i < n; i++) {
        sub3p(b + i * 3, cb, bt + i * 3);
    }
    DUMP("Am", 3, n, at);
    DUMP("Bm", n, 3, bt);

    float h[9];
    matmul1<false>(at, bt, h, 3, n, 3);

    float u[9], s[3], v[9], vt[9];
    DUMP("H", 3, 3, h);
    svd(h, u, s, v);
    DUMP("U", 3, 3, u);
    DUMP("S", 1, 3, s);
    DUMP("V", 3, 3, v);
    // svd already output torch svd Vt.T
    // trans<float>(v, v, 3, 3);
    DUMP("Vt", 3, 3, v);

    matmul<3, 3, 3, true>(v, u, r);
    DUMP("r", 3, 3, r);

    float dt = det3x3<float>(r);
    DUMP("det", 1, 1, &dt);
    if (dt < 0) {
        float ss[9] = { 1, 0, 0, 0, 1, 0, 0, 0, -1 };
        float st[9];
        matmul<3, 3, 3, false>(v, ss, st);
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
    DUMP("R", 3, 3, r);
    DUMP("t", 3, 3, t);
}

// pos: npos x 3
// newpos: npos x 3, output
// values: size nval
// edge_index: nedge x 2
// mask_rotate: npos
// tmp: tmp memory, 18 + max(9*nedge, 6 * npos) floats
__device__ __forceinline__ void modify_conformer(const float *REST pos, OUT float *REST newpos,
                                                 const float *REST values,
                                                 const int *REST edge_index,
                                                 const uint8_t *REST mask_rotate, int npos,
                                                 int nval, int nedge, float *REST tmp) {
    float *center, *rot_mat, *tr, *rot;
    center         = tmp, tmp += 3;
    rot_mat        = tmp, tmp += 9;
    tr             = tmp, tmp += 3;
    rot            = tmp, tmp += 3;
    DUMPARR(0, 0, "coords", npos, 3, pos);
    DUMPARR(0, 0, "values", 1, nval, values);

    FOR_LOOP(i, 3) {
        tr[i]  = (SigmoidForward<float>(values[i]) - 0.5) * 10;
        rot[i] = (SigmoidForward<float>(values[3 + i]) - 0.5) * 2 * PI;
    }
    __syncthreads();
    DUMPARR(0, 0, "tr update", 1, 3, tr);
    DUMPARR(0, 0, "rot update", 1, 3, rot);

    if (IS_MAIN_THREAD()) {
        float3 rot1 = make_float3(rot[0], rot[1], rot[2]);
        float4 qu   = axis_angle_to_quaternion(rot1);
        quaternion_to_matrix(qu, rot_mat);
        DUMPARR(0, 0, "rot mat", 3, 3, rot_mat);
        center[0] = center[1] = center[2] = 0;
        for (int i = 0; i < npos; i++) {
            center[0] += pos[i * 3];
            center[1] += pos[i * 3 + 1];
            center[2] += pos[i * 3 + 2];
        }
        float renpos = reciprocal(((float)npos));
        center[0] *= renpos, center[1] *= renpos, center[2] *= renpos;
        tr[0] += center[0], tr[1] += center[1], tr[2] += center[2];
        DUMPARR(0, 0, "center", 1, 3, center);
        DUMPARR(0, 0, "tr update + center", 1, 3, tr);
    }
    // todo: should we sync?
    __syncthreads();

    // calc new pos
    FOR_LOOP(n, npos) {
        const float *p    = pos + 3 * n;
        float *newp = newpos + 3 * n;
        float np[3];
        sub3p(p, center, np);
        float rotp[3];
        matmul<1, 3, 3, true>(np, rot_mat, rotp);
        add3p(rotp, tr, newp);
    }
    __syncthreads();
    DUMPARR(0, 0, "new pos", npos, 3, newpos);

    // require max(9*nedge, 6 * npos)
    if (nval > 6) {
        float *flexpos, *ctmp, *r, *t;
        flexpos = tmp, tmp += npos * 3;
        r       = tmp, tmp += 9;
        t       = tmp, tmp += 3;
        // require nedge * 9 floats
        modify_conformer_torsion_angles(
          newpos, flexpos, tmp, mask_rotate, edge_index, values + 6, npos, nedge);
        if (IS_MAIN_THREAD()) {
            // require 2 x npos x 3 floats
            rigid_transform_Kabsch_3D_torch(flexpos, newpos,  tmp, r, t, npos);
            matmul1<true>(flexpos, r, newpos, npos, 3, 3);
            FOR_LOOP(n, npos) {
                float *outp = newpos + n * 3;
                add3p(outp, t);
            }
            DUMP("aligned", npos, 3, newpos);
        }
        __syncthreads();
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
__device__ __forceinline__ void smooth_l1_loss(const float *REST a, const float *REST b,
                                               float *REST tmp, float beta, int m, int n,
                                               const uint8_t *REST flags, float *REST out) {
    float rebeta = reciprocal(beta);
    FOR_LOOP(i, m * n) {
        if (flags == nullptr || flags[i] != 0) {
            tmp[i] = smooth_l1_loss(a[i], b[i], beta, rebeta);
            // DBG("%d: %f %f beta %f %f -> %f\n", i, a[i], b[i], beta, rebeta, tmp[i]);
        }
    }
    __syncthreads();
    if (IS_MAIN_THREAD()) {
        float sum = 0;
        int total = 0;
        // DUMP("smooth a", m, n, a);
        // DUMP("smooth b", m, n, b);
        // DUMP("smooth tmp", m, n, tmp);
        // if (flags) {
        //     DUMPARR(0, 0, "smooth sum", m, n, flags);
        // }
        for (int i = 0; i < m * n; i++) {
            if (flags == nullptr || flags[i] != 0) {
                total++;
                sum += tmp[i];
            }
        }
        DBG("sum %f total %d mean %f\n", sum, total, sum / total);
        if (mean) {
            // todo: total == 0
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
// require tmp :
//   npred * ncross + npred * npred + 2 + ((npred * ncross + 3) >> 2)
//    + npred * max(npred, ncross)
__device__ __forceinline__ void single_SF_loss(
    const float *REST predict,            // npredx3
    const float *REST pocket,             // npocketx3
    const float *REST dist_predict,       // npred x npocket
    const float *REST holo_dist_predict,  // npred x npred
    float dist_threshold, int npred, int npocket, float *REST tmp, float *REST out) {
    int ncross = npred * npocket;
    int nsq    = npred * npred;
    float *dist, *holo_dist, *cross_dist_score, *dist_score;
    uint8_t *flags;
    // tmp mem: ncross + nsq + 2 + ((ncross + 3) >> 2)
    dist             = tmp, tmp += ncross;
    holo_dist        = tmp, tmp += nsq;
    cross_dist_score = tmp, tmp++;
    dist_score       = tmp, tmp++;
    flags            = (uint8_t *)tmp, tmp += ((ncross + 3) >> 2);

    float divpocket = reciprocal((float)npocket);
    float divpred = reciprocal((float)npred);
    FOR_LOOP(i, ncross) {
        // i / npred
        int idxpred = floor(i * divpocket);
        int idxpocket  = i - idxpred * npocket;
        const float *p1   = predict + 3 * idxpred;
        const float *p2   = pocket + 3 * idxpocket;
        dist[i]     = norm3df(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
        flags[i]       = dist_predict[i] < dist_threshold ? 1 : 0;
    }
    FOR_LOOP(i, nsq) {
        // i / npred
        int idxpred = floor(i * divpred);
        int remain  = i - idxpred * npred;
        if (remain < npred) {
            const float *p1   = predict + 3 * idxpred;
            const float *p2   = predict + 3 * remain;
            holo_dist[i] = norm3df(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
        }
    }
    __syncthreads();
    // DUMPARR(0, 0, "dist", npred, ncross, dist);

    // require npred x npocket floats
    smooth_l1_loss<true>(dist_predict, dist, tmp, 1.0, npred, npocket, flags, cross_dist_score);
    // require npred x npred floats
    smooth_l1_loss<true>(holo_dist_predict, holo_dist, tmp, 1.0, npred, npred, nullptr, dist_score);

    __syncthreads();

    if (IS_MAIN_THREAD()) {
        *out = *cross_dist_score + *dist_score * 1.0;
        DUMPARR(0, 0, "cross dist score", 1, 1, cross_dist_score);
        DUMPARR(0, 0, "dist score", 1, 1, dist_score);
        DUMPARR(0, 0, "loss", 1, 1, out);
        // printf("grid %d loss %p %f\n", blockIdx.x, out, *out);
        // printf("cross dist score %f dist score %f loss %f\n", *cross_dist_score, *dist_score, *out);
    }
    // if (blockIdx.x == 3 && threadIdx.x == 0) {
    //     printf("cross dist score %f dist score %f loss %f\n", *cross_dist_score, *dist_score, *out);
    // }
}
// mem req:
//    npred * 3 + max(
//          18 + max(9*nedge, 6 * npos),
//          npred * npocket + npred * npred + 2 + ((npred * npocket + 3) >> 2) + npred * max(npred, npocket)
//    )
__global__ void dock_kernel(const float *REST init_coord, const float *REST pocket,
                            const float *REST pred_cross_dist, const float *REST pred_holo_dist,
                            const float *REST values, const int *REST torsions,
                            const uint8_t *REST masks, int npred, int npocket, int nval,
                            int ntorsion, OUT float *REST loss, float *REST dev) {
    extern __shared__ float sm[];

    float *new_pos, *tmp;
    if (dev == nullptr) {
        new_pos = sm;  // require float * npred * 3
        tmp     = &sm[npred * 3];
    } else {
        new_pos = dev;  // require float * npred * 3
        tmp     = &dev[npred * 3];
    }

    // require 18 + max(9*nedge, 6 * npos) floats
    modify_conformer(init_coord, new_pos, values, torsions, masks, npred, nval, ntorsion, tmp);
    // require tmp npred *(npocket + npred + max(npred, npocket)+ 2) * float + npred*npocket
    // require tmp :
    //   npred * npocket + npred * npred + 2 + ((npred * npocket + 3) >> 2)
    //    + npred * max(npred, npocket)
    single_SF_loss(new_pos, pocket, pred_cross_dist, pred_holo_dist, 6, npred, npocket, tmp, loss);
}
__device__ __forceinline__ void sm_copy(float *dst, float *src, int sz) {
}

__global__ void dock_grad_kernel(const float *REST init_coord, const float *REST pocket,
                                 const float *REST pred_cross_dist,
                                 const float *REST pred_holo_dist, const float *REST values,
                                 const int *REST torsions, const uint8_t *REST masks, int npred,
                                 int npocket, int nval, int ngval, int ntorsion,
                                 OUT float *REST loss, float *REST dev, int blksz /*in floats*/) {
#if 1
    const float *sm_init_coord = init_coord, *sm_pocket = pocket, *sm_pred_cross_dist = pred_cross_dist, *sm_pred_holo_dist = pred_holo_dist;
    const int  *sm_torsions = torsions;
    const uint8_t *sm_masks = masks;
#else // this tries to use sm to speed up, but there is a risk that sm might not be enough, so disable it
    extern __shared__ float sm[];
    float *sm_init_coord, *sm_pocket, *sm_pred_cross_dist, *sm_pred_holo_dist;
    int  *sm_torsions;
    uint8_t *sm_masks;
    int n = 0, sz = 0;

    sz = npred * 3, sm_init_coord = &sm[n], n += sz;
    FOR_LOOP(i, sz) {
        sm_init_coord[i] = init_coord[i];
    }
    sz = npocket * 3, sm_pocket = &sm[n], n += sz;
    FOR_LOOP(i, sz) {
        sm_pocket[i] = pocket[i];
    }
    sz = npred * npocket, sm_pred_cross_dist = &sm[n], n += sz;
    FOR_LOOP(i, sz) {
        sm_pred_cross_dist[i] = pred_cross_dist[i];
    }
    sz = npred * npred, sm_pred_holo_dist = &sm[n], n += sz;
    FOR_LOOP(i, sz) {
        sm_pred_holo_dist[i] = pred_holo_dist[i];
    }
    sz = ntorsion * 2, sm_torsions = (int *)&sm[n], n += sz;
    FOR_LOOP(i, sz) {
        sm_torsions[i] = torsions[i];
    }
    int msz = (npred * ntorsion + 3) >> 2; // how many 4 bytes in masks
    int *dmasks, *smasks;
    sz = msz, smasks = (int *)masks, dmasks = (int *)&sm[n], n += sz;
    FOR_LOOP(i, sz) {
        dmasks[i] = smasks[i];
    }
    sm_masks = (uint8_t *)dmasks;
    __syncthreads();
#endif

    #if 0 // debug: all runs in block #0, to avoid concurrency between blocks
    if (blockIdx.x == 0) {
        for (int group = 0; group < ngval; group++) {
        float *new_pos, *tmp;
        if (dev == nullptr) {
            tmp = &sm[blksz * group];  // require float * npred * 3
        } else {
            tmp = dev + blksz * group;  // require float * npred * 3
        }
        new_pos = tmp, tmp += npred * 3;

        const float *vals = values + group * nval;
        if (group == 0) {
            DUMPARR(0, 0, "input masks", ntorsion, npred, masks);
        }
        modify_conformer(sm_init_coord, new_pos, vals, sm_torsions, sm_masks, npred, nval, ntorsion, tmp);
        single_SF_loss(
          new_pos, sm_pocket, sm_pred_cross_dist, sm_pred_holo_dist, 6, npred, npocket, tmp, loss + group);
        // DUMPARR1("loss", 1, nval+1, loss+group);
    }}
    #else
    int group      = blockIdx.x;
    if (group < ngval) {
        float *new_pos, *tmp;
        tmp = dev + blksz * group; 
        new_pos = tmp, tmp += npred * 3;

        const float *vals = values + group * nval;
        modify_conformer(sm_init_coord, new_pos, vals, sm_torsions, sm_masks, npred, nval, ntorsion, tmp);
        single_SF_loss(
          new_pos, sm_pocket, sm_pred_cross_dist, sm_pred_holo_dist, 6, npred, npocket, tmp, loss + group);
    }
    #endif
}
#if 0
__global__ void sched(float *data) {
    printf(">>>> enter block %d %d thread %d %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    for (int i = 0; i < 1000000; i++) {
        *data += (float)i;
    }
    printf("==== Leave block %d %d thread %d %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}
#endif

static size_t sm_max_size = 0;
void sm_init(int device) {
    cudaDeviceProp prop;
    auto err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        return;
    }
    sm_max_size = prop.sharedMemPerBlock;
}
void dock_gpu(float *init_coord, float *pocket, float *pred_cross_dist, float *pred_holo_dist,
              float *values, int *torsions, uint8_t *masks, int npred, int npocket, int nval,
              int ntorsion, float *loss, float *dev, int &devSize, cudaStream_t stream,
              int smMaxSize) {
    //    npred * 3 + max(
    //          18 + max(9*nedge, 6 * npos),
    //          npred * npocket + npred * npred + 2 + ((npred * npocket + 3) >> 2) + npred *
    //          max(npred, npocket)
    int smsize = npred * 3
                 + std::max(18 + std::max(9 * ntorsion, 6 * npred),
                            npred * npocket + npred * npred + 2 + ((npred * npocket + 3) >> 2)
                              + npred * std::max(npred, npocket));
    smsize *= sizeof(float);

    if (dev == nullptr) {
        devSize = smsize;
        return;
    }
    float *tmp = nullptr;
    if (smsize > smMaxSize) {
        tmp = dev;
        assert(smsize <= devSize);
        smsize = 0;
    }

    // std::cout << "npred " << npred << " npocket " << npocket << " mem " << smsize << std::endl;

    dim3 block(npred);
    dim3 grid(1, 1, 1);
    dock_kernel<<<grid, block, smsize, stream>>>(init_coord, pocket, pred_cross_dist,
                                                 pred_holo_dist, values, torsions, masks, npred,
                                                 npocket, nval, ntorsion, loss, tmp);
}
void dock_grad_gpu(float *init_coord, float *pocket, float *pred_cross_dist, float *pred_holo_dist,
                   float *values, int *torsions, uint8_t *masks, int npred, int npocket, int nval,
                   int ngval, int ntorsion,
                   float *loss,  // ngval float array
                   float *dev,
                   int &devSize,  // in bytes
                   cudaStream_t stream, int smMaxSize) {
    //    npred * 3 + max(
    //          18 + max(9*nedge, 6 * npos),
    //          npred * npocket + npred * npred + 2 + ((npred * npocket + 3) >> 2) + npred *
    //          max(npred, npocket)
    int smsize = npred * 3
                 + std::max(18 + std::max(9 * ntorsion, 6 * npred),
                            npred * npocket + npred * npred + 2 + ((npred * npocket + 3) >> 2)
                              + npred * std::max(npred, npocket));
    smsize *= sizeof(float);
    // smsize += 1024 * 1024;
    int blksz = smsize;  // in bytes
    smsize *= ngval;

    // get required memories
    if (dev == nullptr) {
        // std::cout << "eval blksz " << blksz << std::endl;
        devSize = smsize;
        return;
    }

    float *tmp = nullptr;
    if (smsize > smMaxSize) {
        tmp = dev;
        // std::cout << "blksize " << blksz << " smsize " << smsize << " max sm " << smMaxSize << std::endl;
        assert(smsize <= devSize);
    }
    assert(ntorsion + 6 == nval);

    dim3 block(npred);
    dim3 grid(ngval);
    dock_grad_kernel<<<grid, block, 0, stream>>>(
        init_coord, pocket, pred_cross_dist, pred_holo_dist, values, torsions, masks, npred,
        npocket, nval, ngval, ntorsion, loss, tmp, blksz / sizeof(float));
}
}  // namespace dock