#include <algorithm>
#include <iostream>
#include <assert.h>

#include "dockcu.h"

#include "math.h"
#define REST __restrict__
#define IN
#define OUT


#if USE_DOUBLE
typedef double3 dtype3;
typedef double4 dtype4;
#define EXP(x) exp(x)
#define SQRT(x) sqrt(x)
#define NORM3D(x, y, z) norm3d(x, y, z)
#define reciprocal(x) __drcp_rn(x)
#define RNORM3D(x, y, z) rnorm3d(x, y, z)
#define make_dtype3 make_double3
#define make_dtype4 make_double4
#else
typedef float3 dtype3;
typedef float4 dtype4;
#define EXP(x) expf(x)
#define SQRT(x) sqrtf(x)
#define NORM3D norm3df
#define reciprocal(x) __frcp_rn(x)
#define RNORM3D rnorm3df
#define make_dtype3 make_float3
#define make_dtype4 make_float4
#endif

namespace dock {
#define DOCKDBG 1
#define GRIDIDX 1
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
#define FOR_LOOP(i, end) for (int i = threadIdx.x; i < end; i += blockDim.x)
#define IS_THREAD(n)     (threadIdx.x == (n))
#define IS_MAIN_THREAD() IS_THREAD(0)
#define PI               3.141592653589793

__device__ void dumparr(int m, int n, const int * REST p) {
    printf(" ====== t[%d,%d] (%dx%d) ======\n", threadIdx.x, threadIdx.y, m, n);
    for (int i = 0; i < m; i++) {
        printf("\t%d:\t", i);
        const int *p1 = p + i * n;
        for (int j = 0; j < n; j++) {
            printf("%d ", int(p1[j]));
        }
        printf("\n");
    }
    printf("\n");
}
__device__ void dumparr(int m, int n, const uint8_t * REST p) {
    printf(" ====== t[%d,%d] (%dx%d) ======\n", threadIdx.x, threadIdx.y, m, n);
    for (int i = 0; i < m; i++) {
        printf("\t%d:\t", i);
        const uint8_t *p1 = p + i * n;
        for (int j = 0; j < n; j++) {
            printf("%d ", int(p1[j]));
        }
        printf("\n");
    }
    printf("\n");
}
__device__ void dumparr(int m, int n, const float * REST p) {
    printf(" ====== t[%d,%d] (%dx%d) ======\n", threadIdx.x, threadIdx.y, m, n);
    for (int i = 0; i < m; i++) {
        printf("\t%d:\t" , i);
        const float *p1 = p + i * n;
        for (int j = 0; j < n; j++) {
            printf("%f ", p1[j]);
        }
        printf("\n");
    }
    printf("\n");
}
__device__ void dumparr(int m, int n, const double * REST p) {
    printf(" ====== t[%d,%d] (%dx%d) ======\n", threadIdx.x, threadIdx.y, m, n);
    for (int i = 0; i < m; i++) {
        printf("\t%d:\t" , i);
        const double *p1 = p + i * n;
        for (int j = 0; j < n; j++) {
            printf("%f ", p1[j]);
        }
        printf("\n");
    }
    printf("\n");
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
#define PRINT(...) do {\
    if (INGRID) printf(__VA_ARGS__);\
} while(0)

#else
#define DUMP(hdr, m, n, p) 
#define DUMPARR(x1, y1, hdr, m, n, p) 
#define PRINT(...)
#endif
# define DUMPARR1(hdr, m, n, p) \
  do { \
   if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) { \
    printf(hdr); \
    dumparr(m, n, p); \
   } \
  } while (0)

__device__ __forceinline__ dtype3 sub3(dtype3 a, dtype3 b) {
    return make_dtype3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ void sub3p(const dtype * REST a, const dtype * REST b, dtype * REST out) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}
__device__ __forceinline__ void add3p(const dtype * REST a, const dtype * REST b, dtype * REST out) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
}
__device__ __forceinline__ void sub3p(IN OUT dtype * REST a, const dtype * REST b) {
    a[0] = a[0] - b[0];
    a[1] = a[1] - b[1];
    a[2] = a[2] - b[2];
}
__device__ __forceinline__ void add3p(IN OUT dtype * REST a, const dtype * REST b) {
    a[0] = a[0] + b[0];
    a[1] = a[1] + b[1];
    a[2] = a[2] + b[2];
}
// https://zhuanlan.zhihu.com/p/543841297
// sigmoid(x) = 1/(1+e^(-x))
template <typename T>
__device__ __forceinline__ T SigmoidForward(T x) {
    return T(1) / (T(1) + EXP(-x));
}
template <typename T>
__device__ __forceinline__ void SigmoidBackward(T dy, T y) {
    return dy * y * (T(1) - y);
}
// sum(abs(x)**ord)**(1./ord) where ord = 2
template <int n>
__device__ __forceinline__ dtype Norm2(const dtype * REST d) {
    dtype f = 0;
    for (int i = 0; i < n; i++) {
        dtype x = d[i];  // abs is ignored for ^2
        f += x * x;
    }
    return SQRT(f);
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
__device__ __forceinline__ dtype4 axis_angle_to_quaternion(dtype3 axis) {
    dtype angle = NORM3D(axis.x, axis.y, axis.z);  // sqrt(x^2 + y^2 + z^2)
    dtype half  = angle * 0.5f;
    DBG("axis %f %f %f angle %f half %f\n", axis.x, axis.y, axis.z, angle, half);
    dtype sins;
    if (angle < 1e-6) {
        sins = 0.5f - (angle * angle) * 0.020833333333333332;  // 1/48 = 0.020833333333333332
    } else {
        sins = sin(half) * reciprocal(angle);
    }
    DBG("sins %f \n", sins);
    dtype4 ret;
    ret.x = cos(half);
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
__device__ __forceinline__ void quaternion_to_matrix(dtype4 q, OUT dtype * REST m) {
    dtype4 q2 = make_dtype4(q.x * q.x, q.y * q.y, q.z * q.z, q.w * q.w);
    dtype s   = reciprocal(q2.x + q2.y + q2.z + q2.w) * 2.0;
    m[0] = 1 - s * (q2.z + q2.w), m[1] = s * (q.y * q.z - q.w * q.x),
    m[2] = s * (q.y * q.w + q.z * q.x), m[3] = s * (q.y * q.z + q.w * q.x),
    m[4] = 1 - s * (q2.y + q2.w), m[5] = s * (q.z * q.w - q.y * q.x),
    m[6] = s * (q.y * q.w - q.z * q.x), m[7] = s * (q.z * q.w + q.y * q.x),
    m[8] = 1 - s * (q2.y + q2.z);
}
// p: source point, m: rotate matrix(3x3), n: n-th dim after rotate, value [0,2]
__device__ __forceinline__ dtype rotate(dtype3 p, const dtype * REST m, int n) {
    return p.x * m[n * 3 + 0] + p.y * m[n * 3 + 1] + p.z * m[n * 3 + 2];
}
// calc m1[m,n] @ m2[n,k]
// if trans, calc m1[m,n] @ m2[k,n].T
template <int m, int n, int k, bool trans>
__device__ __forceinline__ void matmul(const dtype * REST m1, const dtype * REST m2, dtype * REST out) {
    int s = 0;
    for (int r = 0; r < m; r++) {  // rows
        int row = r * n;
        for (int c = 0; c < k; c++) {  // columns
            const dtype *rp, *cp;
            if (trans) {
                rp        = &m1[row];
                cp        = &m2[c * k];
                dtype sum = 0.;
                for (int i = 0; i < n; i++) {
                    sum += rp[i] * cp[i];
                }
                out[s++] = sum;

            } else {
                rp        = &m1[row];
                cp        = &m2[c];
                dtype sum = 0.;
                for (int i = 0; i < n; i++) {
                    sum += rp[i] * cp[i * k];
                }
                out[s++] = sum;
            }
        }
    }
}
template <bool trans>
__device__ __forceinline__ void matmul1(const dtype * REST m1, const dtype * REST m2, dtype * REST out, int m, int n, int k) {
    int s = 0;
    for (int r = 0; r < m; r++) {  // rows
        int row = r * n;
        for (int c = 0; c < k; c++) {  // columns
            const dtype *rp, *cp;
            if (trans) {
                rp        = &m1[row];
                cp        = &m2[c * k];
                dtype sum = 0.;
                for (int i = 0; i < n; i++) {
                    sum += rp[i] * cp[i];
                }
                out[s++] = sum;

            } else {
                rp        = &m1[row];
                cp        = &m2[c];
                dtype sum = 0.;
                for (int i = 0; i < n; i++) {
                    sum += rp[i] * cp[i * k];
                }
                out[s++] = sum;
            }
        }
    }
}
// m1: mxn, m2: nxk, out: mxk
// if ltrans, m1: nxm
// if rtrans, m2: kxn
// concurrent mat multiple
template <bool trans>
__device__ __forceinline__ void matmulc(const dtype * REST m1, const dtype * REST m2, dtype * REST out, int m, int n, int k) {
    FOR_LOOP(i, m * k) {
        int c = 0, r = 0, idx = 0;
        // find col and row
        for (r = 0; r < m; r++, idx += k) {
            int imin = idx;
            int imax = idx + k;
            if (i >= imin && i < imax) {
                // i is in this row
                c = i - idx;
                break;
            }
        }
        // printf("i = %d c = %d r = %d\n", i, c, k);
        // multiple row r from m1 and col c from m2
        // columns
        const dtype *rp, *cp;
        dtype sum = 0.;
        int row = r * n;
        if (trans) {
            rp        = &m1[row];
            cp        = &m2[c * k];
            for (int j = 0; j < n; j++) {
                sum += rp[j] * cp[j];
            }
        } else {
            rp        = &m1[row];
            cp        = &m2[c];
            for (int j = 0, j1 = 0; j < n; j++, j1 += k) {
                sum += rp[j] * cp[j1];
            }
        }
        out[i] = sum;
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
__device__ __forceinline__ void gen_matrix_from_rot_vec(const dtype * REST k, dtype * REST out, double theta) {
    dtype tsin = sin(theta);
    dtype tcos = 1.0 - cos(theta);
    DUMPARR(0, 0, "K", 3, 3, k);
    matmul<n, n, n, false>(k, k, out);
    DUMPARR(0, 0, "sin", 1, 1, &tsin);
    DUMPARR(0, 0, "cos", 1, 1, &tcos);
    DUMPARR(0, 0, "KK", 3, 3, out);
    int idx = 0;
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            out[idx] = out[idx] * tcos + tsin * k[idx] + (r == c ? 1. : 0.);
            idx++;
        }
    }
    DUMPARR(0, 0, "R", 3, 3, out);
}
__device__ __forceinline__ void gen_matrix_from_rot_vec3(const dtype * REST k, dtype * REST out, dtype theta) {
    dtype K[3][3] = { { 0., -k[2], k[1] }, { k[2], 0., -k[0] }, { -k[1], k[0], 0. } };
    gen_matrix_from_rot_vec<3>((dtype *)K, out, theta);
}
#if 0
__device__ __forceinline__ void modify_conformer_torsion_angles_single(dtype *pos,
                                                                       dtype *out,
                                                                       uint8_t *mask,
                                                                       int u,
                                                                       int v,
                                                                       dtype theta,
                                                                       int n,
                                                                       dtype *tmp) {
    dtype *pu  = pos + 3 * u;
    dtype *pv  = pos + 3 * v;
    dtype *rot = tmp;
    if (threadIdx.x == 0) {
        dtype rot_vec[3];
        sub3p(pu, pv, rot_vec);
        DUMP("rot vec", 1, 3, rot_vec);
        dtype norm = RNORM3D(rot_vec[0], rot_vec[1], rot_vec[2]);
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
            dtype tp[3];
            dtype *p    = pos + 3 * i;
            dtype *outp = out + 3 * i;
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
// require tmp: 9 * nedge dtypes
__device__ __forceinline__ void modify_conformer_torsion_angles_concurrent(
    const dtype *REST pos, OUT dtype *REST out, dtype *REST tmp, const uint8_t *REST mask_rotate,
    const int *REST edge_index, const dtype *REST torsion_updates, int n, int nedge) {
    dtype *rot_mats = tmp;
    tmp += nedge * 9;
    FOR_LOOP(i, n) {
        int seq  = i * 3;
        out[seq] = pos[seq], out[seq + 1] = pos[seq + 1], out[seq + 2] = pos[seq + 2];
        if (i < nedge) {
            int idx_edge = i;
            int u        = edge_index[idx_edge * 2];
            int v        = edge_index[idx_edge * 2 + 1];

            dtype theta   = torsion_updates[idx_edge];
            const dtype *pu     = pos + 3 * u;
            const dtype *pv     = pos + 3 * v;
            dtype *rot    = rot_mats + idx_edge * 9;
            dtype rot_vec[3];
            sub3p(pu, pv, rot_vec);
            DUMPARR(1, 0, "rot vec", 1, 3, rot_vec);
            dtype norm = RNORM3D(rot_vec[0], rot_vec[1], rot_vec[2]);
            rot_vec[0] *= norm, rot_vec[1] *= norm, rot_vec[2] *= norm;
            DUMPARR(1, 0, "rot vec norm", 1, 3, rot_vec);
            DBG("torsion updates %f\n", theta);
            gen_matrix_from_rot_vec3(rot_vec, rot, theta);
            DUMPARR(1, 0, "rot mat", 3, 3, rot);
        }
    }

    __syncthreads();
    DUMPARR(0, 0, "mask 0", 1, n, mask_rotate);
    DUMPARR(0, 0, "mask 1", 1, n, mask_rotate + n);
    FOR_LOOP(i, n) {
        const dtype *p    = pos + 3 * i;
        dtype *outp = out + 3 * i;
        for (int k = 0; k < nedge; k++) {
            const uint8_t *mask = mask_rotate + n * k;
            if (mask[i] != 0) {
                int u      = edge_index[k * 2];
                int v      = edge_index[k * 2 + 1];
                const dtype *pu  = pos + 3 * u;
                const dtype *pv  = pos + 3 * v;
                const dtype *rot = rot_mats + k * 9;
                dtype tp[3];
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
    const dtype *REST pos, OUT dtype *REST out, dtype *REST tmp, const uint8_t *REST mask_rotate,
    const int *REST edge_index, const dtype *REST torsion_updates, int n, int nedge) {
    dtype *rot_mats = tmp;
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

            dtype theta     = torsion_updates[idx_edge];
            const dtype *pu = out + 3 * u;
            const dtype *pv1 = out + 3 * v;
            dtype pv[3]; // must be copied
            pv[0] = pv1[0], pv[1] = pv1[1], pv[2] = pv1[2];
            dtype rot_vec[3];
            dtype rot[9];
            PRINT("idx edge %d\n", k);
            sub3p(pu, pv, rot_vec);
            DUMP("rot vec", 1, 3, rot_vec);
            dtype norm = RNORM3D(rot_vec[0], rot_vec[1], rot_vec[2]);
            rot_vec[0] *= norm, rot_vec[1] *= norm, rot_vec[2] *= norm;
            DUMP("rot vec norm", 1, 3, rot_vec);
            DBG("torsion updates %f\n", theta);
            gen_matrix_from_rot_vec3(rot_vec, rot, theta);
            DUMP("rot mat", 3, 3, rot);
            DUMP("posv", 1, 3, pv);
            const uint8_t *mask = mask_rotate + n * k;
            for (int i = 0; i < n; i++) {
                if (mask[i] != 0) {
                    dtype *outp = out + 3 * i;
                    PRINT("change pos %d\n", i);
                    DUMP("Before", 1, 3, outp);
                    dtype tp[3];
                    sub3p(outp, pv, tp);
                    DUMP("Tp", 1, 3, tp);
                    matmul<1, 3, 3, true>(tp, rot, outp);
                    DUMP("TpMul", 1, 3, outp);
                    DUMP("TpMulPv", 1, 3, pv);
                    add3p(outp, pv);
                    DUMP("After", 1, 3, outp);
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
__device__ __forceinline__ void rigid_transform_Kabsch_3D_torch_main(const dtype *REST a,
                                                                const dtype *REST b,
                                                                dtype *REST tmp, OUT dtype *REST r,
                                                                OUT dtype *REST t, int n) {
    // mean of a, b

    dtype *at, *bt;
    at = tmp, tmp += n * 3;
    bt = tmp, tmp += n * 3;
    DUMP("A", n, 3, a);
    DUMP("B", n, 3, b);
    dtype ca[3] = { 0., 0., 0. };
    dtype cb[3] = { 0., 0., 0. };
    const dtype * REST tpa = a;
    const dtype *REST tpb = b;
    for (int i = 0; i < n * 3; i += 3) {
        add3p(ca, tpa + i);
        add3p(cb, tpb + i);
    }
    dtype re = reciprocal((dtype)n);
    for (int i = 0; i < 3; i++) {
        ca[i] *= re;
        cb[i] *= re;
    }
    DUMP("Center A", 1, 3, ca);
    DUMP("Center B", 1, 3, cb);
    // substract mean and turn nx3 -> 3xn
    for (int i = 0; i < 3; i++) {
        dtype * REST tp = at + i * n;
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

    dtype h[9];
    matmul1<false>(at, bt, h, 3, n, 3);
    DUMP("H", 3, 3, h);

    float hf[9],uf[9], s[3], vf[9];
    hf[0] = h[0], hf[1] = h[1], hf[2] = h[2];
    hf[3] = h[3], hf[4] = h[4], hf[5] = h[5];
    hf[6] = h[6], hf[7] = h[7], hf[8] = h[8];
    svd(hf, uf, s, vf);
    // svd already output torch svd Vt.T
    // trans<dtype>(v, v, 3, 3);
    DUMP("U", 3, 3, uf);
    DUMP("Vt", 3, 3, vf);
    DUMP("S", 1, 3, s);

    dtype u[9], v[9];
    u[0] = uf[0], u[1] = uf[1], u[2] = uf[2];
    u[3] = uf[3], u[4] = uf[4], u[5] = uf[5];
    u[6] = uf[6], u[7] = uf[7], u[8] = uf[8];
    v[0] = vf[0], v[1] = vf[1], v[2] = vf[2];
    v[3] = vf[3], v[4] = vf[4], v[5] = vf[5];
    v[6] = vf[6], v[7] = vf[7], v[8] = vf[8];

    matmul<3, 3, 3, true>(v, u, r);
    DUMP("r", 3, 3, r);

    dtype dt = det3x3<dtype>(r);
    DUMP("det", 1, 1, &dt);
    if (dt < 0) {
        dtype ss[9] = { 1, 0, 0, 0, 1, 0, 0, 0, -1 };
        dtype st[9];
        matmul<3, 3, 3, false>(v, ss, st);
        matmul<3, 3, 3, true>(st, u, r);
    }
    dtype r1[9];
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
__device__ __forceinline__ void rigid_transform_Kabsch_3D_torch(const dtype *REST a,
                                                                const dtype *REST b,
                                                                dtype *REST tmp, OUT dtype *REST r,
                                                                OUT dtype *REST t, int n) {
    // mean of a, b

    dtype *at, *bt, *h;
    at = tmp, tmp += n * 3;
    bt = tmp, tmp += n * 3;
    h = tmp, tmp += 9;

    DUMPARR(0, 0, "A", n, 3, a);
    DUMPARR(0, 0, "B", n, 3, b);
    dtype ca[3] = { 0., 0., 0. };
    dtype cb[3] = { 0., 0., 0. };
    const dtype * REST tpa = a;
    const dtype *REST tpb = b;
    dtype re = 0;

    if (IS_MAIN_THREAD()) {
        for (int i = 0; i < n; i++) {
            add3p(ca, tpa + i * 3);
            add3p(cb, tpb + i * 3);
        }
        re = reciprocal((dtype)n);
    }
    __syncthreads();
    FOR_LOOP(i, 3) {
        ca[i] *= re;
        cb[i] *= re;
    }
    __syncthreads();
    DUMPARR(0, 0, "Center A", 1, 3, ca);
    DUMPARR(0, 0, "Center B", 1, 3, cb);

    // a - ca -> at
    FOR_LOOP(i, n) {
        sub3p(b + i * 3, cb, bt + i * 3);
        // turn nx3 -> 3xn, now row[i] -> col[i]
        dtype *tp       = at + i;
        const dtype *ta = a + i * 3;
        tp[0]           = ta[0] - ca[0];
        tp[n]           = ta[1] - ca[1];
        tp[2 * n]       = ta[2] - ca[2];
    }
    __syncthreads();
    DUMPARR(0, 0, "Am", 3, n, at);
    DUMPARR(0, 0, "Bm", n, 3, bt);

    matmulc<false>(at, bt, h, 3, n, 3);
    __syncthreads();

    DUMPARR(0, 0, "H", 3, 3, h);
    if (IS_MAIN_THREAD()) {
        float hf[9], uf[9], s[3], vf[9];
        hf[0] = h[0], hf[1] = h[1], hf[2] = h[2];
        hf[3] = h[3], hf[4] = h[4], hf[5] = h[5];
        hf[6] = h[6], hf[7] = h[7], hf[8] = h[8];
        svd(hf, uf, s, vf);
        // svd already output torch svd Vt.T
        // trans<dtype>(v, v, 3, 3);
        DUMP("U", 3, 3, uf);
        DUMP("Vt", 3, 3, vf);
        DUMP("S", 1, 3, s);

        dtype u[9], v[9];
        u[0] = uf[0], u[1] = uf[1], u[2] = uf[2];
        u[3] = uf[3], u[4] = uf[4], u[5] = uf[5];
        u[6] = uf[6], u[7] = uf[7], u[8] = uf[8];
        v[0] = vf[0], v[1] = vf[1], v[2] = vf[2];
        v[3] = vf[3], v[4] = vf[4], v[5] = vf[5];
        v[6] = vf[6], v[7] = vf[7], v[8] = vf[8];

        matmul<3, 3, 3, true>(v, u, r);
        DUMP("r", 3, 3, r);

        dtype dt = det3x3<dtype>(r);
        DUMP("det", 1, 1, &dt);
        if (dt < 0) {
            dtype ss[9] = { 1, 0, 0, 0, 1, 0, 0, 0, -1 };
            dtype st[9];
            matmul<3, 3, 3, false>(v, ss, st);
            matmul<3, 3, 3, true>(st, u, r);
        }
        dtype r1[9];
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
    __syncthreads();
}

// pos: npos x 3
// newpos: npos x 3, output
// values: size nval
// edge_index: nedge x 2
// mask_rotate: npos
// tmp: tmp memory, 18 + max(9*nedge, 6 * npos+9) dtypes
// sm: npos * 9 + 12 dtypes
__device__ __forceinline__ void modify_conformer(const dtype *REST pos, OUT dtype *REST newpos,
                                                 const dtype *REST values,
                                                 const int *REST edge_index,
                                                 const uint8_t *REST mask_rotate, int npos,
                                                 int nval, int nedge, dtype *REST tmp, dtype * sm) {
    dtype *center, *rot_mat, *tr, *rot;
    center         = tmp, tmp += 3;
    rot_mat        = tmp, tmp += 9;
    tr             = tmp, tmp += 3;
    rot            = tmp, tmp += 3;

    DUMPARR(0, 0, "coords", npos, 3, pos);
    DUMPARR(0, 0, "values", 1, nval, values);

    FOR_LOOP(i, 3) {
        tr[i]  = (SigmoidForward<dtype>(values[i]) - 0.5) * 10;
        rot[i] = (SigmoidForward<dtype>(values[3 + i]) - 0.5) * 2 * PI;
    }
    __syncthreads();
    DUMPARR(0, 0, "tr update", 1, 3, tr);
    DUMPARR(0, 0, "rot update", 1, 3, rot);

    if (IS_MAIN_THREAD()) {
        dtype3 rot1 = make_dtype3(rot[0], rot[1], rot[2]);
        dtype4 qu   = axis_angle_to_quaternion(rot1);
        quaternion_to_matrix(qu, rot_mat);
        DUMPARR(0, 0, "rot mat", 3, 3, rot_mat);
        center[0] = center[1] = center[2] = 0;
        for (int i = 0; i < npos; i++) {
            center[0] += pos[i * 3];
            center[1] += pos[i * 3 + 1];
            center[2] += pos[i * 3 + 2];
        }
        dtype renpos = reciprocal(((dtype)npos));
        center[0] *= renpos, center[1] *= renpos, center[2] *= renpos;
        tr[0] += center[0], tr[1] += center[1], tr[2] += center[2];
        DUMPARR(0, 0, "center", 1, 3, center);
        DUMPARR(0, 0, "tr update + center", 1, 3, tr);
    }
    // todo: should we sync?
    __syncthreads();

    // calc new pos
    FOR_LOOP(n, npos) {
        const dtype *p    = pos + 3 * n;
        dtype *newp = newpos + 3 * n;
        dtype np[3];
        sub3p(p, center, np);
        dtype rotp[3];
        matmul<1, 3, 3, true>(np, rot_mat, rotp);
        add3p(rotp, tr, newp);
    }
    __syncthreads();
    DUMPARR(0, 0, "new pos", npos, 3, newpos);

    // require max(9*nedge, 6 * npos+9)
    if (nval > 6) {
        dtype *flexpos, *r, *t;
        if (sm == nullptr) {
            flexpos = tmp, tmp += npos * 3;
            r       = tmp, tmp += 9;
            t       = tmp, tmp += 3;
        } else {
            flexpos = sm, sm += npos * 3;
            r       = sm, sm += 9;
            t       = sm, sm += 3;
        }
        // require nedge * 9 dtypes
        modify_conformer_torsion_angles(newpos, flexpos, tmp, mask_rotate, edge_index, values + 6,
                                        npos, nedge);
        // require 2 x npos x 3 + 9 dtypes
        rigid_transform_Kabsch_3D_torch(flexpos, newpos, tmp, r, t, npos);
        matmulc<true>(flexpos, r, newpos, npos, 3, 3);
        __syncthreads();
        FOR_LOOP(n, npos) {
            dtype *outp = newpos + n * 3;
            add3p(outp, t);
        }
        __syncthreads();
        DUMPARR(0, 0, "aligned", npos, 3, newpos);
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
__device__ __forceinline__ dtype smooth_l1_loss(dtype a, dtype b, dtype beta, dtype rebeta) {
    dtype z = abs(a - b);
    return z < beta ? 0.5 * z * z * rebeta : z - 0.5 * beta;
}
// a, b: array of m x n
// beta: smooth beta
// tmp: at least m x n dtypes required
// flags, nullptr or flag indicates if corresponding a and b should be calculated
// mean: true for smooth with mean, or false for sum
template <bool mean>
__device__ __forceinline__ void smooth_l1_loss(const dtype *REST a, const dtype *REST b,
                                               dtype *REST tmp, dtype beta, int m, int n,
                                               const uint8_t *REST flags, dtype *REST out) {
    dtype rebeta = reciprocal(beta);
    FOR_LOOP(i, m * n) {
        if (flags == nullptr || flags[i] != 0) {
            tmp[i] = smooth_l1_loss(a[i], b[i], beta, rebeta);
            // DBG("%d: %f %f beta %f %f -> %f\n", i, a[i], b[i], beta, rebeta, tmp[i]);
        }
    }
    __syncthreads();
    if (IS_MAIN_THREAD()) {
        dtype sum = 0;
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
    const dtype *REST predict,            // npredx3
    const dtype *REST pocket,             // npocketx3
    const dtype *REST dist_predict,       // npred x npocket
    const dtype *REST holo_dist_predict,  // npred x npred
    dtype dist_threshold, int npred, int npocket, dtype *REST tmp, dtype *REST out) {
    int ncross = npred * npocket;
    int nsq    = npred * npred;
    dtype *dist, *holo_dist, *cross_dist_score, *dist_score;
    uint8_t *flags;
    // tmp mem: ncross + nsq + 2 + ((ncross + 3) >> 2)
    dist             = tmp, tmp += ncross;
    holo_dist        = tmp, tmp += nsq;
    cross_dist_score = tmp, tmp++;
    dist_score       = tmp, tmp++;
    flags            = (uint8_t *)tmp, tmp += ((ncross + 3) >> 2);

#if 1
    // for each predict, calc distance to each pocket and predict
    FOR_LOOP(i, npred) {
        // i / npred
        const dtype *p1   = predict + 3 * i;
        const dtype *p2   = pocket;
        const dtype *p3 = predict;
        // idx = i * npocket + j, 0 <= i < npred, 0 <= j < npocket
        // dist[idx]: distance from pred[i] to pocket[j]
        for (int j = 0, idx = i * npocket; j < npocket; j++, idx++) {
            dist[idx]     = NORM3D(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
            flags[idx]       = dist_predict[idx] < dist_threshold ? 1 : 0;
            p2 += 3;
        }
        // idx = i * npred + j, 0 <= i < npred, 0 <= j < npred
        // holo_dist[idx]: distance from pred[i] to pred[j]
        for (int j = 0, idx = i * npred; j < npred; j++, idx++) {
            holo_dist[idx] = NORM3D(p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2]);
            p3 += 3;
        }
    }
#else
    dtype divpocket = reciprocal((dtype)npocket);
    dtype divpred = reciprocal((dtype)npred);
    FOR_LOOP(i, ncross) {
        // i / npred
        int idxpred = floor(i * divpocket);
        int idxpocket  = i - idxpred * npocket;
        const dtype *p1   = predict + 3 * idxpred;
        const dtype *p2   = pocket + 3 * idxpocket;
        dist[i]     = NORM3D(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
        flags[i]       = dist_predict[i] < dist_threshold ? 1 : 0;
    }
    FOR_LOOP(i, nsq) {
        // i / npred
        int idxpred = floor(i * divpred);
        int remain  = i - idxpred * npred;
        if (remain < npred) {
            const dtype *p1   = predict + 3 * idxpred;
            const dtype *p2   = predict + 3 * remain;
            holo_dist[i] = NORM3D(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
        }
    }
#endif
    __syncthreads();
    DUMPARR(0, 0, "dist", npred, npocket, dist);
    DUMPARR(0, 0, "holo dist", npred, npred, holo_dist);
    DUMPARR(0, 0, "holo dist pred", npred, npred, holo_dist_predict);

    // require npred x npocket dtypes
    smooth_l1_loss<true>(dist_predict, dist, tmp, 1.0, npred, npocket, flags, cross_dist_score);
    // require npred x npred dtypes
    smooth_l1_loss<true>(holo_dist_predict, holo_dist, tmp, 1.0, npred, npred, nullptr, dist_score);

    __syncthreads();

    if (IS_MAIN_THREAD()) {
        *out = *cross_dist_score + *dist_score * 1.0;
        DUMPARR(0, 0, "cross dist score", 1, 1, cross_dist_score);
        DUMPARR(0, 0, "dist score", 1, 1, dist_score);
        DUMPARR(0, 0, "loss", 1, 1, out);
        // printf("%d: %f\n", blockIdx.x, *out);
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
__global__ void dock_kernel(const dtype *REST init_coord, const dtype *REST pocket,
                            const dtype *REST pred_cross_dist, const dtype *REST pred_holo_dist,
                            const dtype *REST values, const int *REST torsions,
                            const uint8_t *REST masks, int npred, int npocket, int nval,
                            int ntorsion, OUT dtype *REST loss, dtype *REST dev) {
    extern __shared__ dtype sm[];

    dtype *new_pos, *tmp;
    if (dev == nullptr) {
        new_pos = sm;  // require dtype * npred * 3
        tmp     = &sm[npred * 3];
    } else {
        new_pos = dev;  // require dtype * npred * 3
        tmp     = &dev[npred * 3];
    }

    // require 18 + max(9*nedge, 6 * npos+9) dtypes
    modify_conformer(init_coord, new_pos, values, torsions, masks, npred, nval, ntorsion, tmp, nullptr);
    // require tmp npred *(npocket + npred + max(npred, npocket)+ 2) * dtype + npred*npocket
    // require tmp :
    //   npred * npocket + npred * npred + 2 + ((npred * npocket + 3) >> 2)
    //    + npred * max(npred, npocket)
    single_SF_loss(new_pos, pocket, pred_cross_dist, pred_holo_dist, 6, npred, npocket, tmp, loss);
}

__global__ void dock_grad_kernel(const dtype *REST init_coord, const dtype *REST pocket,
                                 const dtype *REST pred_cross_dist,
                                 const dtype *REST pred_holo_dist, const dtype *REST values,
                                 const int *REST torsions, const uint8_t *REST masks, int npred,
                                 int npocket, int nval, int ngval, int ntorsion,
                                 OUT dtype *REST loss, dtype *REST dev, int blksz /*in dtypes*/, dtype eps) {
    extern __shared__ dtype sm[];

    #if 0 // debug: all runs in block #0, to avoid concurrency between blocks
    if (blockIdx.x == 0) 
        for (int group = 0; group < ngval; group++) 
    #else
    int group      = blockIdx.x;
    if (group < ngval) 
    #endif
    {
        dtype *new_pos, *tmp, *vals;
        tmp = dev + blksz * group; 
        new_pos = tmp, tmp += npred * 3;

        // prepare values
        // vals = sm + nval * group;
        vals = sm;
        dtype *smtmp = vals + nval;
        FOR_LOOP(i, nval) {
            vals[i] = values[i];
            if (group > 0) {
                vals[group - 1] += eps;
            }
        }
        __syncthreads();

        // DUMPARR1("torsion", ntorsion, 2, torsions);
        modify_conformer(init_coord, new_pos, vals, torsions, masks, npred, nval, ntorsion, tmp, nullptr);
        single_SF_loss(new_pos, pocket, pred_cross_dist, pred_holo_dist, 6, npred, npocket, tmp, loss + group);
    }
}
#if 0
__global__ void sched(dtype *data) {
    printf(">>>> enter block %d %d thread %d %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    for (int i = 0; i < 1000000; i++) {
        *data += (dtype)i;
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
int dock_grad_gpu_block_size(int npred, int npocket, int ntorsion) {
    //    npred * 3 + max(
    //          18 + max(9*nedge, 6 * npos),
    //          npred * npocket + npred * npred + 2 + ((npred * npocket + 3) >> 2) + npred *
    //          max(npred, npocket)
    int blksz = npred * 3
                 + std::max(18 + std::max(9 * ntorsion, 6 * npred+9),
                            npred * npocket + npred * npred + 2 + ((npred * npocket + 3) >> 2)
                              + npred * std::max(npred, npocket));
    blksz *= sizeof(dtype);
    return blksz;
}
int dock_grad_gpu_mem_size(int npred, int npocket, int nval, int ntorsion) {
    int ngval = nval + 1; // calc loss and grads for each x in values
    int blksz = dock_grad_gpu_block_size(npred, npocket, ntorsion);
    return blksz * ngval;
}

// values should be nval dtypes
void dock_grad_gpu(dtype *init_coord, dtype *pocket, dtype *pred_cross_dist, dtype *pred_holo_dist,
                   dtype *values, int *torsions, uint8_t *masks, int npred, int npocket, int nval,
                   int ntorsion,
                   dtype *loss,  // ngval dtype array
                   dtype *dev,
                   int devSize,  // in bytes
                   cudaStream_t stream, int smMaxSize, dtype eps) {
    assert(ntorsion + 6 == nval);
    int ngval = nval + 1; // calc loss and grads for each x in values
    int blksz = dock_grad_gpu_block_size(npred, npocket, ntorsion);
    int smsize = (nval + npred * 3 + 12) * sizeof(dtype); // for values, and svd calc

    printf("smsize %d\n", smsize);
    dim3 block(npred);
    dim3 grid(ngval);
    dock_grad_kernel<<<grid, block, smsize, stream>>>(
        init_coord, pocket, pred_cross_dist, pred_holo_dist, values, torsions, masks, npred,
        npocket, nval, ngval, ntorsion, loss, dev, blksz / sizeof(dtype), eps);
}
#if 0
void dock_grad_gpu(dtype *init_coord, dtype *pocket, dtype *pred_cross_dist, dtype *pred_holo_dist,
                   dtype *values, int *torsions, uint8_t *masks, int npred, int npocket, int nval,
                   int ngval, int ntorsion,
                   dtype *loss,  // ngval dtype array
                   dtype *dev,
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
    smsize *= sizeof(dtype);
    // smsize += 1024 * 1024;
    int blksz = smsize;  // in bytes
    smsize *= ngval;

    // get required memories
    if (dev == nullptr) {
        // std::cout << "eval blksz " << blksz << std::endl;
        devSize = smsize;
        return;
    }

    dtype *tmp = nullptr;
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
        npocket, nval, ngval, ntorsion, loss, tmp, blksz / sizeof(dtype));
}
#endif

__global__ void collect_best_dock_kernel(dtype * REST losses, dtype * REST x, dtype * REST g, dtype * REST bestLoss, dtype * REST bestValues, int nval, dtype reps) {
    int idx = threadIdx.x;
    dtype loss = losses[0];
    g[idx] = (losses[idx+1] - loss) * reps;
    if (loss < *bestLoss) {
        if (idx == 0) {
            *bestLoss = loss;
            // printf("best loss %f\n", *bestLoss);
        }
        bestValues[idx] = x[idx];
    }
    // printf("loss %f best %f, idx %d loss %f g %f v %f\n", loss, *bestLoss, idx, losses[idx+1],  g[idx], bestValues[idx]);
}
void collect_best_dock(dtype * losses, dtype *x, dtype *g, dtype *bestLoss, dtype *bestValues, int nval, dtype eps, cudaStream_t stream) {
    collect_best_dock_kernel<<<dim3(1), dim3(nval), 0, stream>>>(losses, x, g, bestLoss, bestValues, nval, 1.0/eps);
}

};  // namespace dock