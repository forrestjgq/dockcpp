#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// ================================= 矩阵奇异值分解 =================================
template <typename _Tp>
static void JacobiSVD(std::vector<std::vector<_Tp>>& At,
                      std::vector<std::vector<_Tp>>& _W,
                      std::vector<std::vector<_Tp>>& Vt) {
    double minval = FLT_MIN;
    _Tp eps       = (_Tp)(FLT_EPSILON * 2);
    const int m   = At[0].size();
    const int n   = _W.size();
    const int n1  = m;  // urows
    std::vector<double> W(n, 0.);

    for (int i = 0; i < n; i++) {
        double sd{ 0. };
        for (int k = 0; k < m; k++) {
            _Tp t = At[i][k];
            sd += (double)t * t;
        }
        W[i] = sd;

        for (int k = 0; k < n; k++)
            Vt[i][k] = 0;
        Vt[i][i] = 1;
    }

    int max_iter = std::max(m, 30);
    for (int iter = 0; iter < max_iter; iter++) {
        bool changed = false;
        _Tp c, s;

        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                _Tp *Ai = At[i].data(), *Aj = At[j].data();
                double a = W[i], p = 0, b = W[j];

                for (int k = 0; k < m; k++)
                    p += (double)Ai[k] * Aj[k];

                if (std::abs(p) <= eps * std::sqrt((double)a * b))
                    continue;

                p *= 2;
                double beta = a - b, gamma = hypot((double)p, beta);
                if (beta < 0) {
                    double delta = (gamma - beta) * 0.5;
                    s            = (_Tp)std::sqrt(delta / gamma);
                    c            = (_Tp)(p / (gamma * s * 2));
                } else {
                    c = (_Tp)std::sqrt((gamma + beta) / (gamma * 2));
                    s = (_Tp)(p / (gamma * c * 2));
                }

                a = b = 0;
                for (int k = 0; k < m; k++) {
                    _Tp t0 = c * Ai[k] + s * Aj[k];
                    _Tp t1 = -s * Ai[k] + c * Aj[k];
                    Ai[k]  = t0;
                    Aj[k]  = t1;

                    a += (double)t0 * t0;
                    b += (double)t1 * t1;
                }
                W[i] = a;
                W[j] = b;

                changed = true;

                _Tp *Vi = Vt[i].data(), *Vj = Vt[j].data();

                for (int k = 0; k < n; k++) {
                    _Tp t0 = c * Vi[k] + s * Vj[k];
                    _Tp t1 = -s * Vi[k] + c * Vj[k];
                    Vi[k]  = t0;
                    Vj[k]  = t1;
                }
            }
        }

        if (!changed)
            break;
    }

    for (int i = 0; i < n; i++) {
        double sd{ 0. };
        for (int k = 0; k < m; k++) {
            _Tp t = At[i][k];
            sd += (double)t * t;
        }
        W[i] = std::sqrt(sd);
    }

    for (int i = 0; i < n - 1; i++) {
        int j = i;
        for (int k = i + 1; k < n; k++) {
            if (W[j] < W[k])
                j = k;
        }
        if (i != j) {
            std::swap(W[i], W[j]);

            for (int k = 0; k < m; k++)
                std::swap(At[i][k], At[j][k]);

            for (int k = 0; k < n; k++)
                std::swap(Vt[i][k], Vt[j][k]);
        }
    }

    for (int i = 0; i < n; i++)
        _W[i][0] = (_Tp)W[i];

    srand(time(nullptr));

    for (int i = 0; i < n1; i++) {
        double sd = i < n ? W[i] : 0;

        for (int ii = 0; ii < 100 && sd <= minval; ii++) {
            // if we got a zero singular value, then in order to get the corresponding left singular
            // vector we generate a random vector, project it to the previously computed left
            // singular vectors, subtract the projection and normalize the difference.
            const _Tp val0 = (_Tp)(1. / m);
            for (int k = 0; k < m; k++) {
                unsigned int rng = rand() % 4294967295;  // 2^32 - 1
                _Tp val          = (rng & 256) != 0 ? val0 : -val0;
                At[i][k]         = val;
            }
            for (int iter = 0; iter < 2; iter++) {
                for (int j = 0; j < i; j++) {
                    sd = 0;
                    for (int k = 0; k < m; k++)
                        sd += At[i][k] * At[j][k];
                    _Tp asum = 0;
                    for (int k = 0; k < m; k++) {
                        _Tp t    = (_Tp)(At[i][k] - sd * At[j][k]);
                        At[i][k] = t;
                        asum += std::abs(t);
                    }
                    asum = asum > eps * 100 ? 1 / asum : 0;
                    for (int k = 0; k < m; k++)
                        At[i][k] *= asum;
                }
            }

            sd = 0;
            for (int k = 0; k < m; k++) {
                _Tp t = At[i][k];
                sd += (double)t * t;
            }
            sd = std::sqrt(sd);
        }

        _Tp s = (_Tp)(sd > minval ? 1 / sd : 0.);
        for (int k = 0; k < m; k++)
            At[i][k] *= s;
    }
}

// matSrc为原始矩阵，支持非方阵，matD存放奇异值，matU存放左奇异向量，matVt存放转置的右奇异向量
template <typename _Tp>
int svd(const std::vector<std::vector<_Tp>>& matSrc,
        std::vector<std::vector<_Tp>>& matD,
        std::vector<std::vector<_Tp>>& matU,
        std::vector<std::vector<_Tp>>& matVt) {
    int m = matSrc.size();
    int n = matSrc[0].size();
    for (const auto& sz : matSrc) {
        if (n != (int)sz.size()) {
            fprintf(stderr, "matrix dimension dismatch\n");
            return -1;
        }
    }

    bool at = false;
    if (m < n) {
        std::swap(m, n);
        at = true;
    }

    matD.resize(n);
    for (int i = 0; i < n; ++i) {
        matD[i].resize(1, (_Tp)0);
    }
    matU.resize(m);
    for (int i = 0; i < m; ++i) {
        matU[i].resize(m, (_Tp)0);
    }
    matVt.resize(n);
    for (int i = 0; i < n; ++i) {
        matVt[i].resize(n, (_Tp)0);
    }
    std::vector<std::vector<_Tp>> tmp_u = matU, tmp_v = matVt;

    std::vector<std::vector<_Tp>> tmp_a, tmp_a_;
    if (!at)
        cv::transpose(matSrc, tmp_a);
    else
        tmp_a = matSrc;

    if (m == n) {
        tmp_a_ = tmp_a;
    } else {
        tmp_a_.resize(m);
        for (int i = 0; i < m; ++i) {
            tmp_a_[i].resize(m, (_Tp)0);
        }
        for (int i = 0; i < n; ++i) {
            tmp_a_[i].assign(tmp_a[i].begin(), tmp_a[i].end());
        }
    }
    JacobiSVD(tmp_a_, matD, tmp_v);

    if (!at) {
        cv::transpose(tmp_a_, matU);
        matVt = tmp_v;
    } else {
        cv::transpose(tmp_v, matU);
        matVt = tmp_a_;
    }

    return 0;
}
void print_matrix(std::vector<std::vector<float>>& v) {
    for (auto i = 0u; i < v.size(); i++) {
        printf("%d: ", i);
        auto& p1 = v[i];
        for (auto j = 0u; j < p1.size(); j++) {
            printf("%f ", p1[j]);
        }
        printf("\n");
    }
}
int test_SVD() {
    // std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
    //				{ -3.6f, 9.2f, 0.5f, 7.2f },
    //				{ 4.3f, 1.3f, 9.4f, -3.4f },
    //				{ 6.4f, 0.1f, -3.7f, 0.9f } };
    // const int rows{ 4 }, cols{ 4 };

    // std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
    //				{ -3.6f, 9.2f, 0.5f, 7.2f },
    //				{ 4.3f, 1.3f, 9.4f, -3.4f } };
    // const int rows{ 3 }, cols{ 4 };

    std::vector<std::vector<float>> vec{ { 0.68f, 0.597f },
                                         { -0.211f, 0.823f },
                                         { 0.566f, -0.605f } };
    const int rows{ 3 }, cols{ 2 };

    fprintf(stderr, "source matrix:\n");
    print_matrix(vec);

    fprintf(stderr, "\nc++ implement singular value decomposition:\n");
    std::vector<std::vector<float>> matD, matU, matVt;
    if (svd(vec, matD, matU, matVt) != 0) {
        fprintf(stderr, "C++ implement singular value decomposition fail\n");
        return -1;
    }
    fprintf(stderr, "singular values:\n");
    print_matrix(matD);
    fprintf(stderr, "left singular vectors:\n");
    print_matrix(matU);
    fprintf(stderr, "transposed matrix of right singular values:\n");
    print_matrix(matVt);
#if 1
    fprintf(stderr, "\nopencv singular value decomposition:\n");
    cv::Mat mat(rows, cols, CV_32FC1);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            mat.at<float>(y, x) = vec.at(y).at(x);
        }
    }

    /*
        w calculated singular values
        u calculated left singular vectors
        vt transposed matrix of right singular vectors
    */
    cv::Mat w, u, vt, v;
    cv::SVD::compute(mat, w, u, vt, 4);
    // cv::transpose(vt, v);

    fprintf(stderr, "singular values:\n");
    std::cout << w << std::endl;
    fprintf(stderr, "left singular vectors:\n");
    std::cout << u << std::endl;
    fprintf(stderr, "transposed matrix of right singular values:\n");
    std::cout << vt << std::endl;
#endif
    return 0;
}

int main() {
    test_SVD();
    return 0;
}