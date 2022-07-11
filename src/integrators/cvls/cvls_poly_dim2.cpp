#include "cvls_poly_dim2.h"

namespace pbrt {
void Poly_Order0_Dim2_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order0_Dim2_luminance::update(const Eigen::VectorXd &s,
                                        const Spectrum &fx) {
    float l = fx.y();

    float c[1];
    c[0] = 1.0;
    for (int i = 0; i < 1; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 1; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order0_Dim2_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order0_Dim2_luminance::G() const {
    float v = X[0];
    int div[1];
    div[0] = 1;
    for (int i = 1; i < 1; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order0_Dim2_luminance::g(const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[1];
    c[0] = 1.0;
    for (int i = 1; i < 1; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order0_Dim2_luminance::sgd(const Eigen::VectorXd &s, Float v,
                                     const Float lr) {
    float c[1];
    c[0] = 1.0;
    Float e = (g(s) - v);
    for (int i = 0; i < 1; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order0_Dim2_luminance::g_then_sgd(const Eigen::VectorXd &s, Float v,
                                             const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[1];
    c[0] = 1.0;
    for (int i = 1; i < 1; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 1; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}

/////Order 1
void Poly_Order1_Dim2_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order1_Dim2_luminance::update(const Eigen::VectorXd &s,
                                        const Spectrum &fx) {
    float l = fx.y();

    Float c[3];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    for (int i = 0; i < 3; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 3; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order1_Dim2_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order1_Dim2_luminance::G() const {
    float v = X[0];
    int div[3];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    for (int i = 1; i < 3; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order1_Dim2_luminance::g(const Eigen::VectorXd &s) const {
    float v = X[0];
    Float c[3];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    for (int i = 1; i < 3; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order1_Dim2_luminance::sgd(const Eigen::VectorXd &s, Float v,
                                     const Float lr) {
    Float c[3];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    Float e = (g(s) - v);
    for (int i = 0; i < 3; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order1_Dim2_luminance::g_then_sgd(const Eigen::VectorXd &s, Float v,
                                             const Float lr) {
    // Evaluation g
    float g = X[0];
    Float c[3];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    for (int i = 1; i < 3; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 3; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}

/////Ordedr 2
void Poly_Order2_Dim2_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order2_Dim2_luminance::update(const Eigen::VectorXd &s,
                                        const Spectrum &fx) {
    float l = fx.y();

    Float c[6];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    for (int i = 0; i < 6; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 6; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order2_Dim2_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order2_Dim2_luminance::G() const {
    float v = X[0];
    int div[6];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 3;
    div[4] = 4;
    div[5] = 3;
    for (int i = 1; i < 6; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order2_Dim2_luminance::g(const Eigen::VectorXd &s) const {
    float v = X[0];
    Float c[6];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    for (int i = 1; i < 6; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order2_Dim2_luminance::sgd(const Eigen::VectorXd &s, Float v,
                                     const Float lr) {
    Float c[6];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    Float e = (g(s) - v);
    for (int i = 0; i < 6; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order2_Dim2_luminance::g_then_sgd(const Eigen::VectorXd &s, Float v,
                                             const Float lr) {
    // Evaluation g
    float g = X[0];
    Float c[6];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    for (int i = 1; i < 6; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 6; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}

////Order 3
void Poly_Order3_Dim2_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order3_Dim2_luminance::update(const Eigen::VectorXd &s,
                                        const Spectrum &fx) {
    float l = fx.y();

    Float c[10];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    c[6] = s[0] * s[0] * s[0];
    c[7] = s[0] * s[0] * s[1];
    c[8] = s[0] * s[1] * s[1];
    c[9] = s[1] * s[1] * s[1];
    for (int i = 0; i < 10; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 10; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order3_Dim2_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order3_Dim2_luminance::G() const {
    float v = X[0];
    int div[10];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 3;
    div[4] = 4;
    div[5] = 3;
    div[6] = 4;
    div[7] = 6;
    div[8] = 6;
    div[9] = 4;
    for (int i = 1; i < 10; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order3_Dim2_luminance::g(const Eigen::VectorXd &s) const {
    float v = X[0];
    Float c[10];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    c[6] = s[0] * s[0] * s[0];
    c[7] = s[0] * s[0] * s[1];
    c[8] = s[0] * s[1] * s[1];
    c[9] = s[1] * s[1] * s[1];
    for (int i = 1; i < 10; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order3_Dim2_luminance::sgd(const Eigen::VectorXd &s, Float v,
                                     const Float lr) {
    Float c[10];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    c[6] = s[0] * s[0] * s[0];
    c[7] = s[0] * s[0] * s[1];
    c[8] = s[0] * s[1] * s[1];
    c[9] = s[1] * s[1] * s[1];
    Float e = (g(s) - v);
    for (int i = 0; i < 10; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order3_Dim2_luminance::g_then_sgd(const Eigen::VectorXd &s, Float v,
                                             const Float lr) {
    // Evaluation g
    float g = X[0];
    Float c[10];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    c[6] = s[0] * s[0] * s[0];
    c[7] = s[0] * s[0] * s[1];
    c[8] = s[0] * s[1] * s[1];
    c[9] = s[1] * s[1] * s[1];
    for (int i = 1; i < 10; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 10; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}

////Order 4
void Poly_Order4_Dim2_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order4_Dim2_luminance::update(const Eigen::VectorXd &s,
                                        const Spectrum &fx) {
    float l = fx.y();

    Float c[15];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    c[6] = s[0] * s[0] * s[0];
    c[7] = s[0] * s[0] * s[1];
    c[8] = s[0] * s[1] * s[1];
    c[9] = s[1] * s[1] * s[1];
    c[10] = s[0] * s[0] * s[0] * s[0];
    c[11] = s[0] * s[0] * s[0] * s[1];
    c[12] = s[0] * s[0] * s[1] * s[1];
    c[13] = s[0] * s[1] * s[1] * s[1];
    c[14] = s[1] * s[1] * s[1] * s[1];
    for (int i = 0; i < 15; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 15; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order4_Dim2_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order4_Dim2_luminance::G() const {
    float v = X[0];
    int div[15];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 3;
    div[4] = 4;
    div[5] = 3;
    div[6] = 4;
    div[7] = 6;
    div[8] = 6;
    div[9] = 4;
    div[10] = 5;
    div[11] = 8;
    div[12] = 9;
    div[13] = 8;
    div[14] = 5;
    for (int i = 1; i < 15; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order4_Dim2_luminance::g(const Eigen::VectorXd &s) const {
    float v = X[0];
    Float c[15];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    c[6] = s[0] * s[0] * s[0];
    c[7] = s[0] * s[0] * s[1];
    c[8] = s[0] * s[1] * s[1];
    c[9] = s[1] * s[1] * s[1];
    c[10] = s[0] * s[0] * s[0] * s[0];
    c[11] = s[0] * s[0] * s[0] * s[1];
    c[12] = s[0] * s[0] * s[1] * s[1];
    c[13] = s[0] * s[1] * s[1] * s[1];
    c[14] = s[1] * s[1] * s[1] * s[1];
    for (int i = 1; i < 15; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order4_Dim2_luminance::sgd(const Eigen::VectorXd &s, Float v,
                                     const Float lr) {
    Float c[15];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    c[6] = s[0] * s[0] * s[0];
    c[7] = s[0] * s[0] * s[1];
    c[8] = s[0] * s[1] * s[1];
    c[9] = s[1] * s[1] * s[1];
    c[10] = s[0] * s[0] * s[0] * s[0];
    c[11] = s[0] * s[0] * s[0] * s[1];
    c[12] = s[0] * s[0] * s[1] * s[1];
    c[13] = s[0] * s[1] * s[1] * s[1];
    c[14] = s[1] * s[1] * s[1] * s[1];
    Float e = (g(s) - v);
    for (int i = 0; i < 15; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order4_Dim2_luminance::g_then_sgd(const Eigen::VectorXd &s, Float v,
                                             const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[35];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0] * s[0];
    c[5] = s[0] * s[1];
    c[6] = s[0] * s[2];
    c[7] = s[1] * s[1];
    c[8] = s[1] * s[2];
    c[9] = s[2] * s[2];
    c[10] = s[0] * s[0] * s[0];
    c[11] = s[0] * s[0] * s[1];
    c[12] = s[0] * s[0] * s[2];
    c[13] = s[0] * s[1] * s[1];
    c[14] = s[0] * s[1] * s[2];
    c[15] = s[0] * s[2] * s[2];
    c[16] = s[1] * s[1] * s[1];
    c[17] = s[1] * s[1] * s[2];
    c[18] = s[1] * s[2] * s[2];
    c[19] = s[2] * s[2] * s[2];
    c[20] = s[0] * s[0] * s[0] * s[0];
    c[21] = s[0] * s[0] * s[0] * s[1];
    c[22] = s[0] * s[0] * s[0] * s[2];
    c[23] = s[0] * s[0] * s[1] * s[1];
    c[24] = s[0] * s[0] * s[1] * s[2];
    c[25] = s[0] * s[0] * s[2] * s[2];
    c[26] = s[0] * s[1] * s[1] * s[1];
    c[27] = s[0] * s[1] * s[1] * s[2];
    c[28] = s[0] * s[1] * s[2] * s[2];
    c[29] = s[0] * s[2] * s[2] * s[2];
    c[30] = s[1] * s[1] * s[1] * s[1];
    c[31] = s[1] * s[1] * s[1] * s[2];
    c[32] = s[1] * s[1] * s[2] * s[2];
    c[33] = s[1] * s[2] * s[2] * s[2];
    c[34] = s[2] * s[2] * s[2] * s[2];
    for (int i = 1; i < 15; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 15; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}

////Order 5
void Poly_Order5_Dim2_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order5_Dim2_luminance::update(const Eigen::VectorXd &s,
                                        const Spectrum &fx) {
    float l = fx.y();

    Float c[21];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    c[6] = s[0] * s[0] * s[0];
    c[7] = s[0] * s[0] * s[1];
    c[8] = s[0] * s[1] * s[1];
    c[9] = s[1] * s[1] * s[1];
    c[10] = s[0] * s[0] * s[0] * s[0];
    c[11] = s[0] * s[0] * s[0] * s[1];
    c[12] = s[0] * s[0] * s[1] * s[1];
    c[13] = s[0] * s[1] * s[1] * s[1];
    c[14] = s[1] * s[1] * s[1] * s[1];
    c[15] = s[0] * s[0] * s[0] * s[0] * s[0];
    c[16] = s[0] * s[0] * s[0] * s[0] * s[1];
    c[17] = s[0] * s[0] * s[0] * s[1] * s[1];
    c[18] = s[0] * s[0] * s[1] * s[1] * s[1];
    c[19] = s[0] * s[1] * s[1] * s[1] * s[1];
    c[20] = s[1] * s[1] * s[1] * s[1] * s[1];
    for (int i = 0; i < 21; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 21; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order5_Dim2_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order5_Dim2_luminance::G() const {
    float v = X[0];
    int div[21];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 3;
    div[4] = 4;
    div[5] = 3;
    div[6] = 4;
    div[7] = 6;
    div[8] = 6;
    div[9] = 4;
    div[10] = 5;
    div[11] = 8;
    div[12] = 9;
    div[13] = 8;
    div[14] = 5;
    div[15] = 6;
    div[16] = 10;
    div[17] = 12;
    div[18] = 12;
    div[19] = 10;
    div[20] = 6;
    for (int i = 1; i < 21; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order5_Dim2_luminance::g(const Eigen::VectorXd &s) const {
    float v = X[0];
    Float c[21];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    c[6] = s[0] * s[0] * s[0];
    c[7] = s[0] * s[0] * s[1];
    c[8] = s[0] * s[1] * s[1];
    c[9] = s[1] * s[1] * s[1];
    c[10] = s[0] * s[0] * s[0] * s[0];
    c[11] = s[0] * s[0] * s[0] * s[1];
    c[12] = s[0] * s[0] * s[1] * s[1];
    c[13] = s[0] * s[1] * s[1] * s[1];
    c[14] = s[1] * s[1] * s[1] * s[1];
    c[15] = s[0] * s[0] * s[0] * s[0] * s[0];
    c[16] = s[0] * s[0] * s[0] * s[0] * s[1];
    c[17] = s[0] * s[0] * s[0] * s[1] * s[1];
    c[18] = s[0] * s[0] * s[1] * s[1] * s[1];
    c[19] = s[0] * s[1] * s[1] * s[1] * s[1];
    c[20] = s[1] * s[1] * s[1] * s[1] * s[1];
    for (int i = 1; i < 21; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order5_Dim2_luminance::sgd(const Eigen::VectorXd &s, Float v,
                                     const Float lr) {
    Float c[21];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    c[6] = s[0] * s[0] * s[0];
    c[7] = s[0] * s[0] * s[1];
    c[8] = s[0] * s[1] * s[1];
    c[9] = s[1] * s[1] * s[1];
    c[10] = s[0] * s[0] * s[0] * s[0];
    c[11] = s[0] * s[0] * s[0] * s[1];
    c[12] = s[0] * s[0] * s[1] * s[1];
    c[13] = s[0] * s[1] * s[1] * s[1];
    c[14] = s[1] * s[1] * s[1] * s[1];
    c[15] = s[0] * s[0] * s[0] * s[0] * s[0];
    c[16] = s[0] * s[0] * s[0] * s[0] * s[1];
    c[17] = s[0] * s[0] * s[0] * s[1] * s[1];
    c[18] = s[0] * s[0] * s[1] * s[1] * s[1];
    c[19] = s[0] * s[1] * s[1] * s[1] * s[1];
    c[20] = s[1] * s[1] * s[1] * s[1] * s[1];
    Float e = (g(s) - v);
    for (int i = 0; i < 21; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order5_Dim2_luminance::g_then_sgd(const Eigen::VectorXd &s, Float v,
                                             const Float lr) {
    // Evaluation g
    float g = X[0];
    Float c[21];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[0] * s[0];
    c[4] = s[0] * s[1];
    c[5] = s[1] * s[1];
    c[6] = s[0] * s[0] * s[0];
    c[7] = s[0] * s[0] * s[1];
    c[8] = s[0] * s[1] * s[1];
    c[9] = s[1] * s[1] * s[1];
    c[10] = s[0] * s[0] * s[0] * s[0];
    c[11] = s[0] * s[0] * s[0] * s[1];
    c[12] = s[0] * s[0] * s[1] * s[1];
    c[13] = s[0] * s[1] * s[1] * s[1];
    c[14] = s[1] * s[1] * s[1] * s[1];
    c[15] = s[0] * s[0] * s[0] * s[0] * s[0];
    c[16] = s[0] * s[0] * s[0] * s[0] * s[1];
    c[17] = s[0] * s[0] * s[0] * s[1] * s[1];
    c[18] = s[0] * s[0] * s[1] * s[1] * s[1];
    c[19] = s[0] * s[1] * s[1] * s[1] * s[1];
    c[20] = s[1] * s[1] * s[1] * s[1] * s[1];
    for (int i = 1; i < 21; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 21; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}

}  // namespace pbrt