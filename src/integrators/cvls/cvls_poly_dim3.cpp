#include "cvls_poly_dim3.h"

namespace pbrt {

void Poly_Order0_Dim3_luminance::reset() {
    A.setZero();
    b.setZero();
    grads.setZero();
    X.setZero();
}
void Poly_Order0_Dim3_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
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
void Poly_Order0_Dim3_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order0_Dim3_luminance::G() const {
    float v = X[0];
    int div[1];
    div[0] = 1;
    for (int i = 1; i < 1; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order0_Dim3_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[1];
    c[0] = 1.0;
    for (int i = 1; i < 1; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order0_Dim3_luminance::g_then_sgd_grad( const Eigen::VectorXd &s, Float v) {
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
        grads(i) += 2 * c[i] * e;
    }
}
void Poly_Order0_Dim3_luminance::apply_grad(const Float lr) {
    X -= lr * grads;
    grads.setZero();
}
void Poly_Order0_Dim3_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[1];
    c[0] = 1.0;
    Float e = (g(s) - v);
    for (int i = 0; i < 1; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order0_Dim3_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
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
void Poly_Order1_Dim3_luminance::reset() {
    A.setZero();
    b.setZero();
    grads.setZero();
    X.setZero();
}
void Poly_Order1_Dim3_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[4];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    for (int i = 0; i < 4; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 4; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order1_Dim3_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order1_Dim3_luminance::G() const {
    float v = X[0];
    int div[4];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    for (int i = 1; i < 4; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order1_Dim3_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[4];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    for (int i = 1; i < 4; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order1_Dim3_luminance::g_then_sgd_grad( const Eigen::VectorXd &s, Float v) {
    // Evaluation g
    float g = X[0];
    float c[4];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    for (int i = 1; i < 4; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 4; ++i) {
        grads(i) += 2 * c[i] * e;
    }
}
void Poly_Order1_Dim3_luminance::apply_grad(const Float lr) {
    X -= lr * grads;
    grads.setZero();
}
void Poly_Order1_Dim3_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[4];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    Float e = (g(s) - v);
    for (int i = 0; i < 4; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order1_Dim3_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[4];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    for (int i = 1; i < 4; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 4; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}
void Poly_Order2_Dim3_luminance::reset() {
    A.setZero();
    b.setZero();
    grads.setZero();
    X.setZero();
}
void Poly_Order2_Dim3_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[10];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    for (int i = 0; i < 10; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 10; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order2_Dim3_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order2_Dim3_luminance::G() const {
    float v = X[0];
    int div[10];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 3;
    div[5] = 4;
    div[6] = 4;
    div[7] = 3;
    div[8] = 4;
    div[9] = 3;
    for (int i = 1; i < 10; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order2_Dim3_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[10];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    for (int i = 1; i < 10; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order2_Dim3_luminance::g_then_sgd_grad( const Eigen::VectorXd &s, Float v) {
    // Evaluation g
    float g = X[0];
    float c[10];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    for (int i = 1; i < 10; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 10; ++i) {
        grads(i) += 2 * c[i] * e;
    }
}
void Poly_Order2_Dim3_luminance::apply_grad(const Float lr) {
    X -= lr * grads;
    grads.setZero();
}
void Poly_Order2_Dim3_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[10];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    Float e = (g(s) - v);
    for (int i = 0; i < 10; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order2_Dim3_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[10];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
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
void Poly_Order3_Dim3_luminance::reset() {
    A.setZero();
    b.setZero();
    grads.setZero();
    X.setZero();
}
void Poly_Order3_Dim3_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[20];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    for (int i = 0; i < 20; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 20; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order3_Dim3_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order3_Dim3_luminance::G() const {
    float v = X[0];
    int div[20];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 3;
    div[5] = 4;
    div[6] = 4;
    div[7] = 3;
    div[8] = 4;
    div[9] = 3;
    div[10] = 4;
    div[11] = 6;
    div[12] = 6;
    div[13] = 6;
    div[14] = 8;
    div[15] = 6;
    div[16] = 4;
    div[17] = 6;
    div[18] = 6;
    div[19] = 4;
    for (int i = 1; i < 20; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order3_Dim3_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[20];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    for (int i = 1; i < 20; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order3_Dim3_luminance::g_then_sgd_grad( const Eigen::VectorXd &s, Float v) {
    // Evaluation g
    float g = X[0];
    float c[20];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    for (int i = 1; i < 20; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 20; ++i) {
        grads(i) += 2 * c[i] * e;
    }
}
void Poly_Order3_Dim3_luminance::apply_grad(const Float lr) {
    X -= lr * grads;
    grads.setZero();
}
void Poly_Order3_Dim3_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[20];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    Float e = (g(s) - v);
    for (int i = 0; i < 20; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order3_Dim3_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[20];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    for (int i = 1; i < 20; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 20; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}
void Poly_Order4_Dim3_luminance::reset() {
    A.setZero();
    b.setZero();
    grads.setZero();
    X.setZero();
}
void Poly_Order4_Dim3_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[35];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    for (int i = 0; i < 35; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 35; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order4_Dim3_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order4_Dim3_luminance::G() const {
    float v = X[0];
    int div[35];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 3;
    div[5] = 4;
    div[6] = 4;
    div[7] = 3;
    div[8] = 4;
    div[9] = 3;
    div[10] = 4;
    div[11] = 6;
    div[12] = 6;
    div[13] = 6;
    div[14] = 8;
    div[15] = 6;
    div[16] = 4;
    div[17] = 6;
    div[18] = 6;
    div[19] = 4;
    div[20] = 5;
    div[21] = 8;
    div[22] = 8;
    div[23] = 9;
    div[24] = 12;
    div[25] = 9;
    div[26] = 8;
    div[27] = 12;
    div[28] = 12;
    div[29] = 8;
    div[30] = 5;
    div[31] = 8;
    div[32] = 9;
    div[33] = 8;
    div[34] = 5;
    for (int i = 1; i < 35; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order4_Dim3_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[35];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    for (int i = 1; i < 35; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order4_Dim3_luminance::g_then_sgd_grad( const Eigen::VectorXd &s, Float v) {
    // Evaluation g
    float g = X[0];
    float c[35];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    for (int i = 1; i < 35; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 35; ++i) {
        grads(i) += 2 * c[i] * e;
    }
}
void Poly_Order4_Dim3_luminance::apply_grad(const Float lr) {
    X -= lr * grads;
    grads.setZero();
}
void Poly_Order4_Dim3_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[35];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    Float e = (g(s) - v);
    for (int i = 0; i < 35; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order4_Dim3_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[35];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    for (int i = 1; i < 35; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 35; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}
void Poly_Order5_Dim3_luminance::reset() {
    A.setZero();
    b.setZero();
    grads.setZero();
    X.setZero();
}
void Poly_Order5_Dim3_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[56];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    for (int i = 0; i < 56; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 56; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order5_Dim3_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order5_Dim3_luminance::G() const {
    float v = X[0];
    int div[56];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 3;
    div[5] = 4;
    div[6] = 4;
    div[7] = 3;
    div[8] = 4;
    div[9] = 3;
    div[10] = 4;
    div[11] = 6;
    div[12] = 6;
    div[13] = 6;
    div[14] = 8;
    div[15] = 6;
    div[16] = 4;
    div[17] = 6;
    div[18] = 6;
    div[19] = 4;
    div[20] = 5;
    div[21] = 8;
    div[22] = 8;
    div[23] = 9;
    div[24] = 12;
    div[25] = 9;
    div[26] = 8;
    div[27] = 12;
    div[28] = 12;
    div[29] = 8;
    div[30] = 5;
    div[31] = 8;
    div[32] = 9;
    div[33] = 8;
    div[34] = 5;
    div[35] = 6;
    div[36] = 10;
    div[37] = 10;
    div[38] = 12;
    div[39] = 16;
    div[40] = 12;
    div[41] = 12;
    div[42] = 18;
    div[43] = 18;
    div[44] = 12;
    div[45] = 10;
    div[46] = 16;
    div[47] = 18;
    div[48] = 16;
    div[49] = 10;
    div[50] = 6;
    div[51] = 10;
    div[52] = 12;
    div[53] = 12;
    div[54] = 10;
    div[55] = 6;
    for (int i = 1; i < 56; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order5_Dim3_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[56];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    for (int i = 1; i < 56; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order5_Dim3_luminance::g_then_sgd_grad( const Eigen::VectorXd &s, Float v) {
    // Evaluation g
    float g = X[0];
    float c[56];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    for (int i = 1; i < 56; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 56; ++i) {
        grads(i) += 2 * c[i] * e;
    }
}
void Poly_Order5_Dim3_luminance::apply_grad(const Float lr) {
    X -= lr * grads;
    grads.setZero();
}
void Poly_Order5_Dim3_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[56];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    Float e = (g(s) - v);
    for (int i = 0; i < 56; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order5_Dim3_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[56];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    for (int i = 1; i < 56; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 56; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}
void Poly_Order6_Dim3_luminance::reset() {
    A.setZero();
    b.setZero();
    grads.setZero();
    X.setZero();
}
void Poly_Order6_Dim3_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[84];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    c[56] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[57] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[58] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[59] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[60] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[61] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[62] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[63] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[64] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[65] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[66] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[67] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[68] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[69] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[70] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[71] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[72] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[73] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[74] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[75] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[76] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[77] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[78] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[79] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[80] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[81] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[82] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[83] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    for (int i = 0; i < 84; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 84; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order6_Dim3_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order6_Dim3_luminance::G() const {
    float v = X[0];
    int div[84];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 3;
    div[5] = 4;
    div[6] = 4;
    div[7] = 3;
    div[8] = 4;
    div[9] = 3;
    div[10] = 4;
    div[11] = 6;
    div[12] = 6;
    div[13] = 6;
    div[14] = 8;
    div[15] = 6;
    div[16] = 4;
    div[17] = 6;
    div[18] = 6;
    div[19] = 4;
    div[20] = 5;
    div[21] = 8;
    div[22] = 8;
    div[23] = 9;
    div[24] = 12;
    div[25] = 9;
    div[26] = 8;
    div[27] = 12;
    div[28] = 12;
    div[29] = 8;
    div[30] = 5;
    div[31] = 8;
    div[32] = 9;
    div[33] = 8;
    div[34] = 5;
    div[35] = 6;
    div[36] = 10;
    div[37] = 10;
    div[38] = 12;
    div[39] = 16;
    div[40] = 12;
    div[41] = 12;
    div[42] = 18;
    div[43] = 18;
    div[44] = 12;
    div[45] = 10;
    div[46] = 16;
    div[47] = 18;
    div[48] = 16;
    div[49] = 10;
    div[50] = 6;
    div[51] = 10;
    div[52] = 12;
    div[53] = 12;
    div[54] = 10;
    div[55] = 6;
    div[56] = 7;
    div[57] = 12;
    div[58] = 12;
    div[59] = 15;
    div[60] = 20;
    div[61] = 15;
    div[62] = 16;
    div[63] = 24;
    div[64] = 24;
    div[65] = 16;
    div[66] = 15;
    div[67] = 24;
    div[68] = 27;
    div[69] = 24;
    div[70] = 15;
    div[71] = 12;
    div[72] = 20;
    div[73] = 24;
    div[74] = 24;
    div[75] = 20;
    div[76] = 12;
    div[77] = 7;
    div[78] = 12;
    div[79] = 15;
    div[80] = 16;
    div[81] = 15;
    div[82] = 12;
    div[83] = 7;
    for (int i = 1; i < 84; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order6_Dim3_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[84];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    c[56] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[57] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[58] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[59] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[60] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[61] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[62] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[63] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[64] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[65] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[66] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[67] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[68] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[69] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[70] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[71] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[72] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[73] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[74] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[75] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[76] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[77] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[78] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[79] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[80] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[81] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[82] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[83] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    for (int i = 1; i < 84; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order6_Dim3_luminance::g_then_sgd_grad( const Eigen::VectorXd &s, Float v) {
    // Evaluation g
    float g = X[0];
    float c[84];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    c[56] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[57] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[58] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[59] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[60] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[61] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[62] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[63] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[64] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[65] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[66] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[67] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[68] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[69] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[70] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[71] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[72] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[73] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[74] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[75] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[76] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[77] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[78] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[79] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[80] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[81] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[82] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[83] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    for (int i = 1; i < 84; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 84; ++i) {
        grads(i) += 2 * c[i] * e;
    }
}
void Poly_Order6_Dim3_luminance::apply_grad(const Float lr) {
    X -= lr * grads;
    grads.setZero();
}
void Poly_Order6_Dim3_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[84];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    c[56] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[57] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[58] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[59] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[60] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[61] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[62] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[63] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[64] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[65] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[66] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[67] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[68] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[69] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[70] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[71] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[72] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[73] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[74] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[75] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[76] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[77] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[78] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[79] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[80] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[81] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[82] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[83] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    Float e = (g(s) - v);
    for (int i = 0; i < 84; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order6_Dim3_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[84];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    c[56] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[57] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[58] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[59] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[60] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[61] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[62] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[63] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[64] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[65] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[66] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[67] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[68] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[69] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[70] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[71] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[72] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[73] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[74] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[75] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[76] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[77] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[78] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[79] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[80] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[81] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[82] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[83] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    for (int i = 1; i < 84; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 84; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}
void Poly_Order7_Dim3_luminance::reset() {
    A.setZero();
    b.setZero();
    grads.setZero();
    X.setZero();
}
void Poly_Order7_Dim3_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[120];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    c[56] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[57] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[58] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[59] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[60] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[61] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[62] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[63] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[64] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[65] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[66] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[67] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[68] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[69] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[70] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[71] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[72] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[73] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[74] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[75] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[76] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[77] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[78] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[79] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[80] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[81] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[82] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[83] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[84] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[85] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[86] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[87] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[88] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[89] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[90] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[91] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[92] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[93] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[94] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[95] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[96] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[97] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[98] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[99] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[100] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[101] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[102] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[103] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[104] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[105] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[106] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[107] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[108] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[109] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[110] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[111] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[112] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[113] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[114] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[115] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[116] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[117] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[118] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[119] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    for (int i = 0; i < 120; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 120; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order7_Dim3_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order7_Dim3_luminance::G() const {
    float v = X[0];
    int div[120];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 3;
    div[5] = 4;
    div[6] = 4;
    div[7] = 3;
    div[8] = 4;
    div[9] = 3;
    div[10] = 4;
    div[11] = 6;
    div[12] = 6;
    div[13] = 6;
    div[14] = 8;
    div[15] = 6;
    div[16] = 4;
    div[17] = 6;
    div[18] = 6;
    div[19] = 4;
    div[20] = 5;
    div[21] = 8;
    div[22] = 8;
    div[23] = 9;
    div[24] = 12;
    div[25] = 9;
    div[26] = 8;
    div[27] = 12;
    div[28] = 12;
    div[29] = 8;
    div[30] = 5;
    div[31] = 8;
    div[32] = 9;
    div[33] = 8;
    div[34] = 5;
    div[35] = 6;
    div[36] = 10;
    div[37] = 10;
    div[38] = 12;
    div[39] = 16;
    div[40] = 12;
    div[41] = 12;
    div[42] = 18;
    div[43] = 18;
    div[44] = 12;
    div[45] = 10;
    div[46] = 16;
    div[47] = 18;
    div[48] = 16;
    div[49] = 10;
    div[50] = 6;
    div[51] = 10;
    div[52] = 12;
    div[53] = 12;
    div[54] = 10;
    div[55] = 6;
    div[56] = 7;
    div[57] = 12;
    div[58] = 12;
    div[59] = 15;
    div[60] = 20;
    div[61] = 15;
    div[62] = 16;
    div[63] = 24;
    div[64] = 24;
    div[65] = 16;
    div[66] = 15;
    div[67] = 24;
    div[68] = 27;
    div[69] = 24;
    div[70] = 15;
    div[71] = 12;
    div[72] = 20;
    div[73] = 24;
    div[74] = 24;
    div[75] = 20;
    div[76] = 12;
    div[77] = 7;
    div[78] = 12;
    div[79] = 15;
    div[80] = 16;
    div[81] = 15;
    div[82] = 12;
    div[83] = 7;
    div[84] = 8;
    div[85] = 14;
    div[86] = 14;
    div[87] = 18;
    div[88] = 24;
    div[89] = 18;
    div[90] = 20;
    div[91] = 30;
    div[92] = 30;
    div[93] = 20;
    div[94] = 20;
    div[95] = 32;
    div[96] = 36;
    div[97] = 32;
    div[98] = 20;
    div[99] = 18;
    div[100] = 30;
    div[101] = 36;
    div[102] = 36;
    div[103] = 30;
    div[104] = 18;
    div[105] = 14;
    div[106] = 24;
    div[107] = 30;
    div[108] = 32;
    div[109] = 30;
    div[110] = 24;
    div[111] = 14;
    div[112] = 8;
    div[113] = 14;
    div[114] = 18;
    div[115] = 20;
    div[116] = 20;
    div[117] = 18;
    div[118] = 14;
    div[119] = 8;
    for (int i = 1; i < 120; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order7_Dim3_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[120];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    c[56] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[57] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[58] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[59] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[60] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[61] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[62] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[63] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[64] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[65] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[66] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[67] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[68] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[69] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[70] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[71] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[72] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[73] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[74] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[75] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[76] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[77] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[78] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[79] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[80] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[81] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[82] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[83] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[84] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[85] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[86] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[87] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[88] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[89] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[90] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[91] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[92] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[93] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[94] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[95] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[96] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[97] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[98] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[99] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[100] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[101] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[102] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[103] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[104] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[105] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[106] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[107] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[108] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[109] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[110] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[111] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[112] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[113] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[114] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[115] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[116] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[117] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[118] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[119] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    for (int i = 1; i < 120; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order7_Dim3_luminance::g_then_sgd_grad( const Eigen::VectorXd &s, Float v) {
    // Evaluation g
    float g = X[0];
    float c[120];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    c[56] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[57] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[58] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[59] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[60] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[61] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[62] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[63] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[64] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[65] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[66] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[67] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[68] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[69] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[70] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[71] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[72] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[73] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[74] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[75] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[76] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[77] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[78] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[79] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[80] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[81] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[82] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[83] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[84] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[85] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[86] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[87] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[88] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[89] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[90] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[91] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[92] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[93] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[94] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[95] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[96] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[97] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[98] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[99] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[100] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[101] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[102] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[103] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[104] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[105] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[106] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[107] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[108] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[109] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[110] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[111] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[112] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[113] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[114] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[115] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[116] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[117] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[118] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[119] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    for (int i = 1; i < 120; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 120; ++i) {
        grads(i) += 2 * c[i] * e;
    }
}
void Poly_Order7_Dim3_luminance::apply_grad(const Float lr) {
    X -= lr * grads;
    grads.setZero();
}
void Poly_Order7_Dim3_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[120];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    c[56] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[57] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[58] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[59] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[60] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[61] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[62] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[63] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[64] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[65] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[66] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[67] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[68] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[69] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[70] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[71] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[72] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[73] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[74] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[75] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[76] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[77] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[78] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[79] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[80] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[81] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[82] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[83] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[84] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[85] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[86] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[87] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[88] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[89] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[90] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[91] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[92] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[93] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[94] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[95] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[96] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[97] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[98] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[99] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[100] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[101] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[102] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[103] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[104] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[105] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[106] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[107] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[108] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[109] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[110] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[111] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[112] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[113] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[114] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[115] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[116] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[117] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[118] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[119] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    Float e = (g(s) - v);
    for (int i = 0; i < 120; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order7_Dim3_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[120];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[0]*s[0];
    c[5] = s[0]*s[1];
    c[6] = s[0]*s[2];
    c[7] = s[1]*s[1];
    c[8] = s[1]*s[2];
    c[9] = s[2]*s[2];
    c[10] = s[0]*s[0]*s[0];
    c[11] = s[0]*s[0]*s[1];
    c[12] = s[0]*s[0]*s[2];
    c[13] = s[0]*s[1]*s[1];
    c[14] = s[0]*s[1]*s[2];
    c[15] = s[0]*s[2]*s[2];
    c[16] = s[1]*s[1]*s[1];
    c[17] = s[1]*s[1]*s[2];
    c[18] = s[1]*s[2]*s[2];
    c[19] = s[2]*s[2]*s[2];
    c[20] = s[0]*s[0]*s[0]*s[0];
    c[21] = s[0]*s[0]*s[0]*s[1];
    c[22] = s[0]*s[0]*s[0]*s[2];
    c[23] = s[0]*s[0]*s[1]*s[1];
    c[24] = s[0]*s[0]*s[1]*s[2];
    c[25] = s[0]*s[0]*s[2]*s[2];
    c[26] = s[0]*s[1]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[2]*s[2];
    c[29] = s[0]*s[2]*s[2]*s[2];
    c[30] = s[1]*s[1]*s[1]*s[1];
    c[31] = s[1]*s[1]*s[1]*s[2];
    c[32] = s[1]*s[1]*s[2]*s[2];
    c[33] = s[1]*s[2]*s[2]*s[2];
    c[34] = s[2]*s[2]*s[2]*s[2];
    c[35] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[39] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[40] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[42] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[43] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[44] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[45] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[48] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[49] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[50] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[51] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[52] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[53] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[54] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[55] = s[2]*s[2]*s[2]*s[2]*s[2];
    c[56] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[57] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[58] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[59] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[60] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[61] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[62] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[63] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[64] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[65] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[66] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[67] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[68] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[69] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[70] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[71] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[72] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[73] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[74] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[75] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[76] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[77] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[78] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[79] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[80] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[81] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[82] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[83] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[84] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[0];
    c[85] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[1];
    c[86] = s[0]*s[0]*s[0]*s[0]*s[0]*s[0]*s[2];
    c[87] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1]*s[1];
    c[88] = s[0]*s[0]*s[0]*s[0]*s[0]*s[1]*s[2];
    c[89] = s[0]*s[0]*s[0]*s[0]*s[0]*s[2]*s[2];
    c[90] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1]*s[1];
    c[91] = s[0]*s[0]*s[0]*s[0]*s[1]*s[1]*s[2];
    c[92] = s[0]*s[0]*s[0]*s[0]*s[1]*s[2]*s[2];
    c[93] = s[0]*s[0]*s[0]*s[0]*s[2]*s[2]*s[2];
    c[94] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1]*s[1];
    c[95] = s[0]*s[0]*s[0]*s[1]*s[1]*s[1]*s[2];
    c[96] = s[0]*s[0]*s[0]*s[1]*s[1]*s[2]*s[2];
    c[97] = s[0]*s[0]*s[0]*s[1]*s[2]*s[2]*s[2];
    c[98] = s[0]*s[0]*s[0]*s[2]*s[2]*s[2]*s[2];
    c[99] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[100] = s[0]*s[0]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[101] = s[0]*s[0]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[102] = s[0]*s[0]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[103] = s[0]*s[0]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[104] = s[0]*s[0]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[105] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[106] = s[0]*s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[107] = s[0]*s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[108] = s[0]*s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[109] = s[0]*s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[110] = s[0]*s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[111] = s[0]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[112] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1]*s[1];
    c[113] = s[1]*s[1]*s[1]*s[1]*s[1]*s[1]*s[2];
    c[114] = s[1]*s[1]*s[1]*s[1]*s[1]*s[2]*s[2];
    c[115] = s[1]*s[1]*s[1]*s[1]*s[2]*s[2]*s[2];
    c[116] = s[1]*s[1]*s[1]*s[2]*s[2]*s[2]*s[2];
    c[117] = s[1]*s[1]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[118] = s[1]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    c[119] = s[2]*s[2]*s[2]*s[2]*s[2]*s[2]*s[2];
    for (int i = 1; i < 120; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 120; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}



}