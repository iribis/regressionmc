#include "cvls_poly_dim4.h"

namespace pbrt {

void Poly_Order0_Dim4_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order0_Dim4_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
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
void Poly_Order0_Dim4_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order0_Dim4_luminance::G() const {
    float v = X[0];
    int div[1];
    div[0] = 1;
    for (int i = 1; i < 1; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order0_Dim4_luminance::Gpix(Float ax, Float ay, Float bx, Float by) const {
    float v = X[0];
    float f[1];
    f[0] = ((bx - ax)*(by - ay)) / 1;
    for (int i = 1; i < 1; ++i) {
        v += X[i] * f[i];
    }
    return v;
}
float Poly_Order0_Dim4_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[1];
    c[0] = 1.0;
    for (int i = 1; i < 1; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order0_Dim4_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[1];
    c[0] = 1.0;
    Float e = (g(s) - v);
    for (int i = 0; i < 1; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order0_Dim4_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
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
void Poly_Order1_Dim4_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order1_Dim4_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[5];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    for (int i = 0; i < 5; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 5; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order1_Dim4_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order1_Dim4_luminance::G() const {
    float v = X[0];
    int div[5];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 2;
    for (int i = 1; i < 5; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order1_Dim4_luminance::Gpix(Float ax, Float ay, Float bx, Float by) const {
    float v = X[0];
    float f[5];
    f[0] = ((bx - ax)*(by - ay)) / 1;
    f[1] = ((bx*bx - ax*ax)*(by - ay)) / 2;
    f[2] = ((bx - ax)*(by*by - ay*ay)) / 2;
    f[3] = ((bx - ax)*(by - ay)) / 2;
    f[4] = ((bx - ax)*(by - ay)) / 2;
    for (int i = 1; i < 5; ++i) {
        v += X[i] * f[i];
    }
    return v;
}
float Poly_Order1_Dim4_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[5];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    for (int i = 1; i < 5; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order1_Dim4_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[5];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    Float e = (g(s) - v);
    for (int i = 0; i < 5; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order1_Dim4_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[5];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    for (int i = 1; i < 5; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 5; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}
void Poly_Order2_Dim4_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order2_Dim4_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[15];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    for (int i = 0; i < 15; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 15; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order2_Dim4_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order2_Dim4_luminance::G() const {
    float v = X[0];
    int div[15];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 2;
    div[5] = 3;
    div[6] = 4;
    div[7] = 4;
    div[8] = 4;
    div[9] = 3;
    div[10] = 4;
    div[11] = 4;
    div[12] = 3;
    div[13] = 4;
    div[14] = 3;
    for (int i = 1; i < 15; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order2_Dim4_luminance::Gpix(Float ax, Float ay, Float bx, Float by) const {
    float v = X[0];
    float f[15];
    f[0] = ((bx - ax)*(by - ay)) / 1;
    f[1] = ((bx*bx - ax*ax)*(by - ay)) / 2;
    f[2] = ((bx - ax)*(by*by - ay*ay)) / 2;
    f[3] = ((bx - ax)*(by - ay)) / 2;
    f[4] = ((bx - ax)*(by - ay)) / 2;
    f[5] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 3;
    f[6] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 4;
    f[7] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[8] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[9] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 3;
    f[10] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[11] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[12] = ((bx - ax)*(by - ay)) / 3;
    f[13] = ((bx - ax)*(by - ay)) / 4;
    f[14] = ((bx - ax)*(by - ay)) / 3;
    for (int i = 1; i < 15; ++i) {
        v += X[i] * f[i];
    }
    return v;
}
float Poly_Order2_Dim4_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[15];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    for (int i = 1; i < 15; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order2_Dim4_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[15];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    Float e = (g(s) - v);
    for (int i = 0; i < 15; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order2_Dim4_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[15];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
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
void Poly_Order3_Dim4_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order3_Dim4_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[35];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    c[15] = s[0]*s[0]*s[0];
    c[16] = s[0]*s[0]*s[1];
    c[17] = s[0]*s[0]*s[2];
    c[18] = s[0]*s[0]*s[3];
    c[19] = s[0]*s[1]*s[1];
    c[20] = s[0]*s[1]*s[2];
    c[21] = s[0]*s[1]*s[3];
    c[22] = s[0]*s[2]*s[2];
    c[23] = s[0]*s[2]*s[3];
    c[24] = s[0]*s[3]*s[3];
    c[25] = s[1]*s[1]*s[1];
    c[26] = s[1]*s[1]*s[2];
    c[27] = s[1]*s[1]*s[3];
    c[28] = s[1]*s[2]*s[2];
    c[29] = s[1]*s[2]*s[3];
    c[30] = s[1]*s[3]*s[3];
    c[31] = s[2]*s[2]*s[2];
    c[32] = s[2]*s[2]*s[3];
    c[33] = s[2]*s[3]*s[3];
    c[34] = s[3]*s[3]*s[3];
    for (int i = 0; i < 35; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 35; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order3_Dim4_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order3_Dim4_luminance::G() const {
    float v = X[0];
    int div[35];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 2;
    div[5] = 3;
    div[6] = 4;
    div[7] = 4;
    div[8] = 4;
    div[9] = 3;
    div[10] = 4;
    div[11] = 4;
    div[12] = 3;
    div[13] = 4;
    div[14] = 3;
    div[15] = 4;
    div[16] = 6;
    div[17] = 6;
    div[18] = 6;
    div[19] = 6;
    div[20] = 8;
    div[21] = 8;
    div[22] = 6;
    div[23] = 8;
    div[24] = 6;
    div[25] = 4;
    div[26] = 6;
    div[27] = 6;
    div[28] = 6;
    div[29] = 8;
    div[30] = 6;
    div[31] = 4;
    div[32] = 6;
    div[33] = 6;
    div[34] = 4;
    for (int i = 1; i < 35; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order3_Dim4_luminance::Gpix(Float ax, Float ay, Float bx, Float by) const {
    float v = X[0];
    float f[35];
    f[0] = ((bx - ax)*(by - ay)) / 1;
    f[1] = ((bx*bx - ax*ax)*(by - ay)) / 2;
    f[2] = ((bx - ax)*(by*by - ay*ay)) / 2;
    f[3] = ((bx - ax)*(by - ay)) / 2;
    f[4] = ((bx - ax)*(by - ay)) / 2;
    f[5] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 3;
    f[6] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 4;
    f[7] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[8] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[9] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 3;
    f[10] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[11] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[12] = ((bx - ax)*(by - ay)) / 3;
    f[13] = ((bx - ax)*(by - ay)) / 4;
    f[14] = ((bx - ax)*(by - ay)) / 3;
    f[15] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 4;
    f[16] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 6;
    f[17] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 6;
    f[18] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 6;
    f[19] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 6;
    f[20] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 8;
    f[21] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 8;
    f[22] = ((bx*bx - ax*ax)*(by - ay)) / 6;
    f[23] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[24] = ((bx*bx - ax*ax)*(by - ay)) / 6;
    f[25] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 4;
    f[26] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 6;
    f[27] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 6;
    f[28] = ((bx - ax)*(by*by - ay*ay)) / 6;
    f[29] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[30] = ((bx - ax)*(by*by - ay*ay)) / 6;
    f[31] = ((bx - ax)*(by - ay)) / 4;
    f[32] = ((bx - ax)*(by - ay)) / 6;
    f[33] = ((bx - ax)*(by - ay)) / 6;
    f[34] = ((bx - ax)*(by - ay)) / 4;
    for (int i = 1; i < 35; ++i) {
        v += X[i] * f[i];
    }
    return v;
}
float Poly_Order3_Dim4_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[35];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    c[15] = s[0]*s[0]*s[0];
    c[16] = s[0]*s[0]*s[1];
    c[17] = s[0]*s[0]*s[2];
    c[18] = s[0]*s[0]*s[3];
    c[19] = s[0]*s[1]*s[1];
    c[20] = s[0]*s[1]*s[2];
    c[21] = s[0]*s[1]*s[3];
    c[22] = s[0]*s[2]*s[2];
    c[23] = s[0]*s[2]*s[3];
    c[24] = s[0]*s[3]*s[3];
    c[25] = s[1]*s[1]*s[1];
    c[26] = s[1]*s[1]*s[2];
    c[27] = s[1]*s[1]*s[3];
    c[28] = s[1]*s[2]*s[2];
    c[29] = s[1]*s[2]*s[3];
    c[30] = s[1]*s[3]*s[3];
    c[31] = s[2]*s[2]*s[2];
    c[32] = s[2]*s[2]*s[3];
    c[33] = s[2]*s[3]*s[3];
    c[34] = s[3]*s[3]*s[3];
    for (int i = 1; i < 35; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order3_Dim4_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[35];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    c[15] = s[0]*s[0]*s[0];
    c[16] = s[0]*s[0]*s[1];
    c[17] = s[0]*s[0]*s[2];
    c[18] = s[0]*s[0]*s[3];
    c[19] = s[0]*s[1]*s[1];
    c[20] = s[0]*s[1]*s[2];
    c[21] = s[0]*s[1]*s[3];
    c[22] = s[0]*s[2]*s[2];
    c[23] = s[0]*s[2]*s[3];
    c[24] = s[0]*s[3]*s[3];
    c[25] = s[1]*s[1]*s[1];
    c[26] = s[1]*s[1]*s[2];
    c[27] = s[1]*s[1]*s[3];
    c[28] = s[1]*s[2]*s[2];
    c[29] = s[1]*s[2]*s[3];
    c[30] = s[1]*s[3]*s[3];
    c[31] = s[2]*s[2]*s[2];
    c[32] = s[2]*s[2]*s[3];
    c[33] = s[2]*s[3]*s[3];
    c[34] = s[3]*s[3]*s[3];
    Float e = (g(s) - v);
    for (int i = 0; i < 35; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order3_Dim4_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[35];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    c[15] = s[0]*s[0]*s[0];
    c[16] = s[0]*s[0]*s[1];
    c[17] = s[0]*s[0]*s[2];
    c[18] = s[0]*s[0]*s[3];
    c[19] = s[0]*s[1]*s[1];
    c[20] = s[0]*s[1]*s[2];
    c[21] = s[0]*s[1]*s[3];
    c[22] = s[0]*s[2]*s[2];
    c[23] = s[0]*s[2]*s[3];
    c[24] = s[0]*s[3]*s[3];
    c[25] = s[1]*s[1]*s[1];
    c[26] = s[1]*s[1]*s[2];
    c[27] = s[1]*s[1]*s[3];
    c[28] = s[1]*s[2]*s[2];
    c[29] = s[1]*s[2]*s[3];
    c[30] = s[1]*s[3]*s[3];
    c[31] = s[2]*s[2]*s[2];
    c[32] = s[2]*s[2]*s[3];
    c[33] = s[2]*s[3]*s[3];
    c[34] = s[3]*s[3]*s[3];
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
void Poly_Order4_Dim4_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order4_Dim4_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[70];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    c[15] = s[0]*s[0]*s[0];
    c[16] = s[0]*s[0]*s[1];
    c[17] = s[0]*s[0]*s[2];
    c[18] = s[0]*s[0]*s[3];
    c[19] = s[0]*s[1]*s[1];
    c[20] = s[0]*s[1]*s[2];
    c[21] = s[0]*s[1]*s[3];
    c[22] = s[0]*s[2]*s[2];
    c[23] = s[0]*s[2]*s[3];
    c[24] = s[0]*s[3]*s[3];
    c[25] = s[1]*s[1]*s[1];
    c[26] = s[1]*s[1]*s[2];
    c[27] = s[1]*s[1]*s[3];
    c[28] = s[1]*s[2]*s[2];
    c[29] = s[1]*s[2]*s[3];
    c[30] = s[1]*s[3]*s[3];
    c[31] = s[2]*s[2]*s[2];
    c[32] = s[2]*s[2]*s[3];
    c[33] = s[2]*s[3]*s[3];
    c[34] = s[3]*s[3]*s[3];
    c[35] = s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[3];
    c[39] = s[0]*s[0]*s[1]*s[1];
    c[40] = s[0]*s[0]*s[1]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[3];
    c[42] = s[0]*s[0]*s[2]*s[2];
    c[43] = s[0]*s[0]*s[2]*s[3];
    c[44] = s[0]*s[0]*s[3]*s[3];
    c[45] = s[0]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[3];
    c[48] = s[0]*s[1]*s[2]*s[2];
    c[49] = s[0]*s[1]*s[2]*s[3];
    c[50] = s[0]*s[1]*s[3]*s[3];
    c[51] = s[0]*s[2]*s[2]*s[2];
    c[52] = s[0]*s[2]*s[2]*s[3];
    c[53] = s[0]*s[2]*s[3]*s[3];
    c[54] = s[0]*s[3]*s[3]*s[3];
    c[55] = s[1]*s[1]*s[1]*s[1];
    c[56] = s[1]*s[1]*s[1]*s[2];
    c[57] = s[1]*s[1]*s[1]*s[3];
    c[58] = s[1]*s[1]*s[2]*s[2];
    c[59] = s[1]*s[1]*s[2]*s[3];
    c[60] = s[1]*s[1]*s[3]*s[3];
    c[61] = s[1]*s[2]*s[2]*s[2];
    c[62] = s[1]*s[2]*s[2]*s[3];
    c[63] = s[1]*s[2]*s[3]*s[3];
    c[64] = s[1]*s[3]*s[3]*s[3];
    c[65] = s[2]*s[2]*s[2]*s[2];
    c[66] = s[2]*s[2]*s[2]*s[3];
    c[67] = s[2]*s[2]*s[3]*s[3];
    c[68] = s[2]*s[3]*s[3]*s[3];
    c[69] = s[3]*s[3]*s[3]*s[3];
    for (int i = 0; i < 70; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 70; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order4_Dim4_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order4_Dim4_luminance::G() const {
    float v = X[0];
    int div[70];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 2;
    div[5] = 3;
    div[6] = 4;
    div[7] = 4;
    div[8] = 4;
    div[9] = 3;
    div[10] = 4;
    div[11] = 4;
    div[12] = 3;
    div[13] = 4;
    div[14] = 3;
    div[15] = 4;
    div[16] = 6;
    div[17] = 6;
    div[18] = 6;
    div[19] = 6;
    div[20] = 8;
    div[21] = 8;
    div[22] = 6;
    div[23] = 8;
    div[24] = 6;
    div[25] = 4;
    div[26] = 6;
    div[27] = 6;
    div[28] = 6;
    div[29] = 8;
    div[30] = 6;
    div[31] = 4;
    div[32] = 6;
    div[33] = 6;
    div[34] = 4;
    div[35] = 5;
    div[36] = 8;
    div[37] = 8;
    div[38] = 8;
    div[39] = 9;
    div[40] = 12;
    div[41] = 12;
    div[42] = 9;
    div[43] = 12;
    div[44] = 9;
    div[45] = 8;
    div[46] = 12;
    div[47] = 12;
    div[48] = 12;
    div[49] = 16;
    div[50] = 12;
    div[51] = 8;
    div[52] = 12;
    div[53] = 12;
    div[54] = 8;
    div[55] = 5;
    div[56] = 8;
    div[57] = 8;
    div[58] = 9;
    div[59] = 12;
    div[60] = 9;
    div[61] = 8;
    div[62] = 12;
    div[63] = 12;
    div[64] = 8;
    div[65] = 5;
    div[66] = 8;
    div[67] = 9;
    div[68] = 8;
    div[69] = 5;
    for (int i = 1; i < 70; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order4_Dim4_luminance::Gpix(Float ax, Float ay, Float bx, Float by) const {
    float v = X[0];
    float f[70];
    f[0] = ((bx - ax)*(by - ay)) / 1;
    f[1] = ((bx*bx - ax*ax)*(by - ay)) / 2;
    f[2] = ((bx - ax)*(by*by - ay*ay)) / 2;
    f[3] = ((bx - ax)*(by - ay)) / 2;
    f[4] = ((bx - ax)*(by - ay)) / 2;
    f[5] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 3;
    f[6] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 4;
    f[7] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[8] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[9] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 3;
    f[10] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[11] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[12] = ((bx - ax)*(by - ay)) / 3;
    f[13] = ((bx - ax)*(by - ay)) / 4;
    f[14] = ((bx - ax)*(by - ay)) / 3;
    f[15] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 4;
    f[16] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 6;
    f[17] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 6;
    f[18] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 6;
    f[19] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 6;
    f[20] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 8;
    f[21] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 8;
    f[22] = ((bx*bx - ax*ax)*(by - ay)) / 6;
    f[23] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[24] = ((bx*bx - ax*ax)*(by - ay)) / 6;
    f[25] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 4;
    f[26] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 6;
    f[27] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 6;
    f[28] = ((bx - ax)*(by*by - ay*ay)) / 6;
    f[29] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[30] = ((bx - ax)*(by*by - ay*ay)) / 6;
    f[31] = ((bx - ax)*(by - ay)) / 4;
    f[32] = ((bx - ax)*(by - ay)) / 6;
    f[33] = ((bx - ax)*(by - ay)) / 6;
    f[34] = ((bx - ax)*(by - ay)) / 4;
    f[35] = ((bx*bx*bx*bx*bx - ax*ax*ax*ax*ax)*(by - ay)) / 5;
    f[36] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by*by - ay*ay)) / 8;
    f[37] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 8;
    f[38] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 8;
    f[39] = ((bx*bx*bx - ax*ax*ax)*(by*by*by - ay*ay*ay)) / 9;
    f[40] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 12;
    f[41] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 12;
    f[42] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 9;
    f[43] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 12;
    f[44] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 9;
    f[45] = ((bx*bx - ax*ax)*(by*by*by*by - ay*ay*ay*ay)) / 8;
    f[46] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 12;
    f[47] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 12;
    f[48] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 12;
    f[49] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 16;
    f[50] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 12;
    f[51] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[52] = ((bx*bx - ax*ax)*(by - ay)) / 12;
    f[53] = ((bx*bx - ax*ax)*(by - ay)) / 12;
    f[54] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[55] = ((bx - ax)*(by*by*by*by*by - ay*ay*ay*ay*ay)) / 5;
    f[56] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 8;
    f[57] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 8;
    f[58] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 9;
    f[59] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 12;
    f[60] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 9;
    f[61] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[62] = ((bx - ax)*(by*by - ay*ay)) / 12;
    f[63] = ((bx - ax)*(by*by - ay*ay)) / 12;
    f[64] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[65] = ((bx - ax)*(by - ay)) / 5;
    f[66] = ((bx - ax)*(by - ay)) / 8;
    f[67] = ((bx - ax)*(by - ay)) / 9;
    f[68] = ((bx - ax)*(by - ay)) / 8;
    f[69] = ((bx - ax)*(by - ay)) / 5;
    for (int i = 1; i < 70; ++i) {
        v += X[i] * f[i];
    }
    return v;
}
float Poly_Order4_Dim4_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[70];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    c[15] = s[0]*s[0]*s[0];
    c[16] = s[0]*s[0]*s[1];
    c[17] = s[0]*s[0]*s[2];
    c[18] = s[0]*s[0]*s[3];
    c[19] = s[0]*s[1]*s[1];
    c[20] = s[0]*s[1]*s[2];
    c[21] = s[0]*s[1]*s[3];
    c[22] = s[0]*s[2]*s[2];
    c[23] = s[0]*s[2]*s[3];
    c[24] = s[0]*s[3]*s[3];
    c[25] = s[1]*s[1]*s[1];
    c[26] = s[1]*s[1]*s[2];
    c[27] = s[1]*s[1]*s[3];
    c[28] = s[1]*s[2]*s[2];
    c[29] = s[1]*s[2]*s[3];
    c[30] = s[1]*s[3]*s[3];
    c[31] = s[2]*s[2]*s[2];
    c[32] = s[2]*s[2]*s[3];
    c[33] = s[2]*s[3]*s[3];
    c[34] = s[3]*s[3]*s[3];
    c[35] = s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[3];
    c[39] = s[0]*s[0]*s[1]*s[1];
    c[40] = s[0]*s[0]*s[1]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[3];
    c[42] = s[0]*s[0]*s[2]*s[2];
    c[43] = s[0]*s[0]*s[2]*s[3];
    c[44] = s[0]*s[0]*s[3]*s[3];
    c[45] = s[0]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[3];
    c[48] = s[0]*s[1]*s[2]*s[2];
    c[49] = s[0]*s[1]*s[2]*s[3];
    c[50] = s[0]*s[1]*s[3]*s[3];
    c[51] = s[0]*s[2]*s[2]*s[2];
    c[52] = s[0]*s[2]*s[2]*s[3];
    c[53] = s[0]*s[2]*s[3]*s[3];
    c[54] = s[0]*s[3]*s[3]*s[3];
    c[55] = s[1]*s[1]*s[1]*s[1];
    c[56] = s[1]*s[1]*s[1]*s[2];
    c[57] = s[1]*s[1]*s[1]*s[3];
    c[58] = s[1]*s[1]*s[2]*s[2];
    c[59] = s[1]*s[1]*s[2]*s[3];
    c[60] = s[1]*s[1]*s[3]*s[3];
    c[61] = s[1]*s[2]*s[2]*s[2];
    c[62] = s[1]*s[2]*s[2]*s[3];
    c[63] = s[1]*s[2]*s[3]*s[3];
    c[64] = s[1]*s[3]*s[3]*s[3];
    c[65] = s[2]*s[2]*s[2]*s[2];
    c[66] = s[2]*s[2]*s[2]*s[3];
    c[67] = s[2]*s[2]*s[3]*s[3];
    c[68] = s[2]*s[3]*s[3]*s[3];
    c[69] = s[3]*s[3]*s[3]*s[3];
    for (int i = 1; i < 70; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order4_Dim4_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[70];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    c[15] = s[0]*s[0]*s[0];
    c[16] = s[0]*s[0]*s[1];
    c[17] = s[0]*s[0]*s[2];
    c[18] = s[0]*s[0]*s[3];
    c[19] = s[0]*s[1]*s[1];
    c[20] = s[0]*s[1]*s[2];
    c[21] = s[0]*s[1]*s[3];
    c[22] = s[0]*s[2]*s[2];
    c[23] = s[0]*s[2]*s[3];
    c[24] = s[0]*s[3]*s[3];
    c[25] = s[1]*s[1]*s[1];
    c[26] = s[1]*s[1]*s[2];
    c[27] = s[1]*s[1]*s[3];
    c[28] = s[1]*s[2]*s[2];
    c[29] = s[1]*s[2]*s[3];
    c[30] = s[1]*s[3]*s[3];
    c[31] = s[2]*s[2]*s[2];
    c[32] = s[2]*s[2]*s[3];
    c[33] = s[2]*s[3]*s[3];
    c[34] = s[3]*s[3]*s[3];
    c[35] = s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[3];
    c[39] = s[0]*s[0]*s[1]*s[1];
    c[40] = s[0]*s[0]*s[1]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[3];
    c[42] = s[0]*s[0]*s[2]*s[2];
    c[43] = s[0]*s[0]*s[2]*s[3];
    c[44] = s[0]*s[0]*s[3]*s[3];
    c[45] = s[0]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[3];
    c[48] = s[0]*s[1]*s[2]*s[2];
    c[49] = s[0]*s[1]*s[2]*s[3];
    c[50] = s[0]*s[1]*s[3]*s[3];
    c[51] = s[0]*s[2]*s[2]*s[2];
    c[52] = s[0]*s[2]*s[2]*s[3];
    c[53] = s[0]*s[2]*s[3]*s[3];
    c[54] = s[0]*s[3]*s[3]*s[3];
    c[55] = s[1]*s[1]*s[1]*s[1];
    c[56] = s[1]*s[1]*s[1]*s[2];
    c[57] = s[1]*s[1]*s[1]*s[3];
    c[58] = s[1]*s[1]*s[2]*s[2];
    c[59] = s[1]*s[1]*s[2]*s[3];
    c[60] = s[1]*s[1]*s[3]*s[3];
    c[61] = s[1]*s[2]*s[2]*s[2];
    c[62] = s[1]*s[2]*s[2]*s[3];
    c[63] = s[1]*s[2]*s[3]*s[3];
    c[64] = s[1]*s[3]*s[3]*s[3];
    c[65] = s[2]*s[2]*s[2]*s[2];
    c[66] = s[2]*s[2]*s[2]*s[3];
    c[67] = s[2]*s[2]*s[3]*s[3];
    c[68] = s[2]*s[3]*s[3]*s[3];
    c[69] = s[3]*s[3]*s[3]*s[3];
    Float e = (g(s) - v);
    for (int i = 0; i < 70; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order4_Dim4_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[70];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    c[15] = s[0]*s[0]*s[0];
    c[16] = s[0]*s[0]*s[1];
    c[17] = s[0]*s[0]*s[2];
    c[18] = s[0]*s[0]*s[3];
    c[19] = s[0]*s[1]*s[1];
    c[20] = s[0]*s[1]*s[2];
    c[21] = s[0]*s[1]*s[3];
    c[22] = s[0]*s[2]*s[2];
    c[23] = s[0]*s[2]*s[3];
    c[24] = s[0]*s[3]*s[3];
    c[25] = s[1]*s[1]*s[1];
    c[26] = s[1]*s[1]*s[2];
    c[27] = s[1]*s[1]*s[3];
    c[28] = s[1]*s[2]*s[2];
    c[29] = s[1]*s[2]*s[3];
    c[30] = s[1]*s[3]*s[3];
    c[31] = s[2]*s[2]*s[2];
    c[32] = s[2]*s[2]*s[3];
    c[33] = s[2]*s[3]*s[3];
    c[34] = s[3]*s[3]*s[3];
    c[35] = s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[3];
    c[39] = s[0]*s[0]*s[1]*s[1];
    c[40] = s[0]*s[0]*s[1]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[3];
    c[42] = s[0]*s[0]*s[2]*s[2];
    c[43] = s[0]*s[0]*s[2]*s[3];
    c[44] = s[0]*s[0]*s[3]*s[3];
    c[45] = s[0]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[3];
    c[48] = s[0]*s[1]*s[2]*s[2];
    c[49] = s[0]*s[1]*s[2]*s[3];
    c[50] = s[0]*s[1]*s[3]*s[3];
    c[51] = s[0]*s[2]*s[2]*s[2];
    c[52] = s[0]*s[2]*s[2]*s[3];
    c[53] = s[0]*s[2]*s[3]*s[3];
    c[54] = s[0]*s[3]*s[3]*s[3];
    c[55] = s[1]*s[1]*s[1]*s[1];
    c[56] = s[1]*s[1]*s[1]*s[2];
    c[57] = s[1]*s[1]*s[1]*s[3];
    c[58] = s[1]*s[1]*s[2]*s[2];
    c[59] = s[1]*s[1]*s[2]*s[3];
    c[60] = s[1]*s[1]*s[3]*s[3];
    c[61] = s[1]*s[2]*s[2]*s[2];
    c[62] = s[1]*s[2]*s[2]*s[3];
    c[63] = s[1]*s[2]*s[3]*s[3];
    c[64] = s[1]*s[3]*s[3]*s[3];
    c[65] = s[2]*s[2]*s[2]*s[2];
    c[66] = s[2]*s[2]*s[2]*s[3];
    c[67] = s[2]*s[2]*s[3]*s[3];
    c[68] = s[2]*s[3]*s[3]*s[3];
    c[69] = s[3]*s[3]*s[3]*s[3];
    for (int i = 1; i < 70; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 70; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}
void Poly_Order5_Dim4_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order5_Dim4_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[126];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    c[15] = s[0]*s[0]*s[0];
    c[16] = s[0]*s[0]*s[1];
    c[17] = s[0]*s[0]*s[2];
    c[18] = s[0]*s[0]*s[3];
    c[19] = s[0]*s[1]*s[1];
    c[20] = s[0]*s[1]*s[2];
    c[21] = s[0]*s[1]*s[3];
    c[22] = s[0]*s[2]*s[2];
    c[23] = s[0]*s[2]*s[3];
    c[24] = s[0]*s[3]*s[3];
    c[25] = s[1]*s[1]*s[1];
    c[26] = s[1]*s[1]*s[2];
    c[27] = s[1]*s[1]*s[3];
    c[28] = s[1]*s[2]*s[2];
    c[29] = s[1]*s[2]*s[3];
    c[30] = s[1]*s[3]*s[3];
    c[31] = s[2]*s[2]*s[2];
    c[32] = s[2]*s[2]*s[3];
    c[33] = s[2]*s[3]*s[3];
    c[34] = s[3]*s[3]*s[3];
    c[35] = s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[3];
    c[39] = s[0]*s[0]*s[1]*s[1];
    c[40] = s[0]*s[0]*s[1]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[3];
    c[42] = s[0]*s[0]*s[2]*s[2];
    c[43] = s[0]*s[0]*s[2]*s[3];
    c[44] = s[0]*s[0]*s[3]*s[3];
    c[45] = s[0]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[3];
    c[48] = s[0]*s[1]*s[2]*s[2];
    c[49] = s[0]*s[1]*s[2]*s[3];
    c[50] = s[0]*s[1]*s[3]*s[3];
    c[51] = s[0]*s[2]*s[2]*s[2];
    c[52] = s[0]*s[2]*s[2]*s[3];
    c[53] = s[0]*s[2]*s[3]*s[3];
    c[54] = s[0]*s[3]*s[3]*s[3];
    c[55] = s[1]*s[1]*s[1]*s[1];
    c[56] = s[1]*s[1]*s[1]*s[2];
    c[57] = s[1]*s[1]*s[1]*s[3];
    c[58] = s[1]*s[1]*s[2]*s[2];
    c[59] = s[1]*s[1]*s[2]*s[3];
    c[60] = s[1]*s[1]*s[3]*s[3];
    c[61] = s[1]*s[2]*s[2]*s[2];
    c[62] = s[1]*s[2]*s[2]*s[3];
    c[63] = s[1]*s[2]*s[3]*s[3];
    c[64] = s[1]*s[3]*s[3]*s[3];
    c[65] = s[2]*s[2]*s[2]*s[2];
    c[66] = s[2]*s[2]*s[2]*s[3];
    c[67] = s[2]*s[2]*s[3]*s[3];
    c[68] = s[2]*s[3]*s[3]*s[3];
    c[69] = s[3]*s[3]*s[3]*s[3];
    c[70] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[71] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[72] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[73] = s[0]*s[0]*s[0]*s[0]*s[3];
    c[74] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[75] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[76] = s[0]*s[0]*s[0]*s[1]*s[3];
    c[77] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[78] = s[0]*s[0]*s[0]*s[2]*s[3];
    c[79] = s[0]*s[0]*s[0]*s[3]*s[3];
    c[80] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[81] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[82] = s[0]*s[0]*s[1]*s[1]*s[3];
    c[83] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[84] = s[0]*s[0]*s[1]*s[2]*s[3];
    c[85] = s[0]*s[0]*s[1]*s[3]*s[3];
    c[86] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[87] = s[0]*s[0]*s[2]*s[2]*s[3];
    c[88] = s[0]*s[0]*s[2]*s[3]*s[3];
    c[89] = s[0]*s[0]*s[3]*s[3]*s[3];
    c[90] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[91] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[92] = s[0]*s[1]*s[1]*s[1]*s[3];
    c[93] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[94] = s[0]*s[1]*s[1]*s[2]*s[3];
    c[95] = s[0]*s[1]*s[1]*s[3]*s[3];
    c[96] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[97] = s[0]*s[1]*s[2]*s[2]*s[3];
    c[98] = s[0]*s[1]*s[2]*s[3]*s[3];
    c[99] = s[0]*s[1]*s[3]*s[3]*s[3];
    c[100] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[101] = s[0]*s[2]*s[2]*s[2]*s[3];
    c[102] = s[0]*s[2]*s[2]*s[3]*s[3];
    c[103] = s[0]*s[2]*s[3]*s[3]*s[3];
    c[104] = s[0]*s[3]*s[3]*s[3]*s[3];
    c[105] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[106] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[107] = s[1]*s[1]*s[1]*s[1]*s[3];
    c[108] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[109] = s[1]*s[1]*s[1]*s[2]*s[3];
    c[110] = s[1]*s[1]*s[1]*s[3]*s[3];
    c[111] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[112] = s[1]*s[1]*s[2]*s[2]*s[3];
    c[113] = s[1]*s[1]*s[2]*s[3]*s[3];
    c[114] = s[1]*s[1]*s[3]*s[3]*s[3];
    c[115] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[116] = s[1]*s[2]*s[2]*s[2]*s[3];
    c[117] = s[1]*s[2]*s[2]*s[3]*s[3];
    c[118] = s[1]*s[2]*s[3]*s[3]*s[3];
    c[119] = s[1]*s[3]*s[3]*s[3]*s[3];
    c[120] = s[2]*s[2]*s[2]*s[2]*s[2];
    c[121] = s[2]*s[2]*s[2]*s[2]*s[3];
    c[122] = s[2]*s[2]*s[2]*s[3]*s[3];
    c[123] = s[2]*s[2]*s[3]*s[3]*s[3];
    c[124] = s[2]*s[3]*s[3]*s[3]*s[3];
    c[125] = s[3]*s[3]*s[3]*s[3]*s[3];
    for (int i = 0; i < 126; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 126; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order5_Dim4_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order5_Dim4_luminance::G() const {
    float v = X[0];
    int div[126];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 2;
    div[5] = 3;
    div[6] = 4;
    div[7] = 4;
    div[8] = 4;
    div[9] = 3;
    div[10] = 4;
    div[11] = 4;
    div[12] = 3;
    div[13] = 4;
    div[14] = 3;
    div[15] = 4;
    div[16] = 6;
    div[17] = 6;
    div[18] = 6;
    div[19] = 6;
    div[20] = 8;
    div[21] = 8;
    div[22] = 6;
    div[23] = 8;
    div[24] = 6;
    div[25] = 4;
    div[26] = 6;
    div[27] = 6;
    div[28] = 6;
    div[29] = 8;
    div[30] = 6;
    div[31] = 4;
    div[32] = 6;
    div[33] = 6;
    div[34] = 4;
    div[35] = 5;
    div[36] = 8;
    div[37] = 8;
    div[38] = 8;
    div[39] = 9;
    div[40] = 12;
    div[41] = 12;
    div[42] = 9;
    div[43] = 12;
    div[44] = 9;
    div[45] = 8;
    div[46] = 12;
    div[47] = 12;
    div[48] = 12;
    div[49] = 16;
    div[50] = 12;
    div[51] = 8;
    div[52] = 12;
    div[53] = 12;
    div[54] = 8;
    div[55] = 5;
    div[56] = 8;
    div[57] = 8;
    div[58] = 9;
    div[59] = 12;
    div[60] = 9;
    div[61] = 8;
    div[62] = 12;
    div[63] = 12;
    div[64] = 8;
    div[65] = 5;
    div[66] = 8;
    div[67] = 9;
    div[68] = 8;
    div[69] = 5;
    div[70] = 6;
    div[71] = 10;
    div[72] = 10;
    div[73] = 10;
    div[74] = 12;
    div[75] = 16;
    div[76] = 16;
    div[77] = 12;
    div[78] = 16;
    div[79] = 12;
    div[80] = 12;
    div[81] = 18;
    div[82] = 18;
    div[83] = 18;
    div[84] = 24;
    div[85] = 18;
    div[86] = 12;
    div[87] = 18;
    div[88] = 18;
    div[89] = 12;
    div[90] = 10;
    div[91] = 16;
    div[92] = 16;
    div[93] = 18;
    div[94] = 24;
    div[95] = 18;
    div[96] = 16;
    div[97] = 24;
    div[98] = 24;
    div[99] = 16;
    div[100] = 10;
    div[101] = 16;
    div[102] = 18;
    div[103] = 16;
    div[104] = 10;
    div[105] = 6;
    div[106] = 10;
    div[107] = 10;
    div[108] = 12;
    div[109] = 16;
    div[110] = 12;
    div[111] = 12;
    div[112] = 18;
    div[113] = 18;
    div[114] = 12;
    div[115] = 10;
    div[116] = 16;
    div[117] = 18;
    div[118] = 16;
    div[119] = 10;
    div[120] = 6;
    div[121] = 10;
    div[122] = 12;
    div[123] = 12;
    div[124] = 10;
    div[125] = 6;
    for (int i = 1; i < 126; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order5_Dim4_luminance::Gpix(Float ax, Float ay, Float bx, Float by) const {
    float v = X[0];
    float f[126];
    f[0] = ((bx - ax)*(by - ay)) / 1;
    f[1] = ((bx*bx - ax*ax)*(by - ay)) / 2;
    f[2] = ((bx - ax)*(by*by - ay*ay)) / 2;
    f[3] = ((bx - ax)*(by - ay)) / 2;
    f[4] = ((bx - ax)*(by - ay)) / 2;
    f[5] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 3;
    f[6] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 4;
    f[7] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[8] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[9] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 3;
    f[10] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[11] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[12] = ((bx - ax)*(by - ay)) / 3;
    f[13] = ((bx - ax)*(by - ay)) / 4;
    f[14] = ((bx - ax)*(by - ay)) / 3;
    f[15] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 4;
    f[16] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 6;
    f[17] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 6;
    f[18] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 6;
    f[19] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 6;
    f[20] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 8;
    f[21] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 8;
    f[22] = ((bx*bx - ax*ax)*(by - ay)) / 6;
    f[23] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[24] = ((bx*bx - ax*ax)*(by - ay)) / 6;
    f[25] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 4;
    f[26] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 6;
    f[27] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 6;
    f[28] = ((bx - ax)*(by*by - ay*ay)) / 6;
    f[29] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[30] = ((bx - ax)*(by*by - ay*ay)) / 6;
    f[31] = ((bx - ax)*(by - ay)) / 4;
    f[32] = ((bx - ax)*(by - ay)) / 6;
    f[33] = ((bx - ax)*(by - ay)) / 6;
    f[34] = ((bx - ax)*(by - ay)) / 4;
    f[35] = ((bx*bx*bx*bx*bx - ax*ax*ax*ax*ax)*(by - ay)) / 5;
    f[36] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by*by - ay*ay)) / 8;
    f[37] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 8;
    f[38] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 8;
    f[39] = ((bx*bx*bx - ax*ax*ax)*(by*by*by - ay*ay*ay)) / 9;
    f[40] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 12;
    f[41] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 12;
    f[42] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 9;
    f[43] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 12;
    f[44] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 9;
    f[45] = ((bx*bx - ax*ax)*(by*by*by*by - ay*ay*ay*ay)) / 8;
    f[46] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 12;
    f[47] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 12;
    f[48] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 12;
    f[49] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 16;
    f[50] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 12;
    f[51] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[52] = ((bx*bx - ax*ax)*(by - ay)) / 12;
    f[53] = ((bx*bx - ax*ax)*(by - ay)) / 12;
    f[54] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[55] = ((bx - ax)*(by*by*by*by*by - ay*ay*ay*ay*ay)) / 5;
    f[56] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 8;
    f[57] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 8;
    f[58] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 9;
    f[59] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 12;
    f[60] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 9;
    f[61] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[62] = ((bx - ax)*(by*by - ay*ay)) / 12;
    f[63] = ((bx - ax)*(by*by - ay*ay)) / 12;
    f[64] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[65] = ((bx - ax)*(by - ay)) / 5;
    f[66] = ((bx - ax)*(by - ay)) / 8;
    f[67] = ((bx - ax)*(by - ay)) / 9;
    f[68] = ((bx - ax)*(by - ay)) / 8;
    f[69] = ((bx - ax)*(by - ay)) / 5;
    f[70] = ((bx*bx*bx*bx*bx*bx - ax*ax*ax*ax*ax*ax)*(by - ay)) / 6;
    f[71] = ((bx*bx*bx*bx*bx - ax*ax*ax*ax*ax)*(by*by - ay*ay)) / 10;
    f[72] = ((bx*bx*bx*bx*bx - ax*ax*ax*ax*ax)*(by - ay)) / 10;
    f[73] = ((bx*bx*bx*bx*bx - ax*ax*ax*ax*ax)*(by - ay)) / 10;
    f[74] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by*by*by - ay*ay*ay)) / 12;
    f[75] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by*by - ay*ay)) / 16;
    f[76] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by*by - ay*ay)) / 16;
    f[77] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 12;
    f[78] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 16;
    f[79] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 12;
    f[80] = ((bx*bx*bx - ax*ax*ax)*(by*by*by*by - ay*ay*ay*ay)) / 12;
    f[81] = ((bx*bx*bx - ax*ax*ax)*(by*by*by - ay*ay*ay)) / 18;
    f[82] = ((bx*bx*bx - ax*ax*ax)*(by*by*by - ay*ay*ay)) / 18;
    f[83] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 18;
    f[84] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 24;
    f[85] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 18;
    f[86] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 12;
    f[87] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 18;
    f[88] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 18;
    f[89] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 12;
    f[90] = ((bx*bx - ax*ax)*(by*by*by*by*by - ay*ay*ay*ay*ay)) / 10;
    f[91] = ((bx*bx - ax*ax)*(by*by*by*by - ay*ay*ay*ay)) / 16;
    f[92] = ((bx*bx - ax*ax)*(by*by*by*by - ay*ay*ay*ay)) / 16;
    f[93] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 18;
    f[94] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 24;
    f[95] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 18;
    f[96] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 16;
    f[97] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 24;
    f[98] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 24;
    f[99] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 16;
    f[100] = ((bx*bx - ax*ax)*(by - ay)) / 10;
    f[101] = ((bx*bx - ax*ax)*(by - ay)) / 16;
    f[102] = ((bx*bx - ax*ax)*(by - ay)) / 18;
    f[103] = ((bx*bx - ax*ax)*(by - ay)) / 16;
    f[104] = ((bx*bx - ax*ax)*(by - ay)) / 10;
    f[105] = ((bx - ax)*(by*by*by*by*by*by - ay*ay*ay*ay*ay*ay)) / 6;
    f[106] = ((bx - ax)*(by*by*by*by*by - ay*ay*ay*ay*ay)) / 10;
    f[107] = ((bx - ax)*(by*by*by*by*by - ay*ay*ay*ay*ay)) / 10;
    f[108] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 12;
    f[109] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 16;
    f[110] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 12;
    f[111] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 12;
    f[112] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 18;
    f[113] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 18;
    f[114] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 12;
    f[115] = ((bx - ax)*(by*by - ay*ay)) / 10;
    f[116] = ((bx - ax)*(by*by - ay*ay)) / 16;
    f[117] = ((bx - ax)*(by*by - ay*ay)) / 18;
    f[118] = ((bx - ax)*(by*by - ay*ay)) / 16;
    f[119] = ((bx - ax)*(by*by - ay*ay)) / 10;
    f[120] = ((bx - ax)*(by - ay)) / 6;
    f[121] = ((bx - ax)*(by - ay)) / 10;
    f[122] = ((bx - ax)*(by - ay)) / 12;
    f[123] = ((bx - ax)*(by - ay)) / 12;
    f[124] = ((bx - ax)*(by - ay)) / 10;
    f[125] = ((bx - ax)*(by - ay)) / 6;
    for (int i = 1; i < 126; ++i) {
        v += X[i] * f[i];
    }
    return v;
}
float Poly_Order5_Dim4_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[126];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    c[15] = s[0]*s[0]*s[0];
    c[16] = s[0]*s[0]*s[1];
    c[17] = s[0]*s[0]*s[2];
    c[18] = s[0]*s[0]*s[3];
    c[19] = s[0]*s[1]*s[1];
    c[20] = s[0]*s[1]*s[2];
    c[21] = s[0]*s[1]*s[3];
    c[22] = s[0]*s[2]*s[2];
    c[23] = s[0]*s[2]*s[3];
    c[24] = s[0]*s[3]*s[3];
    c[25] = s[1]*s[1]*s[1];
    c[26] = s[1]*s[1]*s[2];
    c[27] = s[1]*s[1]*s[3];
    c[28] = s[1]*s[2]*s[2];
    c[29] = s[1]*s[2]*s[3];
    c[30] = s[1]*s[3]*s[3];
    c[31] = s[2]*s[2]*s[2];
    c[32] = s[2]*s[2]*s[3];
    c[33] = s[2]*s[3]*s[3];
    c[34] = s[3]*s[3]*s[3];
    c[35] = s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[3];
    c[39] = s[0]*s[0]*s[1]*s[1];
    c[40] = s[0]*s[0]*s[1]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[3];
    c[42] = s[0]*s[0]*s[2]*s[2];
    c[43] = s[0]*s[0]*s[2]*s[3];
    c[44] = s[0]*s[0]*s[3]*s[3];
    c[45] = s[0]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[3];
    c[48] = s[0]*s[1]*s[2]*s[2];
    c[49] = s[0]*s[1]*s[2]*s[3];
    c[50] = s[0]*s[1]*s[3]*s[3];
    c[51] = s[0]*s[2]*s[2]*s[2];
    c[52] = s[0]*s[2]*s[2]*s[3];
    c[53] = s[0]*s[2]*s[3]*s[3];
    c[54] = s[0]*s[3]*s[3]*s[3];
    c[55] = s[1]*s[1]*s[1]*s[1];
    c[56] = s[1]*s[1]*s[1]*s[2];
    c[57] = s[1]*s[1]*s[1]*s[3];
    c[58] = s[1]*s[1]*s[2]*s[2];
    c[59] = s[1]*s[1]*s[2]*s[3];
    c[60] = s[1]*s[1]*s[3]*s[3];
    c[61] = s[1]*s[2]*s[2]*s[2];
    c[62] = s[1]*s[2]*s[2]*s[3];
    c[63] = s[1]*s[2]*s[3]*s[3];
    c[64] = s[1]*s[3]*s[3]*s[3];
    c[65] = s[2]*s[2]*s[2]*s[2];
    c[66] = s[2]*s[2]*s[2]*s[3];
    c[67] = s[2]*s[2]*s[3]*s[3];
    c[68] = s[2]*s[3]*s[3]*s[3];
    c[69] = s[3]*s[3]*s[3]*s[3];
    c[70] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[71] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[72] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[73] = s[0]*s[0]*s[0]*s[0]*s[3];
    c[74] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[75] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[76] = s[0]*s[0]*s[0]*s[1]*s[3];
    c[77] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[78] = s[0]*s[0]*s[0]*s[2]*s[3];
    c[79] = s[0]*s[0]*s[0]*s[3]*s[3];
    c[80] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[81] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[82] = s[0]*s[0]*s[1]*s[1]*s[3];
    c[83] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[84] = s[0]*s[0]*s[1]*s[2]*s[3];
    c[85] = s[0]*s[0]*s[1]*s[3]*s[3];
    c[86] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[87] = s[0]*s[0]*s[2]*s[2]*s[3];
    c[88] = s[0]*s[0]*s[2]*s[3]*s[3];
    c[89] = s[0]*s[0]*s[3]*s[3]*s[3];
    c[90] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[91] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[92] = s[0]*s[1]*s[1]*s[1]*s[3];
    c[93] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[94] = s[0]*s[1]*s[1]*s[2]*s[3];
    c[95] = s[0]*s[1]*s[1]*s[3]*s[3];
    c[96] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[97] = s[0]*s[1]*s[2]*s[2]*s[3];
    c[98] = s[0]*s[1]*s[2]*s[3]*s[3];
    c[99] = s[0]*s[1]*s[3]*s[3]*s[3];
    c[100] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[101] = s[0]*s[2]*s[2]*s[2]*s[3];
    c[102] = s[0]*s[2]*s[2]*s[3]*s[3];
    c[103] = s[0]*s[2]*s[3]*s[3]*s[3];
    c[104] = s[0]*s[3]*s[3]*s[3]*s[3];
    c[105] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[106] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[107] = s[1]*s[1]*s[1]*s[1]*s[3];
    c[108] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[109] = s[1]*s[1]*s[1]*s[2]*s[3];
    c[110] = s[1]*s[1]*s[1]*s[3]*s[3];
    c[111] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[112] = s[1]*s[1]*s[2]*s[2]*s[3];
    c[113] = s[1]*s[1]*s[2]*s[3]*s[3];
    c[114] = s[1]*s[1]*s[3]*s[3]*s[3];
    c[115] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[116] = s[1]*s[2]*s[2]*s[2]*s[3];
    c[117] = s[1]*s[2]*s[2]*s[3]*s[3];
    c[118] = s[1]*s[2]*s[3]*s[3]*s[3];
    c[119] = s[1]*s[3]*s[3]*s[3]*s[3];
    c[120] = s[2]*s[2]*s[2]*s[2]*s[2];
    c[121] = s[2]*s[2]*s[2]*s[2]*s[3];
    c[122] = s[2]*s[2]*s[2]*s[3]*s[3];
    c[123] = s[2]*s[2]*s[3]*s[3]*s[3];
    c[124] = s[2]*s[3]*s[3]*s[3]*s[3];
    c[125] = s[3]*s[3]*s[3]*s[3]*s[3];
    for (int i = 1; i < 126; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order5_Dim4_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[126];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    c[15] = s[0]*s[0]*s[0];
    c[16] = s[0]*s[0]*s[1];
    c[17] = s[0]*s[0]*s[2];
    c[18] = s[0]*s[0]*s[3];
    c[19] = s[0]*s[1]*s[1];
    c[20] = s[0]*s[1]*s[2];
    c[21] = s[0]*s[1]*s[3];
    c[22] = s[0]*s[2]*s[2];
    c[23] = s[0]*s[2]*s[3];
    c[24] = s[0]*s[3]*s[3];
    c[25] = s[1]*s[1]*s[1];
    c[26] = s[1]*s[1]*s[2];
    c[27] = s[1]*s[1]*s[3];
    c[28] = s[1]*s[2]*s[2];
    c[29] = s[1]*s[2]*s[3];
    c[30] = s[1]*s[3]*s[3];
    c[31] = s[2]*s[2]*s[2];
    c[32] = s[2]*s[2]*s[3];
    c[33] = s[2]*s[3]*s[3];
    c[34] = s[3]*s[3]*s[3];
    c[35] = s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[3];
    c[39] = s[0]*s[0]*s[1]*s[1];
    c[40] = s[0]*s[0]*s[1]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[3];
    c[42] = s[0]*s[0]*s[2]*s[2];
    c[43] = s[0]*s[0]*s[2]*s[3];
    c[44] = s[0]*s[0]*s[3]*s[3];
    c[45] = s[0]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[3];
    c[48] = s[0]*s[1]*s[2]*s[2];
    c[49] = s[0]*s[1]*s[2]*s[3];
    c[50] = s[0]*s[1]*s[3]*s[3];
    c[51] = s[0]*s[2]*s[2]*s[2];
    c[52] = s[0]*s[2]*s[2]*s[3];
    c[53] = s[0]*s[2]*s[3]*s[3];
    c[54] = s[0]*s[3]*s[3]*s[3];
    c[55] = s[1]*s[1]*s[1]*s[1];
    c[56] = s[1]*s[1]*s[1]*s[2];
    c[57] = s[1]*s[1]*s[1]*s[3];
    c[58] = s[1]*s[1]*s[2]*s[2];
    c[59] = s[1]*s[1]*s[2]*s[3];
    c[60] = s[1]*s[1]*s[3]*s[3];
    c[61] = s[1]*s[2]*s[2]*s[2];
    c[62] = s[1]*s[2]*s[2]*s[3];
    c[63] = s[1]*s[2]*s[3]*s[3];
    c[64] = s[1]*s[3]*s[3]*s[3];
    c[65] = s[2]*s[2]*s[2]*s[2];
    c[66] = s[2]*s[2]*s[2]*s[3];
    c[67] = s[2]*s[2]*s[3]*s[3];
    c[68] = s[2]*s[3]*s[3]*s[3];
    c[69] = s[3]*s[3]*s[3]*s[3];
    c[70] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[71] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[72] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[73] = s[0]*s[0]*s[0]*s[0]*s[3];
    c[74] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[75] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[76] = s[0]*s[0]*s[0]*s[1]*s[3];
    c[77] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[78] = s[0]*s[0]*s[0]*s[2]*s[3];
    c[79] = s[0]*s[0]*s[0]*s[3]*s[3];
    c[80] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[81] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[82] = s[0]*s[0]*s[1]*s[1]*s[3];
    c[83] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[84] = s[0]*s[0]*s[1]*s[2]*s[3];
    c[85] = s[0]*s[0]*s[1]*s[3]*s[3];
    c[86] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[87] = s[0]*s[0]*s[2]*s[2]*s[3];
    c[88] = s[0]*s[0]*s[2]*s[3]*s[3];
    c[89] = s[0]*s[0]*s[3]*s[3]*s[3];
    c[90] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[91] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[92] = s[0]*s[1]*s[1]*s[1]*s[3];
    c[93] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[94] = s[0]*s[1]*s[1]*s[2]*s[3];
    c[95] = s[0]*s[1]*s[1]*s[3]*s[3];
    c[96] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[97] = s[0]*s[1]*s[2]*s[2]*s[3];
    c[98] = s[0]*s[1]*s[2]*s[3]*s[3];
    c[99] = s[0]*s[1]*s[3]*s[3]*s[3];
    c[100] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[101] = s[0]*s[2]*s[2]*s[2]*s[3];
    c[102] = s[0]*s[2]*s[2]*s[3]*s[3];
    c[103] = s[0]*s[2]*s[3]*s[3]*s[3];
    c[104] = s[0]*s[3]*s[3]*s[3]*s[3];
    c[105] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[106] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[107] = s[1]*s[1]*s[1]*s[1]*s[3];
    c[108] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[109] = s[1]*s[1]*s[1]*s[2]*s[3];
    c[110] = s[1]*s[1]*s[1]*s[3]*s[3];
    c[111] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[112] = s[1]*s[1]*s[2]*s[2]*s[3];
    c[113] = s[1]*s[1]*s[2]*s[3]*s[3];
    c[114] = s[1]*s[1]*s[3]*s[3]*s[3];
    c[115] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[116] = s[1]*s[2]*s[2]*s[2]*s[3];
    c[117] = s[1]*s[2]*s[2]*s[3]*s[3];
    c[118] = s[1]*s[2]*s[3]*s[3]*s[3];
    c[119] = s[1]*s[3]*s[3]*s[3]*s[3];
    c[120] = s[2]*s[2]*s[2]*s[2]*s[2];
    c[121] = s[2]*s[2]*s[2]*s[2]*s[3];
    c[122] = s[2]*s[2]*s[2]*s[3]*s[3];
    c[123] = s[2]*s[2]*s[3]*s[3]*s[3];
    c[124] = s[2]*s[3]*s[3]*s[3]*s[3];
    c[125] = s[3]*s[3]*s[3]*s[3]*s[3];
    Float e = (g(s) - v);
    for (int i = 0; i < 126; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order5_Dim4_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[126];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[0]*s[0];
    c[6] = s[0]*s[1];
    c[7] = s[0]*s[2];
    c[8] = s[0]*s[3];
    c[9] = s[1]*s[1];
    c[10] = s[1]*s[2];
    c[11] = s[1]*s[3];
    c[12] = s[2]*s[2];
    c[13] = s[2]*s[3];
    c[14] = s[3]*s[3];
    c[15] = s[0]*s[0]*s[0];
    c[16] = s[0]*s[0]*s[1];
    c[17] = s[0]*s[0]*s[2];
    c[18] = s[0]*s[0]*s[3];
    c[19] = s[0]*s[1]*s[1];
    c[20] = s[0]*s[1]*s[2];
    c[21] = s[0]*s[1]*s[3];
    c[22] = s[0]*s[2]*s[2];
    c[23] = s[0]*s[2]*s[3];
    c[24] = s[0]*s[3]*s[3];
    c[25] = s[1]*s[1]*s[1];
    c[26] = s[1]*s[1]*s[2];
    c[27] = s[1]*s[1]*s[3];
    c[28] = s[1]*s[2]*s[2];
    c[29] = s[1]*s[2]*s[3];
    c[30] = s[1]*s[3]*s[3];
    c[31] = s[2]*s[2]*s[2];
    c[32] = s[2]*s[2]*s[3];
    c[33] = s[2]*s[3]*s[3];
    c[34] = s[3]*s[3]*s[3];
    c[35] = s[0]*s[0]*s[0]*s[0];
    c[36] = s[0]*s[0]*s[0]*s[1];
    c[37] = s[0]*s[0]*s[0]*s[2];
    c[38] = s[0]*s[0]*s[0]*s[3];
    c[39] = s[0]*s[0]*s[1]*s[1];
    c[40] = s[0]*s[0]*s[1]*s[2];
    c[41] = s[0]*s[0]*s[1]*s[3];
    c[42] = s[0]*s[0]*s[2]*s[2];
    c[43] = s[0]*s[0]*s[2]*s[3];
    c[44] = s[0]*s[0]*s[3]*s[3];
    c[45] = s[0]*s[1]*s[1]*s[1];
    c[46] = s[0]*s[1]*s[1]*s[2];
    c[47] = s[0]*s[1]*s[1]*s[3];
    c[48] = s[0]*s[1]*s[2]*s[2];
    c[49] = s[0]*s[1]*s[2]*s[3];
    c[50] = s[0]*s[1]*s[3]*s[3];
    c[51] = s[0]*s[2]*s[2]*s[2];
    c[52] = s[0]*s[2]*s[2]*s[3];
    c[53] = s[0]*s[2]*s[3]*s[3];
    c[54] = s[0]*s[3]*s[3]*s[3];
    c[55] = s[1]*s[1]*s[1]*s[1];
    c[56] = s[1]*s[1]*s[1]*s[2];
    c[57] = s[1]*s[1]*s[1]*s[3];
    c[58] = s[1]*s[1]*s[2]*s[2];
    c[59] = s[1]*s[1]*s[2]*s[3];
    c[60] = s[1]*s[1]*s[3]*s[3];
    c[61] = s[1]*s[2]*s[2]*s[2];
    c[62] = s[1]*s[2]*s[2]*s[3];
    c[63] = s[1]*s[2]*s[3]*s[3];
    c[64] = s[1]*s[3]*s[3]*s[3];
    c[65] = s[2]*s[2]*s[2]*s[2];
    c[66] = s[2]*s[2]*s[2]*s[3];
    c[67] = s[2]*s[2]*s[3]*s[3];
    c[68] = s[2]*s[3]*s[3]*s[3];
    c[69] = s[3]*s[3]*s[3]*s[3];
    c[70] = s[0]*s[0]*s[0]*s[0]*s[0];
    c[71] = s[0]*s[0]*s[0]*s[0]*s[1];
    c[72] = s[0]*s[0]*s[0]*s[0]*s[2];
    c[73] = s[0]*s[0]*s[0]*s[0]*s[3];
    c[74] = s[0]*s[0]*s[0]*s[1]*s[1];
    c[75] = s[0]*s[0]*s[0]*s[1]*s[2];
    c[76] = s[0]*s[0]*s[0]*s[1]*s[3];
    c[77] = s[0]*s[0]*s[0]*s[2]*s[2];
    c[78] = s[0]*s[0]*s[0]*s[2]*s[3];
    c[79] = s[0]*s[0]*s[0]*s[3]*s[3];
    c[80] = s[0]*s[0]*s[1]*s[1]*s[1];
    c[81] = s[0]*s[0]*s[1]*s[1]*s[2];
    c[82] = s[0]*s[0]*s[1]*s[1]*s[3];
    c[83] = s[0]*s[0]*s[1]*s[2]*s[2];
    c[84] = s[0]*s[0]*s[1]*s[2]*s[3];
    c[85] = s[0]*s[0]*s[1]*s[3]*s[3];
    c[86] = s[0]*s[0]*s[2]*s[2]*s[2];
    c[87] = s[0]*s[0]*s[2]*s[2]*s[3];
    c[88] = s[0]*s[0]*s[2]*s[3]*s[3];
    c[89] = s[0]*s[0]*s[3]*s[3]*s[3];
    c[90] = s[0]*s[1]*s[1]*s[1]*s[1];
    c[91] = s[0]*s[1]*s[1]*s[1]*s[2];
    c[92] = s[0]*s[1]*s[1]*s[1]*s[3];
    c[93] = s[0]*s[1]*s[1]*s[2]*s[2];
    c[94] = s[0]*s[1]*s[1]*s[2]*s[3];
    c[95] = s[0]*s[1]*s[1]*s[3]*s[3];
    c[96] = s[0]*s[1]*s[2]*s[2]*s[2];
    c[97] = s[0]*s[1]*s[2]*s[2]*s[3];
    c[98] = s[0]*s[1]*s[2]*s[3]*s[3];
    c[99] = s[0]*s[1]*s[3]*s[3]*s[3];
    c[100] = s[0]*s[2]*s[2]*s[2]*s[2];
    c[101] = s[0]*s[2]*s[2]*s[2]*s[3];
    c[102] = s[0]*s[2]*s[2]*s[3]*s[3];
    c[103] = s[0]*s[2]*s[3]*s[3]*s[3];
    c[104] = s[0]*s[3]*s[3]*s[3]*s[3];
    c[105] = s[1]*s[1]*s[1]*s[1]*s[1];
    c[106] = s[1]*s[1]*s[1]*s[1]*s[2];
    c[107] = s[1]*s[1]*s[1]*s[1]*s[3];
    c[108] = s[1]*s[1]*s[1]*s[2]*s[2];
    c[109] = s[1]*s[1]*s[1]*s[2]*s[3];
    c[110] = s[1]*s[1]*s[1]*s[3]*s[3];
    c[111] = s[1]*s[1]*s[2]*s[2]*s[2];
    c[112] = s[1]*s[1]*s[2]*s[2]*s[3];
    c[113] = s[1]*s[1]*s[2]*s[3]*s[3];
    c[114] = s[1]*s[1]*s[3]*s[3]*s[3];
    c[115] = s[1]*s[2]*s[2]*s[2]*s[2];
    c[116] = s[1]*s[2]*s[2]*s[2]*s[3];
    c[117] = s[1]*s[2]*s[2]*s[3]*s[3];
    c[118] = s[1]*s[2]*s[3]*s[3]*s[3];
    c[119] = s[1]*s[3]*s[3]*s[3]*s[3];
    c[120] = s[2]*s[2]*s[2]*s[2]*s[2];
    c[121] = s[2]*s[2]*s[2]*s[2]*s[3];
    c[122] = s[2]*s[2]*s[2]*s[3]*s[3];
    c[123] = s[2]*s[2]*s[3]*s[3]*s[3];
    c[124] = s[2]*s[3]*s[3]*s[3]*s[3];
    c[125] = s[3]*s[3]*s[3]*s[3]*s[3];
    for (int i = 1; i < 126; ++i) {
        g += c[i] * X[i];
    }
    // Do one step SGD
    Float e = (g - v);
    for (int i = 0; i < 126; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
    return g;
}


}