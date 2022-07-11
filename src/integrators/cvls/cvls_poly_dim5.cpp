#include "cvls_poly_dim5.h"

namespace pbrt {

void Poly_Order0_Dim5_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order0_Dim5_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
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
void Poly_Order0_Dim5_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order0_Dim5_luminance::G() const {
    float v = X[0];
    int div[1];
    div[0] = 1;
    for (int i = 1; i < 1; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order0_Dim5_luminance::Gpix(Float ax, Float ay, Float bx, Float by) const {
    float v = X[0];
    float f[1];
    f[0] = ((bx - ax)*(by - ay)) / 1;
    for (int i = 1; i < 1; ++i) {
        v += X[i] * f[i];
    }
    return v;
}
float Poly_Order0_Dim5_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[1];
    c[0] = 1.0;
    for (int i = 1; i < 1; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order0_Dim5_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[1];
    c[0] = 1.0;
    Float e = (g(s) - v);
    for (int i = 0; i < 1; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order0_Dim5_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
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
void Poly_Order1_Dim5_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order1_Dim5_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[6];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    for (int i = 0; i < 6; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 6; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order1_Dim5_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order1_Dim5_luminance::G() const {
    float v = X[0];
    int div[6];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 2;
    div[5] = 2;
    for (int i = 1; i < 6; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order1_Dim5_luminance::Gpix(Float ax, Float ay, Float bx, Float by) const {
    float v = X[0];
    float f[6];
    f[0] = ((bx - ax)*(by - ay)) / 1;
    f[1] = ((bx*bx - ax*ax)*(by - ay)) / 2;
    f[2] = ((bx - ax)*(by*by - ay*ay)) / 2;
    f[3] = ((bx - ax)*(by - ay)) / 2;
    f[4] = ((bx - ax)*(by - ay)) / 2;
    f[5] = ((bx - ax)*(by - ay)) / 2;
    for (int i = 1; i < 6; ++i) {
        v += X[i] * f[i];
    }
    return v;
}
float Poly_Order1_Dim5_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[6];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    for (int i = 1; i < 6; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order1_Dim5_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[6];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    Float e = (g(s) - v);
    for (int i = 0; i < 6; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order1_Dim5_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[6];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
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
void Poly_Order2_Dim5_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order2_Dim5_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[21];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    c[6] = s[0]*s[0];
    c[7] = s[0]*s[1];
    c[8] = s[0]*s[2];
    c[9] = s[0]*s[3];
    c[10] = s[0]*s[4];
    c[11] = s[1]*s[1];
    c[12] = s[1]*s[2];
    c[13] = s[1]*s[3];
    c[14] = s[1]*s[4];
    c[15] = s[2]*s[2];
    c[16] = s[2]*s[3];
    c[17] = s[2]*s[4];
    c[18] = s[3]*s[3];
    c[19] = s[3]*s[4];
    c[20] = s[4]*s[4];
    for (int i = 0; i < 21; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 21; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order2_Dim5_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order2_Dim5_luminance::G() const {
    float v = X[0];
    int div[21];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 2;
    div[5] = 2;
    div[6] = 3;
    div[7] = 4;
    div[8] = 4;
    div[9] = 4;
    div[10] = 4;
    div[11] = 3;
    div[12] = 4;
    div[13] = 4;
    div[14] = 4;
    div[15] = 3;
    div[16] = 4;
    div[17] = 4;
    div[18] = 3;
    div[19] = 4;
    div[20] = 3;
    for (int i = 1; i < 21; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order2_Dim5_luminance::Gpix(Float ax, Float ay, Float bx, Float by) const {
    float v = X[0];
    float f[21];
    f[0] = ((bx - ax)*(by - ay)) / 1;
    f[1] = ((bx*bx - ax*ax)*(by - ay)) / 2;
    f[2] = ((bx - ax)*(by*by - ay*ay)) / 2;
    f[3] = ((bx - ax)*(by - ay)) / 2;
    f[4] = ((bx - ax)*(by - ay)) / 2;
    f[5] = ((bx - ax)*(by - ay)) / 2;
    f[6] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 3;
    f[7] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 4;
    f[8] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[9] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[10] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[11] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 3;
    f[12] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[13] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[14] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[15] = ((bx - ax)*(by - ay)) / 3;
    f[16] = ((bx - ax)*(by - ay)) / 4;
    f[17] = ((bx - ax)*(by - ay)) / 4;
    f[18] = ((bx - ax)*(by - ay)) / 3;
    f[19] = ((bx - ax)*(by - ay)) / 4;
    f[20] = ((bx - ax)*(by - ay)) / 3;
    for (int i = 1; i < 21; ++i) {
        v += X[i] * f[i];
    }
    return v;
}
float Poly_Order2_Dim5_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[21];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    c[6] = s[0]*s[0];
    c[7] = s[0]*s[1];
    c[8] = s[0]*s[2];
    c[9] = s[0]*s[3];
    c[10] = s[0]*s[4];
    c[11] = s[1]*s[1];
    c[12] = s[1]*s[2];
    c[13] = s[1]*s[3];
    c[14] = s[1]*s[4];
    c[15] = s[2]*s[2];
    c[16] = s[2]*s[3];
    c[17] = s[2]*s[4];
    c[18] = s[3]*s[3];
    c[19] = s[3]*s[4];
    c[20] = s[4]*s[4];
    for (int i = 1; i < 21; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order2_Dim5_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[21];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    c[6] = s[0]*s[0];
    c[7] = s[0]*s[1];
    c[8] = s[0]*s[2];
    c[9] = s[0]*s[3];
    c[10] = s[0]*s[4];
    c[11] = s[1]*s[1];
    c[12] = s[1]*s[2];
    c[13] = s[1]*s[3];
    c[14] = s[1]*s[4];
    c[15] = s[2]*s[2];
    c[16] = s[2]*s[3];
    c[17] = s[2]*s[4];
    c[18] = s[3]*s[3];
    c[19] = s[3]*s[4];
    c[20] = s[4]*s[4];
    Float e = (g(s) - v);
    for (int i = 0; i < 21; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order2_Dim5_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[21];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    c[6] = s[0]*s[0];
    c[7] = s[0]*s[1];
    c[8] = s[0]*s[2];
    c[9] = s[0]*s[3];
    c[10] = s[0]*s[4];
    c[11] = s[1]*s[1];
    c[12] = s[1]*s[2];
    c[13] = s[1]*s[3];
    c[14] = s[1]*s[4];
    c[15] = s[2]*s[2];
    c[16] = s[2]*s[3];
    c[17] = s[2]*s[4];
    c[18] = s[3]*s[3];
    c[19] = s[3]*s[4];
    c[20] = s[4]*s[4];
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
void Poly_Order3_Dim5_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order3_Dim5_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[56];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    c[6] = s[0]*s[0];
    c[7] = s[0]*s[1];
    c[8] = s[0]*s[2];
    c[9] = s[0]*s[3];
    c[10] = s[0]*s[4];
    c[11] = s[1]*s[1];
    c[12] = s[1]*s[2];
    c[13] = s[1]*s[3];
    c[14] = s[1]*s[4];
    c[15] = s[2]*s[2];
    c[16] = s[2]*s[3];
    c[17] = s[2]*s[4];
    c[18] = s[3]*s[3];
    c[19] = s[3]*s[4];
    c[20] = s[4]*s[4];
    c[21] = s[0]*s[0]*s[0];
    c[22] = s[0]*s[0]*s[1];
    c[23] = s[0]*s[0]*s[2];
    c[24] = s[0]*s[0]*s[3];
    c[25] = s[0]*s[0]*s[4];
    c[26] = s[0]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[3];
    c[29] = s[0]*s[1]*s[4];
    c[30] = s[0]*s[2]*s[2];
    c[31] = s[0]*s[2]*s[3];
    c[32] = s[0]*s[2]*s[4];
    c[33] = s[0]*s[3]*s[3];
    c[34] = s[0]*s[3]*s[4];
    c[35] = s[0]*s[4]*s[4];
    c[36] = s[1]*s[1]*s[1];
    c[37] = s[1]*s[1]*s[2];
    c[38] = s[1]*s[1]*s[3];
    c[39] = s[1]*s[1]*s[4];
    c[40] = s[1]*s[2]*s[2];
    c[41] = s[1]*s[2]*s[3];
    c[42] = s[1]*s[2]*s[4];
    c[43] = s[1]*s[3]*s[3];
    c[44] = s[1]*s[3]*s[4];
    c[45] = s[1]*s[4]*s[4];
    c[46] = s[2]*s[2]*s[2];
    c[47] = s[2]*s[2]*s[3];
    c[48] = s[2]*s[2]*s[4];
    c[49] = s[2]*s[3]*s[3];
    c[50] = s[2]*s[3]*s[4];
    c[51] = s[2]*s[4]*s[4];
    c[52] = s[3]*s[3]*s[3];
    c[53] = s[3]*s[3]*s[4];
    c[54] = s[3]*s[4]*s[4];
    c[55] = s[4]*s[4]*s[4];
    for (int i = 0; i < 56; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 56; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order3_Dim5_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order3_Dim5_luminance::G() const {
    float v = X[0];
    int div[56];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 2;
    div[5] = 2;
    div[6] = 3;
    div[7] = 4;
    div[8] = 4;
    div[9] = 4;
    div[10] = 4;
    div[11] = 3;
    div[12] = 4;
    div[13] = 4;
    div[14] = 4;
    div[15] = 3;
    div[16] = 4;
    div[17] = 4;
    div[18] = 3;
    div[19] = 4;
    div[20] = 3;
    div[21] = 4;
    div[22] = 6;
    div[23] = 6;
    div[24] = 6;
    div[25] = 6;
    div[26] = 6;
    div[27] = 8;
    div[28] = 8;
    div[29] = 8;
    div[30] = 6;
    div[31] = 8;
    div[32] = 8;
    div[33] = 6;
    div[34] = 8;
    div[35] = 6;
    div[36] = 4;
    div[37] = 6;
    div[38] = 6;
    div[39] = 6;
    div[40] = 6;
    div[41] = 8;
    div[42] = 8;
    div[43] = 6;
    div[44] = 8;
    div[45] = 6;
    div[46] = 4;
    div[47] = 6;
    div[48] = 6;
    div[49] = 6;
    div[50] = 8;
    div[51] = 6;
    div[52] = 4;
    div[53] = 6;
    div[54] = 6;
    div[55] = 4;
    for (int i = 1; i < 56; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order3_Dim5_luminance::Gpix(Float ax, Float ay, Float bx, Float by) const {
    float v = X[0];
    float f[56];
    f[0] = ((bx - ax)*(by - ay)) / 1;
    f[1] = ((bx*bx - ax*ax)*(by - ay)) / 2;
    f[2] = ((bx - ax)*(by*by - ay*ay)) / 2;
    f[3] = ((bx - ax)*(by - ay)) / 2;
    f[4] = ((bx - ax)*(by - ay)) / 2;
    f[5] = ((bx - ax)*(by - ay)) / 2;
    f[6] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 3;
    f[7] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 4;
    f[8] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[9] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[10] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[11] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 3;
    f[12] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[13] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[14] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[15] = ((bx - ax)*(by - ay)) / 3;
    f[16] = ((bx - ax)*(by - ay)) / 4;
    f[17] = ((bx - ax)*(by - ay)) / 4;
    f[18] = ((bx - ax)*(by - ay)) / 3;
    f[19] = ((bx - ax)*(by - ay)) / 4;
    f[20] = ((bx - ax)*(by - ay)) / 3;
    f[21] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 4;
    f[22] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 6;
    f[23] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 6;
    f[24] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 6;
    f[25] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 6;
    f[26] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 6;
    f[27] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 8;
    f[28] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 8;
    f[29] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 8;
    f[30] = ((bx*bx - ax*ax)*(by - ay)) / 6;
    f[31] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[32] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[33] = ((bx*bx - ax*ax)*(by - ay)) / 6;
    f[34] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[35] = ((bx*bx - ax*ax)*(by - ay)) / 6;
    f[36] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 4;
    f[37] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 6;
    f[38] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 6;
    f[39] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 6;
    f[40] = ((bx - ax)*(by*by - ay*ay)) / 6;
    f[41] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[42] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[43] = ((bx - ax)*(by*by - ay*ay)) / 6;
    f[44] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[45] = ((bx - ax)*(by*by - ay*ay)) / 6;
    f[46] = ((bx - ax)*(by - ay)) / 4;
    f[47] = ((bx - ax)*(by - ay)) / 6;
    f[48] = ((bx - ax)*(by - ay)) / 6;
    f[49] = ((bx - ax)*(by - ay)) / 6;
    f[50] = ((bx - ax)*(by - ay)) / 8;
    f[51] = ((bx - ax)*(by - ay)) / 6;
    f[52] = ((bx - ax)*(by - ay)) / 4;
    f[53] = ((bx - ax)*(by - ay)) / 6;
    f[54] = ((bx - ax)*(by - ay)) / 6;
    f[55] = ((bx - ax)*(by - ay)) / 4;
    for (int i = 1; i < 56; ++i) {
        v += X[i] * f[i];
    }
    return v;
}
float Poly_Order3_Dim5_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[56];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    c[6] = s[0]*s[0];
    c[7] = s[0]*s[1];
    c[8] = s[0]*s[2];
    c[9] = s[0]*s[3];
    c[10] = s[0]*s[4];
    c[11] = s[1]*s[1];
    c[12] = s[1]*s[2];
    c[13] = s[1]*s[3];
    c[14] = s[1]*s[4];
    c[15] = s[2]*s[2];
    c[16] = s[2]*s[3];
    c[17] = s[2]*s[4];
    c[18] = s[3]*s[3];
    c[19] = s[3]*s[4];
    c[20] = s[4]*s[4];
    c[21] = s[0]*s[0]*s[0];
    c[22] = s[0]*s[0]*s[1];
    c[23] = s[0]*s[0]*s[2];
    c[24] = s[0]*s[0]*s[3];
    c[25] = s[0]*s[0]*s[4];
    c[26] = s[0]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[3];
    c[29] = s[0]*s[1]*s[4];
    c[30] = s[0]*s[2]*s[2];
    c[31] = s[0]*s[2]*s[3];
    c[32] = s[0]*s[2]*s[4];
    c[33] = s[0]*s[3]*s[3];
    c[34] = s[0]*s[3]*s[4];
    c[35] = s[0]*s[4]*s[4];
    c[36] = s[1]*s[1]*s[1];
    c[37] = s[1]*s[1]*s[2];
    c[38] = s[1]*s[1]*s[3];
    c[39] = s[1]*s[1]*s[4];
    c[40] = s[1]*s[2]*s[2];
    c[41] = s[1]*s[2]*s[3];
    c[42] = s[1]*s[2]*s[4];
    c[43] = s[1]*s[3]*s[3];
    c[44] = s[1]*s[3]*s[4];
    c[45] = s[1]*s[4]*s[4];
    c[46] = s[2]*s[2]*s[2];
    c[47] = s[2]*s[2]*s[3];
    c[48] = s[2]*s[2]*s[4];
    c[49] = s[2]*s[3]*s[3];
    c[50] = s[2]*s[3]*s[4];
    c[51] = s[2]*s[4]*s[4];
    c[52] = s[3]*s[3]*s[3];
    c[53] = s[3]*s[3]*s[4];
    c[54] = s[3]*s[4]*s[4];
    c[55] = s[4]*s[4]*s[4];
    for (int i = 1; i < 56; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order3_Dim5_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[56];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    c[6] = s[0]*s[0];
    c[7] = s[0]*s[1];
    c[8] = s[0]*s[2];
    c[9] = s[0]*s[3];
    c[10] = s[0]*s[4];
    c[11] = s[1]*s[1];
    c[12] = s[1]*s[2];
    c[13] = s[1]*s[3];
    c[14] = s[1]*s[4];
    c[15] = s[2]*s[2];
    c[16] = s[2]*s[3];
    c[17] = s[2]*s[4];
    c[18] = s[3]*s[3];
    c[19] = s[3]*s[4];
    c[20] = s[4]*s[4];
    c[21] = s[0]*s[0]*s[0];
    c[22] = s[0]*s[0]*s[1];
    c[23] = s[0]*s[0]*s[2];
    c[24] = s[0]*s[0]*s[3];
    c[25] = s[0]*s[0]*s[4];
    c[26] = s[0]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[3];
    c[29] = s[0]*s[1]*s[4];
    c[30] = s[0]*s[2]*s[2];
    c[31] = s[0]*s[2]*s[3];
    c[32] = s[0]*s[2]*s[4];
    c[33] = s[0]*s[3]*s[3];
    c[34] = s[0]*s[3]*s[4];
    c[35] = s[0]*s[4]*s[4];
    c[36] = s[1]*s[1]*s[1];
    c[37] = s[1]*s[1]*s[2];
    c[38] = s[1]*s[1]*s[3];
    c[39] = s[1]*s[1]*s[4];
    c[40] = s[1]*s[2]*s[2];
    c[41] = s[1]*s[2]*s[3];
    c[42] = s[1]*s[2]*s[4];
    c[43] = s[1]*s[3]*s[3];
    c[44] = s[1]*s[3]*s[4];
    c[45] = s[1]*s[4]*s[4];
    c[46] = s[2]*s[2]*s[2];
    c[47] = s[2]*s[2]*s[3];
    c[48] = s[2]*s[2]*s[4];
    c[49] = s[2]*s[3]*s[3];
    c[50] = s[2]*s[3]*s[4];
    c[51] = s[2]*s[4]*s[4];
    c[52] = s[3]*s[3]*s[3];
    c[53] = s[3]*s[3]*s[4];
    c[54] = s[3]*s[4]*s[4];
    c[55] = s[4]*s[4]*s[4];
    Float e = (g(s) - v);
    for (int i = 0; i < 56; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order3_Dim5_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[56];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    c[6] = s[0]*s[0];
    c[7] = s[0]*s[1];
    c[8] = s[0]*s[2];
    c[9] = s[0]*s[3];
    c[10] = s[0]*s[4];
    c[11] = s[1]*s[1];
    c[12] = s[1]*s[2];
    c[13] = s[1]*s[3];
    c[14] = s[1]*s[4];
    c[15] = s[2]*s[2];
    c[16] = s[2]*s[3];
    c[17] = s[2]*s[4];
    c[18] = s[3]*s[3];
    c[19] = s[3]*s[4];
    c[20] = s[4]*s[4];
    c[21] = s[0]*s[0]*s[0];
    c[22] = s[0]*s[0]*s[1];
    c[23] = s[0]*s[0]*s[2];
    c[24] = s[0]*s[0]*s[3];
    c[25] = s[0]*s[0]*s[4];
    c[26] = s[0]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[3];
    c[29] = s[0]*s[1]*s[4];
    c[30] = s[0]*s[2]*s[2];
    c[31] = s[0]*s[2]*s[3];
    c[32] = s[0]*s[2]*s[4];
    c[33] = s[0]*s[3]*s[3];
    c[34] = s[0]*s[3]*s[4];
    c[35] = s[0]*s[4]*s[4];
    c[36] = s[1]*s[1]*s[1];
    c[37] = s[1]*s[1]*s[2];
    c[38] = s[1]*s[1]*s[3];
    c[39] = s[1]*s[1]*s[4];
    c[40] = s[1]*s[2]*s[2];
    c[41] = s[1]*s[2]*s[3];
    c[42] = s[1]*s[2]*s[4];
    c[43] = s[1]*s[3]*s[3];
    c[44] = s[1]*s[3]*s[4];
    c[45] = s[1]*s[4]*s[4];
    c[46] = s[2]*s[2]*s[2];
    c[47] = s[2]*s[2]*s[3];
    c[48] = s[2]*s[2]*s[4];
    c[49] = s[2]*s[3]*s[3];
    c[50] = s[2]*s[3]*s[4];
    c[51] = s[2]*s[4]*s[4];
    c[52] = s[3]*s[3]*s[3];
    c[53] = s[3]*s[3]*s[4];
    c[54] = s[3]*s[4]*s[4];
    c[55] = s[4]*s[4]*s[4];
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
void Poly_Order4_Dim5_luminance::reset() {
    A.setZero();
    b.setZero();
    X.setZero();
}
void Poly_Order4_Dim5_luminance::update( const Eigen::VectorXd &s, const Spectrum &fx) {
    float l = fx.y();

    float c[126];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    c[6] = s[0]*s[0];
    c[7] = s[0]*s[1];
    c[8] = s[0]*s[2];
    c[9] = s[0]*s[3];
    c[10] = s[0]*s[4];
    c[11] = s[1]*s[1];
    c[12] = s[1]*s[2];
    c[13] = s[1]*s[3];
    c[14] = s[1]*s[4];
    c[15] = s[2]*s[2];
    c[16] = s[2]*s[3];
    c[17] = s[2]*s[4];
    c[18] = s[3]*s[3];
    c[19] = s[3]*s[4];
    c[20] = s[4]*s[4];
    c[21] = s[0]*s[0]*s[0];
    c[22] = s[0]*s[0]*s[1];
    c[23] = s[0]*s[0]*s[2];
    c[24] = s[0]*s[0]*s[3];
    c[25] = s[0]*s[0]*s[4];
    c[26] = s[0]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[3];
    c[29] = s[0]*s[1]*s[4];
    c[30] = s[0]*s[2]*s[2];
    c[31] = s[0]*s[2]*s[3];
    c[32] = s[0]*s[2]*s[4];
    c[33] = s[0]*s[3]*s[3];
    c[34] = s[0]*s[3]*s[4];
    c[35] = s[0]*s[4]*s[4];
    c[36] = s[1]*s[1]*s[1];
    c[37] = s[1]*s[1]*s[2];
    c[38] = s[1]*s[1]*s[3];
    c[39] = s[1]*s[1]*s[4];
    c[40] = s[1]*s[2]*s[2];
    c[41] = s[1]*s[2]*s[3];
    c[42] = s[1]*s[2]*s[4];
    c[43] = s[1]*s[3]*s[3];
    c[44] = s[1]*s[3]*s[4];
    c[45] = s[1]*s[4]*s[4];
    c[46] = s[2]*s[2]*s[2];
    c[47] = s[2]*s[2]*s[3];
    c[48] = s[2]*s[2]*s[4];
    c[49] = s[2]*s[3]*s[3];
    c[50] = s[2]*s[3]*s[4];
    c[51] = s[2]*s[4]*s[4];
    c[52] = s[3]*s[3]*s[3];
    c[53] = s[3]*s[3]*s[4];
    c[54] = s[3]*s[4]*s[4];
    c[55] = s[4]*s[4]*s[4];
    c[56] = s[0]*s[0]*s[0]*s[0];
    c[57] = s[0]*s[0]*s[0]*s[1];
    c[58] = s[0]*s[0]*s[0]*s[2];
    c[59] = s[0]*s[0]*s[0]*s[3];
    c[60] = s[0]*s[0]*s[0]*s[4];
    c[61] = s[0]*s[0]*s[1]*s[1];
    c[62] = s[0]*s[0]*s[1]*s[2];
    c[63] = s[0]*s[0]*s[1]*s[3];
    c[64] = s[0]*s[0]*s[1]*s[4];
    c[65] = s[0]*s[0]*s[2]*s[2];
    c[66] = s[0]*s[0]*s[2]*s[3];
    c[67] = s[0]*s[0]*s[2]*s[4];
    c[68] = s[0]*s[0]*s[3]*s[3];
    c[69] = s[0]*s[0]*s[3]*s[4];
    c[70] = s[0]*s[0]*s[4]*s[4];
    c[71] = s[0]*s[1]*s[1]*s[1];
    c[72] = s[0]*s[1]*s[1]*s[2];
    c[73] = s[0]*s[1]*s[1]*s[3];
    c[74] = s[0]*s[1]*s[1]*s[4];
    c[75] = s[0]*s[1]*s[2]*s[2];
    c[76] = s[0]*s[1]*s[2]*s[3];
    c[77] = s[0]*s[1]*s[2]*s[4];
    c[78] = s[0]*s[1]*s[3]*s[3];
    c[79] = s[0]*s[1]*s[3]*s[4];
    c[80] = s[0]*s[1]*s[4]*s[4];
    c[81] = s[0]*s[2]*s[2]*s[2];
    c[82] = s[0]*s[2]*s[2]*s[3];
    c[83] = s[0]*s[2]*s[2]*s[4];
    c[84] = s[0]*s[2]*s[3]*s[3];
    c[85] = s[0]*s[2]*s[3]*s[4];
    c[86] = s[0]*s[2]*s[4]*s[4];
    c[87] = s[0]*s[3]*s[3]*s[3];
    c[88] = s[0]*s[3]*s[3]*s[4];
    c[89] = s[0]*s[3]*s[4]*s[4];
    c[90] = s[0]*s[4]*s[4]*s[4];
    c[91] = s[1]*s[1]*s[1]*s[1];
    c[92] = s[1]*s[1]*s[1]*s[2];
    c[93] = s[1]*s[1]*s[1]*s[3];
    c[94] = s[1]*s[1]*s[1]*s[4];
    c[95] = s[1]*s[1]*s[2]*s[2];
    c[96] = s[1]*s[1]*s[2]*s[3];
    c[97] = s[1]*s[1]*s[2]*s[4];
    c[98] = s[1]*s[1]*s[3]*s[3];
    c[99] = s[1]*s[1]*s[3]*s[4];
    c[100] = s[1]*s[1]*s[4]*s[4];
    c[101] = s[1]*s[2]*s[2]*s[2];
    c[102] = s[1]*s[2]*s[2]*s[3];
    c[103] = s[1]*s[2]*s[2]*s[4];
    c[104] = s[1]*s[2]*s[3]*s[3];
    c[105] = s[1]*s[2]*s[3]*s[4];
    c[106] = s[1]*s[2]*s[4]*s[4];
    c[107] = s[1]*s[3]*s[3]*s[3];
    c[108] = s[1]*s[3]*s[3]*s[4];
    c[109] = s[1]*s[3]*s[4]*s[4];
    c[110] = s[1]*s[4]*s[4]*s[4];
    c[111] = s[2]*s[2]*s[2]*s[2];
    c[112] = s[2]*s[2]*s[2]*s[3];
    c[113] = s[2]*s[2]*s[2]*s[4];
    c[114] = s[2]*s[2]*s[3]*s[3];
    c[115] = s[2]*s[2]*s[3]*s[4];
    c[116] = s[2]*s[2]*s[4]*s[4];
    c[117] = s[2]*s[3]*s[3]*s[3];
    c[118] = s[2]*s[3]*s[3]*s[4];
    c[119] = s[2]*s[3]*s[4]*s[4];
    c[120] = s[2]*s[4]*s[4]*s[4];
    c[121] = s[3]*s[3]*s[3]*s[3];
    c[122] = s[3]*s[3]*s[3]*s[4];
    c[123] = s[3]*s[3]*s[4]*s[4];
    c[124] = s[3]*s[4]*s[4]*s[4];
    c[125] = s[4]*s[4]*s[4]*s[4];
    for (int i = 0; i < 126; ++i) {
        float x1 = c[i];
        b(i) += x1 * l;
        for (int j = i; j < 126; ++j) {
            A(i, j) += x1 * c[j];
        }
    }
}
void Poly_Order4_Dim5_luminance::solve() {
    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }
    X = A.completeOrthogonalDecomposition().solve(b);
}
float Poly_Order4_Dim5_luminance::G() const {
    float v = X[0];
    int div[126];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 2;
    div[5] = 2;
    div[6] = 3;
    div[7] = 4;
    div[8] = 4;
    div[9] = 4;
    div[10] = 4;
    div[11] = 3;
    div[12] = 4;
    div[13] = 4;
    div[14] = 4;
    div[15] = 3;
    div[16] = 4;
    div[17] = 4;
    div[18] = 3;
    div[19] = 4;
    div[20] = 3;
    div[21] = 4;
    div[22] = 6;
    div[23] = 6;
    div[24] = 6;
    div[25] = 6;
    div[26] = 6;
    div[27] = 8;
    div[28] = 8;
    div[29] = 8;
    div[30] = 6;
    div[31] = 8;
    div[32] = 8;
    div[33] = 6;
    div[34] = 8;
    div[35] = 6;
    div[36] = 4;
    div[37] = 6;
    div[38] = 6;
    div[39] = 6;
    div[40] = 6;
    div[41] = 8;
    div[42] = 8;
    div[43] = 6;
    div[44] = 8;
    div[45] = 6;
    div[46] = 4;
    div[47] = 6;
    div[48] = 6;
    div[49] = 6;
    div[50] = 8;
    div[51] = 6;
    div[52] = 4;
    div[53] = 6;
    div[54] = 6;
    div[55] = 4;
    div[56] = 5;
    div[57] = 8;
    div[58] = 8;
    div[59] = 8;
    div[60] = 8;
    div[61] = 9;
    div[62] = 12;
    div[63] = 12;
    div[64] = 12;
    div[65] = 9;
    div[66] = 12;
    div[67] = 12;
    div[68] = 9;
    div[69] = 12;
    div[70] = 9;
    div[71] = 8;
    div[72] = 12;
    div[73] = 12;
    div[74] = 12;
    div[75] = 12;
    div[76] = 16;
    div[77] = 16;
    div[78] = 12;
    div[79] = 16;
    div[80] = 12;
    div[81] = 8;
    div[82] = 12;
    div[83] = 12;
    div[84] = 12;
    div[85] = 16;
    div[86] = 12;
    div[87] = 8;
    div[88] = 12;
    div[89] = 12;
    div[90] = 8;
    div[91] = 5;
    div[92] = 8;
    div[93] = 8;
    div[94] = 8;
    div[95] = 9;
    div[96] = 12;
    div[97] = 12;
    div[98] = 9;
    div[99] = 12;
    div[100] = 9;
    div[101] = 8;
    div[102] = 12;
    div[103] = 12;
    div[104] = 12;
    div[105] = 16;
    div[106] = 12;
    div[107] = 8;
    div[108] = 12;
    div[109] = 12;
    div[110] = 8;
    div[111] = 5;
    div[112] = 8;
    div[113] = 8;
    div[114] = 9;
    div[115] = 12;
    div[116] = 9;
    div[117] = 8;
    div[118] = 12;
    div[119] = 12;
    div[120] = 8;
    div[121] = 5;
    div[122] = 8;
    div[123] = 9;
    div[124] = 8;
    div[125] = 5;
    for (int i = 1; i < 126; ++i) {
        v += X[i] / div[i];
    }
    return v;
}
float Poly_Order4_Dim5_luminance::Gpix(Float ax, Float ay, Float bx, Float by) const {
    float v = X[0];
    float f[126];
    f[0] = ((bx - ax)*(by - ay)) / 1;
    f[1] = ((bx*bx - ax*ax)*(by - ay)) / 2;
    f[2] = ((bx - ax)*(by*by - ay*ay)) / 2;
    f[3] = ((bx - ax)*(by - ay)) / 2;
    f[4] = ((bx - ax)*(by - ay)) / 2;
    f[5] = ((bx - ax)*(by - ay)) / 2;
    f[6] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 3;
    f[7] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 4;
    f[8] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[9] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[10] = ((bx*bx - ax*ax)*(by - ay)) / 4;
    f[11] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 3;
    f[12] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[13] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[14] = ((bx - ax)*(by*by - ay*ay)) / 4;
    f[15] = ((bx - ax)*(by - ay)) / 3;
    f[16] = ((bx - ax)*(by - ay)) / 4;
    f[17] = ((bx - ax)*(by - ay)) / 4;
    f[18] = ((bx - ax)*(by - ay)) / 3;
    f[19] = ((bx - ax)*(by - ay)) / 4;
    f[20] = ((bx - ax)*(by - ay)) / 3;
    f[21] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 4;
    f[22] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 6;
    f[23] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 6;
    f[24] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 6;
    f[25] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 6;
    f[26] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 6;
    f[27] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 8;
    f[28] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 8;
    f[29] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 8;
    f[30] = ((bx*bx - ax*ax)*(by - ay)) / 6;
    f[31] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[32] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[33] = ((bx*bx - ax*ax)*(by - ay)) / 6;
    f[34] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[35] = ((bx*bx - ax*ax)*(by - ay)) / 6;
    f[36] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 4;
    f[37] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 6;
    f[38] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 6;
    f[39] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 6;
    f[40] = ((bx - ax)*(by*by - ay*ay)) / 6;
    f[41] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[42] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[43] = ((bx - ax)*(by*by - ay*ay)) / 6;
    f[44] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[45] = ((bx - ax)*(by*by - ay*ay)) / 6;
    f[46] = ((bx - ax)*(by - ay)) / 4;
    f[47] = ((bx - ax)*(by - ay)) / 6;
    f[48] = ((bx - ax)*(by - ay)) / 6;
    f[49] = ((bx - ax)*(by - ay)) / 6;
    f[50] = ((bx - ax)*(by - ay)) / 8;
    f[51] = ((bx - ax)*(by - ay)) / 6;
    f[52] = ((bx - ax)*(by - ay)) / 4;
    f[53] = ((bx - ax)*(by - ay)) / 6;
    f[54] = ((bx - ax)*(by - ay)) / 6;
    f[55] = ((bx - ax)*(by - ay)) / 4;
    f[56] = ((bx*bx*bx*bx*bx - ax*ax*ax*ax*ax)*(by - ay)) / 5;
    f[57] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by*by - ay*ay)) / 8;
    f[58] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 8;
    f[59] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 8;
    f[60] = ((bx*bx*bx*bx - ax*ax*ax*ax)*(by - ay)) / 8;
    f[61] = ((bx*bx*bx - ax*ax*ax)*(by*by*by - ay*ay*ay)) / 9;
    f[62] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 12;
    f[63] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 12;
    f[64] = ((bx*bx*bx - ax*ax*ax)*(by*by - ay*ay)) / 12;
    f[65] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 9;
    f[66] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 12;
    f[67] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 12;
    f[68] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 9;
    f[69] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 12;
    f[70] = ((bx*bx*bx - ax*ax*ax)*(by - ay)) / 9;
    f[71] = ((bx*bx - ax*ax)*(by*by*by*by - ay*ay*ay*ay)) / 8;
    f[72] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 12;
    f[73] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 12;
    f[74] = ((bx*bx - ax*ax)*(by*by*by - ay*ay*ay)) / 12;
    f[75] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 12;
    f[76] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 16;
    f[77] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 16;
    f[78] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 12;
    f[79] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 16;
    f[80] = ((bx*bx - ax*ax)*(by*by - ay*ay)) / 12;
    f[81] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[82] = ((bx*bx - ax*ax)*(by - ay)) / 12;
    f[83] = ((bx*bx - ax*ax)*(by - ay)) / 12;
    f[84] = ((bx*bx - ax*ax)*(by - ay)) / 12;
    f[85] = ((bx*bx - ax*ax)*(by - ay)) / 16;
    f[86] = ((bx*bx - ax*ax)*(by - ay)) / 12;
    f[87] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[88] = ((bx*bx - ax*ax)*(by - ay)) / 12;
    f[89] = ((bx*bx - ax*ax)*(by - ay)) / 12;
    f[90] = ((bx*bx - ax*ax)*(by - ay)) / 8;
    f[91] = ((bx - ax)*(by*by*by*by*by - ay*ay*ay*ay*ay)) / 5;
    f[92] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 8;
    f[93] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 8;
    f[94] = ((bx - ax)*(by*by*by*by - ay*ay*ay*ay)) / 8;
    f[95] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 9;
    f[96] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 12;
    f[97] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 12;
    f[98] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 9;
    f[99] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 12;
    f[100] = ((bx - ax)*(by*by*by - ay*ay*ay)) / 9;
    f[101] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[102] = ((bx - ax)*(by*by - ay*ay)) / 12;
    f[103] = ((bx - ax)*(by*by - ay*ay)) / 12;
    f[104] = ((bx - ax)*(by*by - ay*ay)) / 12;
    f[105] = ((bx - ax)*(by*by - ay*ay)) / 16;
    f[106] = ((bx - ax)*(by*by - ay*ay)) / 12;
    f[107] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[108] = ((bx - ax)*(by*by - ay*ay)) / 12;
    f[109] = ((bx - ax)*(by*by - ay*ay)) / 12;
    f[110] = ((bx - ax)*(by*by - ay*ay)) / 8;
    f[111] = ((bx - ax)*(by - ay)) / 5;
    f[112] = ((bx - ax)*(by - ay)) / 8;
    f[113] = ((bx - ax)*(by - ay)) / 8;
    f[114] = ((bx - ax)*(by - ay)) / 9;
    f[115] = ((bx - ax)*(by - ay)) / 12;
    f[116] = ((bx - ax)*(by - ay)) / 9;
    f[117] = ((bx - ax)*(by - ay)) / 8;
    f[118] = ((bx - ax)*(by - ay)) / 12;
    f[119] = ((bx - ax)*(by - ay)) / 12;
    f[120] = ((bx - ax)*(by - ay)) / 8;
    f[121] = ((bx - ax)*(by - ay)) / 5;
    f[122] = ((bx - ax)*(by - ay)) / 8;
    f[123] = ((bx - ax)*(by - ay)) / 9;
    f[124] = ((bx - ax)*(by - ay)) / 8;
    f[125] = ((bx - ax)*(by - ay)) / 5;
    for (int i = 1; i < 126; ++i) {
        v += X[i] * f[i];
    }
    return v;
}
float Poly_Order4_Dim5_luminance::g( const Eigen::VectorXd &s) const {
    float v = X[0];
    float c[126];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    c[6] = s[0]*s[0];
    c[7] = s[0]*s[1];
    c[8] = s[0]*s[2];
    c[9] = s[0]*s[3];
    c[10] = s[0]*s[4];
    c[11] = s[1]*s[1];
    c[12] = s[1]*s[2];
    c[13] = s[1]*s[3];
    c[14] = s[1]*s[4];
    c[15] = s[2]*s[2];
    c[16] = s[2]*s[3];
    c[17] = s[2]*s[4];
    c[18] = s[3]*s[3];
    c[19] = s[3]*s[4];
    c[20] = s[4]*s[4];
    c[21] = s[0]*s[0]*s[0];
    c[22] = s[0]*s[0]*s[1];
    c[23] = s[0]*s[0]*s[2];
    c[24] = s[0]*s[0]*s[3];
    c[25] = s[0]*s[0]*s[4];
    c[26] = s[0]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[3];
    c[29] = s[0]*s[1]*s[4];
    c[30] = s[0]*s[2]*s[2];
    c[31] = s[0]*s[2]*s[3];
    c[32] = s[0]*s[2]*s[4];
    c[33] = s[0]*s[3]*s[3];
    c[34] = s[0]*s[3]*s[4];
    c[35] = s[0]*s[4]*s[4];
    c[36] = s[1]*s[1]*s[1];
    c[37] = s[1]*s[1]*s[2];
    c[38] = s[1]*s[1]*s[3];
    c[39] = s[1]*s[1]*s[4];
    c[40] = s[1]*s[2]*s[2];
    c[41] = s[1]*s[2]*s[3];
    c[42] = s[1]*s[2]*s[4];
    c[43] = s[1]*s[3]*s[3];
    c[44] = s[1]*s[3]*s[4];
    c[45] = s[1]*s[4]*s[4];
    c[46] = s[2]*s[2]*s[2];
    c[47] = s[2]*s[2]*s[3];
    c[48] = s[2]*s[2]*s[4];
    c[49] = s[2]*s[3]*s[3];
    c[50] = s[2]*s[3]*s[4];
    c[51] = s[2]*s[4]*s[4];
    c[52] = s[3]*s[3]*s[3];
    c[53] = s[3]*s[3]*s[4];
    c[54] = s[3]*s[4]*s[4];
    c[55] = s[4]*s[4]*s[4];
    c[56] = s[0]*s[0]*s[0]*s[0];
    c[57] = s[0]*s[0]*s[0]*s[1];
    c[58] = s[0]*s[0]*s[0]*s[2];
    c[59] = s[0]*s[0]*s[0]*s[3];
    c[60] = s[0]*s[0]*s[0]*s[4];
    c[61] = s[0]*s[0]*s[1]*s[1];
    c[62] = s[0]*s[0]*s[1]*s[2];
    c[63] = s[0]*s[0]*s[1]*s[3];
    c[64] = s[0]*s[0]*s[1]*s[4];
    c[65] = s[0]*s[0]*s[2]*s[2];
    c[66] = s[0]*s[0]*s[2]*s[3];
    c[67] = s[0]*s[0]*s[2]*s[4];
    c[68] = s[0]*s[0]*s[3]*s[3];
    c[69] = s[0]*s[0]*s[3]*s[4];
    c[70] = s[0]*s[0]*s[4]*s[4];
    c[71] = s[0]*s[1]*s[1]*s[1];
    c[72] = s[0]*s[1]*s[1]*s[2];
    c[73] = s[0]*s[1]*s[1]*s[3];
    c[74] = s[0]*s[1]*s[1]*s[4];
    c[75] = s[0]*s[1]*s[2]*s[2];
    c[76] = s[0]*s[1]*s[2]*s[3];
    c[77] = s[0]*s[1]*s[2]*s[4];
    c[78] = s[0]*s[1]*s[3]*s[3];
    c[79] = s[0]*s[1]*s[3]*s[4];
    c[80] = s[0]*s[1]*s[4]*s[4];
    c[81] = s[0]*s[2]*s[2]*s[2];
    c[82] = s[0]*s[2]*s[2]*s[3];
    c[83] = s[0]*s[2]*s[2]*s[4];
    c[84] = s[0]*s[2]*s[3]*s[3];
    c[85] = s[0]*s[2]*s[3]*s[4];
    c[86] = s[0]*s[2]*s[4]*s[4];
    c[87] = s[0]*s[3]*s[3]*s[3];
    c[88] = s[0]*s[3]*s[3]*s[4];
    c[89] = s[0]*s[3]*s[4]*s[4];
    c[90] = s[0]*s[4]*s[4]*s[4];
    c[91] = s[1]*s[1]*s[1]*s[1];
    c[92] = s[1]*s[1]*s[1]*s[2];
    c[93] = s[1]*s[1]*s[1]*s[3];
    c[94] = s[1]*s[1]*s[1]*s[4];
    c[95] = s[1]*s[1]*s[2]*s[2];
    c[96] = s[1]*s[1]*s[2]*s[3];
    c[97] = s[1]*s[1]*s[2]*s[4];
    c[98] = s[1]*s[1]*s[3]*s[3];
    c[99] = s[1]*s[1]*s[3]*s[4];
    c[100] = s[1]*s[1]*s[4]*s[4];
    c[101] = s[1]*s[2]*s[2]*s[2];
    c[102] = s[1]*s[2]*s[2]*s[3];
    c[103] = s[1]*s[2]*s[2]*s[4];
    c[104] = s[1]*s[2]*s[3]*s[3];
    c[105] = s[1]*s[2]*s[3]*s[4];
    c[106] = s[1]*s[2]*s[4]*s[4];
    c[107] = s[1]*s[3]*s[3]*s[3];
    c[108] = s[1]*s[3]*s[3]*s[4];
    c[109] = s[1]*s[3]*s[4]*s[4];
    c[110] = s[1]*s[4]*s[4]*s[4];
    c[111] = s[2]*s[2]*s[2]*s[2];
    c[112] = s[2]*s[2]*s[2]*s[3];
    c[113] = s[2]*s[2]*s[2]*s[4];
    c[114] = s[2]*s[2]*s[3]*s[3];
    c[115] = s[2]*s[2]*s[3]*s[4];
    c[116] = s[2]*s[2]*s[4]*s[4];
    c[117] = s[2]*s[3]*s[3]*s[3];
    c[118] = s[2]*s[3]*s[3]*s[4];
    c[119] = s[2]*s[3]*s[4]*s[4];
    c[120] = s[2]*s[4]*s[4]*s[4];
    c[121] = s[3]*s[3]*s[3]*s[3];
    c[122] = s[3]*s[3]*s[3]*s[4];
    c[123] = s[3]*s[3]*s[4]*s[4];
    c[124] = s[3]*s[4]*s[4]*s[4];
    c[125] = s[4]*s[4]*s[4]*s[4];
    for (int i = 1; i < 126; ++i) {
        v += c[i] * X[i];
    }
    return v;
}
void Poly_Order4_Dim5_luminance::sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    float c[126];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    c[6] = s[0]*s[0];
    c[7] = s[0]*s[1];
    c[8] = s[0]*s[2];
    c[9] = s[0]*s[3];
    c[10] = s[0]*s[4];
    c[11] = s[1]*s[1];
    c[12] = s[1]*s[2];
    c[13] = s[1]*s[3];
    c[14] = s[1]*s[4];
    c[15] = s[2]*s[2];
    c[16] = s[2]*s[3];
    c[17] = s[2]*s[4];
    c[18] = s[3]*s[3];
    c[19] = s[3]*s[4];
    c[20] = s[4]*s[4];
    c[21] = s[0]*s[0]*s[0];
    c[22] = s[0]*s[0]*s[1];
    c[23] = s[0]*s[0]*s[2];
    c[24] = s[0]*s[0]*s[3];
    c[25] = s[0]*s[0]*s[4];
    c[26] = s[0]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[3];
    c[29] = s[0]*s[1]*s[4];
    c[30] = s[0]*s[2]*s[2];
    c[31] = s[0]*s[2]*s[3];
    c[32] = s[0]*s[2]*s[4];
    c[33] = s[0]*s[3]*s[3];
    c[34] = s[0]*s[3]*s[4];
    c[35] = s[0]*s[4]*s[4];
    c[36] = s[1]*s[1]*s[1];
    c[37] = s[1]*s[1]*s[2];
    c[38] = s[1]*s[1]*s[3];
    c[39] = s[1]*s[1]*s[4];
    c[40] = s[1]*s[2]*s[2];
    c[41] = s[1]*s[2]*s[3];
    c[42] = s[1]*s[2]*s[4];
    c[43] = s[1]*s[3]*s[3];
    c[44] = s[1]*s[3]*s[4];
    c[45] = s[1]*s[4]*s[4];
    c[46] = s[2]*s[2]*s[2];
    c[47] = s[2]*s[2]*s[3];
    c[48] = s[2]*s[2]*s[4];
    c[49] = s[2]*s[3]*s[3];
    c[50] = s[2]*s[3]*s[4];
    c[51] = s[2]*s[4]*s[4];
    c[52] = s[3]*s[3]*s[3];
    c[53] = s[3]*s[3]*s[4];
    c[54] = s[3]*s[4]*s[4];
    c[55] = s[4]*s[4]*s[4];
    c[56] = s[0]*s[0]*s[0]*s[0];
    c[57] = s[0]*s[0]*s[0]*s[1];
    c[58] = s[0]*s[0]*s[0]*s[2];
    c[59] = s[0]*s[0]*s[0]*s[3];
    c[60] = s[0]*s[0]*s[0]*s[4];
    c[61] = s[0]*s[0]*s[1]*s[1];
    c[62] = s[0]*s[0]*s[1]*s[2];
    c[63] = s[0]*s[0]*s[1]*s[3];
    c[64] = s[0]*s[0]*s[1]*s[4];
    c[65] = s[0]*s[0]*s[2]*s[2];
    c[66] = s[0]*s[0]*s[2]*s[3];
    c[67] = s[0]*s[0]*s[2]*s[4];
    c[68] = s[0]*s[0]*s[3]*s[3];
    c[69] = s[0]*s[0]*s[3]*s[4];
    c[70] = s[0]*s[0]*s[4]*s[4];
    c[71] = s[0]*s[1]*s[1]*s[1];
    c[72] = s[0]*s[1]*s[1]*s[2];
    c[73] = s[0]*s[1]*s[1]*s[3];
    c[74] = s[0]*s[1]*s[1]*s[4];
    c[75] = s[0]*s[1]*s[2]*s[2];
    c[76] = s[0]*s[1]*s[2]*s[3];
    c[77] = s[0]*s[1]*s[2]*s[4];
    c[78] = s[0]*s[1]*s[3]*s[3];
    c[79] = s[0]*s[1]*s[3]*s[4];
    c[80] = s[0]*s[1]*s[4]*s[4];
    c[81] = s[0]*s[2]*s[2]*s[2];
    c[82] = s[0]*s[2]*s[2]*s[3];
    c[83] = s[0]*s[2]*s[2]*s[4];
    c[84] = s[0]*s[2]*s[3]*s[3];
    c[85] = s[0]*s[2]*s[3]*s[4];
    c[86] = s[0]*s[2]*s[4]*s[4];
    c[87] = s[0]*s[3]*s[3]*s[3];
    c[88] = s[0]*s[3]*s[3]*s[4];
    c[89] = s[0]*s[3]*s[4]*s[4];
    c[90] = s[0]*s[4]*s[4]*s[4];
    c[91] = s[1]*s[1]*s[1]*s[1];
    c[92] = s[1]*s[1]*s[1]*s[2];
    c[93] = s[1]*s[1]*s[1]*s[3];
    c[94] = s[1]*s[1]*s[1]*s[4];
    c[95] = s[1]*s[1]*s[2]*s[2];
    c[96] = s[1]*s[1]*s[2]*s[3];
    c[97] = s[1]*s[1]*s[2]*s[4];
    c[98] = s[1]*s[1]*s[3]*s[3];
    c[99] = s[1]*s[1]*s[3]*s[4];
    c[100] = s[1]*s[1]*s[4]*s[4];
    c[101] = s[1]*s[2]*s[2]*s[2];
    c[102] = s[1]*s[2]*s[2]*s[3];
    c[103] = s[1]*s[2]*s[2]*s[4];
    c[104] = s[1]*s[2]*s[3]*s[3];
    c[105] = s[1]*s[2]*s[3]*s[4];
    c[106] = s[1]*s[2]*s[4]*s[4];
    c[107] = s[1]*s[3]*s[3]*s[3];
    c[108] = s[1]*s[3]*s[3]*s[4];
    c[109] = s[1]*s[3]*s[4]*s[4];
    c[110] = s[1]*s[4]*s[4]*s[4];
    c[111] = s[2]*s[2]*s[2]*s[2];
    c[112] = s[2]*s[2]*s[2]*s[3];
    c[113] = s[2]*s[2]*s[2]*s[4];
    c[114] = s[2]*s[2]*s[3]*s[3];
    c[115] = s[2]*s[2]*s[3]*s[4];
    c[116] = s[2]*s[2]*s[4]*s[4];
    c[117] = s[2]*s[3]*s[3]*s[3];
    c[118] = s[2]*s[3]*s[3]*s[4];
    c[119] = s[2]*s[3]*s[4]*s[4];
    c[120] = s[2]*s[4]*s[4]*s[4];
    c[121] = s[3]*s[3]*s[3]*s[3];
    c[122] = s[3]*s[3]*s[3]*s[4];
    c[123] = s[3]*s[3]*s[4]*s[4];
    c[124] = s[3]*s[4]*s[4]*s[4];
    c[125] = s[4]*s[4]*s[4]*s[4];
    Float e = (g(s) - v);
    for (int i = 0; i < 126; ++i) {
        X(i) -= lr * 2 * c[i] * e;
    }
}
float Poly_Order4_Dim5_luminance::g_then_sgd( const Eigen::VectorXd &s, Float v, const Float lr) {
    // Evaluation g
    float g = X[0];
    float c[126];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    c[4] = s[3];
    c[5] = s[4];
    c[6] = s[0]*s[0];
    c[7] = s[0]*s[1];
    c[8] = s[0]*s[2];
    c[9] = s[0]*s[3];
    c[10] = s[0]*s[4];
    c[11] = s[1]*s[1];
    c[12] = s[1]*s[2];
    c[13] = s[1]*s[3];
    c[14] = s[1]*s[4];
    c[15] = s[2]*s[2];
    c[16] = s[2]*s[3];
    c[17] = s[2]*s[4];
    c[18] = s[3]*s[3];
    c[19] = s[3]*s[4];
    c[20] = s[4]*s[4];
    c[21] = s[0]*s[0]*s[0];
    c[22] = s[0]*s[0]*s[1];
    c[23] = s[0]*s[0]*s[2];
    c[24] = s[0]*s[0]*s[3];
    c[25] = s[0]*s[0]*s[4];
    c[26] = s[0]*s[1]*s[1];
    c[27] = s[0]*s[1]*s[2];
    c[28] = s[0]*s[1]*s[3];
    c[29] = s[0]*s[1]*s[4];
    c[30] = s[0]*s[2]*s[2];
    c[31] = s[0]*s[2]*s[3];
    c[32] = s[0]*s[2]*s[4];
    c[33] = s[0]*s[3]*s[3];
    c[34] = s[0]*s[3]*s[4];
    c[35] = s[0]*s[4]*s[4];
    c[36] = s[1]*s[1]*s[1];
    c[37] = s[1]*s[1]*s[2];
    c[38] = s[1]*s[1]*s[3];
    c[39] = s[1]*s[1]*s[4];
    c[40] = s[1]*s[2]*s[2];
    c[41] = s[1]*s[2]*s[3];
    c[42] = s[1]*s[2]*s[4];
    c[43] = s[1]*s[3]*s[3];
    c[44] = s[1]*s[3]*s[4];
    c[45] = s[1]*s[4]*s[4];
    c[46] = s[2]*s[2]*s[2];
    c[47] = s[2]*s[2]*s[3];
    c[48] = s[2]*s[2]*s[4];
    c[49] = s[2]*s[3]*s[3];
    c[50] = s[2]*s[3]*s[4];
    c[51] = s[2]*s[4]*s[4];
    c[52] = s[3]*s[3]*s[3];
    c[53] = s[3]*s[3]*s[4];
    c[54] = s[3]*s[4]*s[4];
    c[55] = s[4]*s[4]*s[4];
    c[56] = s[0]*s[0]*s[0]*s[0];
    c[57] = s[0]*s[0]*s[0]*s[1];
    c[58] = s[0]*s[0]*s[0]*s[2];
    c[59] = s[0]*s[0]*s[0]*s[3];
    c[60] = s[0]*s[0]*s[0]*s[4];
    c[61] = s[0]*s[0]*s[1]*s[1];
    c[62] = s[0]*s[0]*s[1]*s[2];
    c[63] = s[0]*s[0]*s[1]*s[3];
    c[64] = s[0]*s[0]*s[1]*s[4];
    c[65] = s[0]*s[0]*s[2]*s[2];
    c[66] = s[0]*s[0]*s[2]*s[3];
    c[67] = s[0]*s[0]*s[2]*s[4];
    c[68] = s[0]*s[0]*s[3]*s[3];
    c[69] = s[0]*s[0]*s[3]*s[4];
    c[70] = s[0]*s[0]*s[4]*s[4];
    c[71] = s[0]*s[1]*s[1]*s[1];
    c[72] = s[0]*s[1]*s[1]*s[2];
    c[73] = s[0]*s[1]*s[1]*s[3];
    c[74] = s[0]*s[1]*s[1]*s[4];
    c[75] = s[0]*s[1]*s[2]*s[2];
    c[76] = s[0]*s[1]*s[2]*s[3];
    c[77] = s[0]*s[1]*s[2]*s[4];
    c[78] = s[0]*s[1]*s[3]*s[3];
    c[79] = s[0]*s[1]*s[3]*s[4];
    c[80] = s[0]*s[1]*s[4]*s[4];
    c[81] = s[0]*s[2]*s[2]*s[2];
    c[82] = s[0]*s[2]*s[2]*s[3];
    c[83] = s[0]*s[2]*s[2]*s[4];
    c[84] = s[0]*s[2]*s[3]*s[3];
    c[85] = s[0]*s[2]*s[3]*s[4];
    c[86] = s[0]*s[2]*s[4]*s[4];
    c[87] = s[0]*s[3]*s[3]*s[3];
    c[88] = s[0]*s[3]*s[3]*s[4];
    c[89] = s[0]*s[3]*s[4]*s[4];
    c[90] = s[0]*s[4]*s[4]*s[4];
    c[91] = s[1]*s[1]*s[1]*s[1];
    c[92] = s[1]*s[1]*s[1]*s[2];
    c[93] = s[1]*s[1]*s[1]*s[3];
    c[94] = s[1]*s[1]*s[1]*s[4];
    c[95] = s[1]*s[1]*s[2]*s[2];
    c[96] = s[1]*s[1]*s[2]*s[3];
    c[97] = s[1]*s[1]*s[2]*s[4];
    c[98] = s[1]*s[1]*s[3]*s[3];
    c[99] = s[1]*s[1]*s[3]*s[4];
    c[100] = s[1]*s[1]*s[4]*s[4];
    c[101] = s[1]*s[2]*s[2]*s[2];
    c[102] = s[1]*s[2]*s[2]*s[3];
    c[103] = s[1]*s[2]*s[2]*s[4];
    c[104] = s[1]*s[2]*s[3]*s[3];
    c[105] = s[1]*s[2]*s[3]*s[4];
    c[106] = s[1]*s[2]*s[4]*s[4];
    c[107] = s[1]*s[3]*s[3]*s[3];
    c[108] = s[1]*s[3]*s[3]*s[4];
    c[109] = s[1]*s[3]*s[4]*s[4];
    c[110] = s[1]*s[4]*s[4]*s[4];
    c[111] = s[2]*s[2]*s[2]*s[2];
    c[112] = s[2]*s[2]*s[2]*s[3];
    c[113] = s[2]*s[2]*s[2]*s[4];
    c[114] = s[2]*s[2]*s[3]*s[3];
    c[115] = s[2]*s[2]*s[3]*s[4];
    c[116] = s[2]*s[2]*s[4]*s[4];
    c[117] = s[2]*s[3]*s[3]*s[3];
    c[118] = s[2]*s[3]*s[3]*s[4];
    c[119] = s[2]*s[3]*s[4]*s[4];
    c[120] = s[2]*s[4]*s[4]*s[4];
    c[121] = s[3]*s[3]*s[3]*s[3];
    c[122] = s[3]*s[3]*s[3]*s[4];
    c[123] = s[3]*s[3]*s[4]*s[4];
    c[124] = s[3]*s[4]*s[4]*s[4];
    c[125] = s[4]*s[4]*s[4]*s[4];
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