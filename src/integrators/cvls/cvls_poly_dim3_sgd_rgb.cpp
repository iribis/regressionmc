#include "cvls_poly_dim3_sgd_rgb.h"

namespace pbrt {

void Poly_Order0_Dim3_rgb_SGD::reset() {
    X.setZero();
}
Spectrum Poly_Order0_Dim3_rgb_SGD::G() const {
    float v[3] = {X(0, 0), X(0, 1), X(0, 2)};
    int div[1];
    div[0] = 1;
    for (int i = 1; i < 1; ++i) {
        v[0] += X(i, 0) / div[i];
        v[1] += X(i, 1) / div[i];
        v[2] += X(i, 2) / div[i];
    }
    Float rgb[3] = {v[0], v[1], v[2]};
    return Spectrum::FromRGB(rgb);
}
Spectrum Poly_Order0_Dim3_rgb_SGD::g( const Eigen::VectorXd &s) const {
    float v[3] = {X(0, 0), X(0, 1), X(0, 2)};
    float c[1];
    c[0] = 1.0;
    for (int i = 1; i < 1; ++i) {
        v[0] += X(i, 0) * c[i];
        v[1] += X(i, 1) * c[i];
        v[2] += X(i, 2) * c[i];
    }
    Float rgb[3] = {v[0], v[1], v[2]};
    return Spectrum::FromRGB(rgb);
}
void Poly_Order0_Dim3_rgb_SGD::sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) {
    float c[1];
    c[0] = 1.0;
    Spectrum e = (g(s) - v);
    Float rgb[3] = {0,0,0};
    e.ToRGB(rgb);
    for (int i = 0; i < 1; ++i) {
        X(i, 0) -= lr * 2 * c[i] * rgb[0];
        X(i, 1) -= lr * 2 * c[i] * rgb[1];
        X(i, 2) -= lr * 2 * c[i] * rgb[2];
    }
}
Spectrum Poly_Order0_Dim3_rgb_SGD::g_then_sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) {
    Float v_rgb[3] = {0,0,0};
    v.ToRGB(v_rgb);
    float c[1];
    c[0] = 1.0;
    // Evaluation g
    float g[3] = {X(0, 0), X(0, 1), X(0, 2)};
    for (int i = 1; i < 1; ++i) {
        g[0] += X(i, 0) * c[i];
        g[1] += X(i, 1) * c[i];
        g[2] += X(i, 2) * c[i];
    }
    // Do one step SGD
    float e[4] = {g[0] - v_rgb[0], g[1] - v_rgb[1], g[2] - v_rgb[2]};
    for (int i = 0; i < 1; ++i) {
        X(i, 0) -= lr * 2 * c[i] * e[0];
        X(i, 1) -= lr * 2 * c[i] * e[1];
        X(i, 2) -= lr * 2 * c[i] * e[2];
    }
    Float rgb[3] = {g[0], g[1], g[2]};
    return Spectrum::FromRGB(rgb);
}
Spectrum Poly_Order0_Dim3_rgb_SGD::g_then_sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr, int nSteps) {
    Float v_rgb[3] = {0,0,0};
    v.ToRGB(v_rgb);
    float c[1];
    c[0] = 1.0;
    float g_first[3];
    float g[3];
    for(int n = 0; n < nSteps; n++) {
        // Evaluation g
        g[0] = X(0, 0);
        g[1] = X(0, 1);
        g[2] = X(0, 2);
        for (int i = 1; i < 1; ++i) {
            g[0] += X(i, 0) * c[i];
            g[1] += X(i, 1) * c[i];
            g[2] += X(i, 2) * c[i];
        }
        if(n == 0) {
            g_first[0] = g[0];
            g_first[1] = g[1];
            g_first[2] = g[2];
        }
        // Do one step SGD
        float e[4] = {g[0] - v_rgb[0], g[1] - v_rgb[1], g[2] - v_rgb[2]};
        for (int i = 0; i < 1; ++i) {
            X(i, 0) -= lr * 2 * c[i] * e[0];
            X(i, 1) -= lr * 2 * c[i] * e[1];
            X(i, 2) -= lr * 2 * c[i] * e[2];
        }
    }
    Float rgb[3] = {g_first[0], g_first[1], g_first[2]};
    return Spectrum::FromRGB(rgb);
}
void Poly_Order1_Dim3_rgb_SGD::reset() {
    X.setZero();
}
Spectrum Poly_Order1_Dim3_rgb_SGD::G() const {
    float v[3] = {X(0, 0), X(0, 1), X(0, 2)};
    int div[4];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    for (int i = 1; i < 4; ++i) {
        v[0] += X(i, 0) / div[i];
        v[1] += X(i, 1) / div[i];
        v[2] += X(i, 2) / div[i];
    }
    Float rgb[3] = {v[0], v[1], v[2]};
    return Spectrum::FromRGB(rgb);
}
Spectrum Poly_Order1_Dim3_rgb_SGD::g( const Eigen::VectorXd &s) const {
    float v[3] = {X(0, 0), X(0, 1), X(0, 2)};
    float c[4];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    for (int i = 1; i < 4; ++i) {
        v[0] += X(i, 0) * c[i];
        v[1] += X(i, 1) * c[i];
        v[2] += X(i, 2) * c[i];
    }
    Float rgb[3] = {v[0], v[1], v[2]};
    return Spectrum::FromRGB(rgb);
}
void Poly_Order1_Dim3_rgb_SGD::sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) {
    float c[4];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    Spectrum e = (g(s) - v);
    Float rgb[3] = {0,0,0};
    e.ToRGB(rgb);
    for (int i = 0; i < 4; ++i) {
        X(i, 0) -= lr * 2 * c[i] * rgb[0];
        X(i, 1) -= lr * 2 * c[i] * rgb[1];
        X(i, 2) -= lr * 2 * c[i] * rgb[2];
    }
}
Spectrum Poly_Order1_Dim3_rgb_SGD::g_then_sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) {
    Float v_rgb[3] = {0,0,0};
    v.ToRGB(v_rgb);
    float c[4];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    // Evaluation g
    float g[3] = {X(0, 0), X(0, 1), X(0, 2)};
    for (int i = 1; i < 4; ++i) {
        g[0] += X(i, 0) * c[i];
        g[1] += X(i, 1) * c[i];
        g[2] += X(i, 2) * c[i];
    }
    // Do one step SGD
    float e[4] = {g[0] - v_rgb[0], g[1] - v_rgb[1], g[2] - v_rgb[2]};
    for (int i = 0; i < 4; ++i) {
        X(i, 0) -= lr * 2 * c[i] * e[0];
        X(i, 1) -= lr * 2 * c[i] * e[1];
        X(i, 2) -= lr * 2 * c[i] * e[2];
    }
    Float rgb[3] = {g[0], g[1], g[2]};
    return Spectrum::FromRGB(rgb);
}
Spectrum Poly_Order1_Dim3_rgb_SGD::g_then_sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr, int nSteps) {
    Float v_rgb[3] = {0,0,0};
    v.ToRGB(v_rgb);
    float c[4];
    c[0] = 1.0;
    c[1] = s[0];
    c[2] = s[1];
    c[3] = s[2];
    float g_first[3];
    float g[3];
    for(int n = 0; n < nSteps; n++) {
        // Evaluation g
        g[0] = X(0, 0);
        g[1] = X(0, 1);
        g[2] = X(0, 2);
        for (int i = 1; i < 4; ++i) {
            g[0] += X(i, 0) * c[i];
            g[1] += X(i, 1) * c[i];
            g[2] += X(i, 2) * c[i];
        }
        if(n == 0) {
            g_first[0] = g[0];
            g_first[1] = g[1];
            g_first[2] = g[2];
        }
        // Do one step SGD
        float e[4] = {g[0] - v_rgb[0], g[1] - v_rgb[1], g[2] - v_rgb[2]};
        for (int i = 0; i < 4; ++i) {
            X(i, 0) -= lr * 2 * c[i] * e[0];
            X(i, 1) -= lr * 2 * c[i] * e[1];
            X(i, 2) -= lr * 2 * c[i] * e[2];
        }
    }
    Float rgb[3] = {g_first[0], g_first[1], g_first[2]};
    return Spectrum::FromRGB(rgb);
}
void Poly_Order2_Dim3_rgb_SGD::reset() {
    X.setZero();
}
Spectrum Poly_Order2_Dim3_rgb_SGD::G() const {
    float v[3] = {X(0, 0), X(0, 1), X(0, 2)};
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
        v[0] += X(i, 0) / div[i];
        v[1] += X(i, 1) / div[i];
        v[2] += X(i, 2) / div[i];
    }
    Float rgb[3] = {v[0], v[1], v[2]};
    return Spectrum::FromRGB(rgb);
}
Spectrum Poly_Order2_Dim3_rgb_SGD::g( const Eigen::VectorXd &s) const {
    float v[3] = {X(0, 0), X(0, 1), X(0, 2)};
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
        v[0] += X(i, 0) * c[i];
        v[1] += X(i, 1) * c[i];
        v[2] += X(i, 2) * c[i];
    }
    Float rgb[3] = {v[0], v[1], v[2]};
    return Spectrum::FromRGB(rgb);
}
void Poly_Order2_Dim3_rgb_SGD::sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) {
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
    Spectrum e = (g(s) - v);
    Float rgb[3] = {0,0,0};
    e.ToRGB(rgb);
    for (int i = 0; i < 10; ++i) {
        X(i, 0) -= lr * 2 * c[i] * rgb[0];
        X(i, 1) -= lr * 2 * c[i] * rgb[1];
        X(i, 2) -= lr * 2 * c[i] * rgb[2];
    }
}
Spectrum Poly_Order2_Dim3_rgb_SGD::g_then_sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) {
    Float v_rgb[3] = {0,0,0};
    v.ToRGB(v_rgb);
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
    // Evaluation g
    float g[3] = {X(0, 0), X(0, 1), X(0, 2)};
    for (int i = 1; i < 10; ++i) {
        g[0] += X(i, 0) * c[i];
        g[1] += X(i, 1) * c[i];
        g[2] += X(i, 2) * c[i];
    }
    // Do one step SGD
    float e[4] = {g[0] - v_rgb[0], g[1] - v_rgb[1], g[2] - v_rgb[2]};
    for (int i = 0; i < 10; ++i) {
        X(i, 0) -= lr * 2 * c[i] * e[0];
        X(i, 1) -= lr * 2 * c[i] * e[1];
        X(i, 2) -= lr * 2 * c[i] * e[2];
    }
    Float rgb[3] = {g[0], g[1], g[2]};
    return Spectrum::FromRGB(rgb);
}
Spectrum Poly_Order2_Dim3_rgb_SGD::g_then_sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr, int nSteps) {
    Float v_rgb[3] = {0,0,0};
    v.ToRGB(v_rgb);
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
    float g_first[3];
    float g[3];
    for(int n = 0; n < nSteps; n++) {
        // Evaluation g
        g[0] = X(0, 0);
        g[1] = X(0, 1);
        g[2] = X(0, 2);
        for (int i = 1; i < 10; ++i) {
            g[0] += X(i, 0) * c[i];
            g[1] += X(i, 1) * c[i];
            g[2] += X(i, 2) * c[i];
        }
        if(n == 0) {
            g_first[0] = g[0];
            g_first[1] = g[1];
            g_first[2] = g[2];
        }
        // Do one step SGD
        float e[4] = {g[0] - v_rgb[0], g[1] - v_rgb[1], g[2] - v_rgb[2]};
        for (int i = 0; i < 10; ++i) {
            X(i, 0) -= lr * 2 * c[i] * e[0];
            X(i, 1) -= lr * 2 * c[i] * e[1];
            X(i, 2) -= lr * 2 * c[i] * e[2];
        }
    }
    Float rgb[3] = {g_first[0], g_first[1], g_first[2]};
    return Spectrum::FromRGB(rgb);
}
void Poly_Order3_Dim3_rgb_SGD::reset() {
    X.setZero();
}
Spectrum Poly_Order3_Dim3_rgb_SGD::G() const {
    float v[3] = {X(0, 0), X(0, 1), X(0, 2)};
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
        v[0] += X(i, 0) / div[i];
        v[1] += X(i, 1) / div[i];
        v[2] += X(i, 2) / div[i];
    }
    Float rgb[3] = {v[0], v[1], v[2]};
    return Spectrum::FromRGB(rgb);
}
Spectrum Poly_Order3_Dim3_rgb_SGD::g( const Eigen::VectorXd &s) const {
    float v[3] = {X(0, 0), X(0, 1), X(0, 2)};
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
        v[0] += X(i, 0) * c[i];
        v[1] += X(i, 1) * c[i];
        v[2] += X(i, 2) * c[i];
    }
    Float rgb[3] = {v[0], v[1], v[2]};
    return Spectrum::FromRGB(rgb);
}
void Poly_Order3_Dim3_rgb_SGD::sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) {
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
    Spectrum e = (g(s) - v);
    Float rgb[3] = {0,0,0};
    e.ToRGB(rgb);
    for (int i = 0; i < 20; ++i) {
        X(i, 0) -= lr * 2 * c[i] * rgb[0];
        X(i, 1) -= lr * 2 * c[i] * rgb[1];
        X(i, 2) -= lr * 2 * c[i] * rgb[2];
    }
}
Spectrum Poly_Order3_Dim3_rgb_SGD::g_then_sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) {
    Float v_rgb[3] = {0,0,0};
    v.ToRGB(v_rgb);
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
    // Evaluation g
    float g[3] = {X(0, 0), X(0, 1), X(0, 2)};
    for (int i = 1; i < 20; ++i) {
        g[0] += X(i, 0) * c[i];
        g[1] += X(i, 1) * c[i];
        g[2] += X(i, 2) * c[i];
    }
    // Do one step SGD
    float e[4] = {g[0] - v_rgb[0], g[1] - v_rgb[1], g[2] - v_rgb[2]};
    for (int i = 0; i < 20; ++i) {
        X(i, 0) -= lr * 2 * c[i] * e[0];
        X(i, 1) -= lr * 2 * c[i] * e[1];
        X(i, 2) -= lr * 2 * c[i] * e[2];
    }
    Float rgb[3] = {g[0], g[1], g[2]};
    return Spectrum::FromRGB(rgb);
}
Spectrum Poly_Order3_Dim3_rgb_SGD::g_then_sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr, int nSteps) {
    Float v_rgb[3] = {0,0,0};
    v.ToRGB(v_rgb);
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
    float g_first[3];
    float g[3];
    for(int n = 0; n < nSteps; n++) {
        // Evaluation g
        g[0] = X(0, 0);
        g[1] = X(0, 1);
        g[2] = X(0, 2);
        for (int i = 1; i < 20; ++i) {
            g[0] += X(i, 0) * c[i];
            g[1] += X(i, 1) * c[i];
            g[2] += X(i, 2) * c[i];
        }
        if(n == 0) {
            g_first[0] = g[0];
            g_first[1] = g[1];
            g_first[2] = g[2];
        }
        // Do one step SGD
        float e[4] = {g[0] - v_rgb[0], g[1] - v_rgb[1], g[2] - v_rgb[2]};
        for (int i = 0; i < 20; ++i) {
            X(i, 0) -= lr * 2 * c[i] * e[0];
            X(i, 1) -= lr * 2 * c[i] * e[1];
            X(i, 2) -= lr * 2 * c[i] * e[2];
        }
    }
    Float rgb[3] = {g_first[0], g_first[1], g_first[2]};
    return Spectrum::FromRGB(rgb);
}
void Poly_Order4_Dim3_rgb_SGD::reset() {
    X.setZero();
}
Spectrum Poly_Order4_Dim3_rgb_SGD::G() const {
    float v[3] = {X(0, 0), X(0, 1), X(0, 2)};
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
        v[0] += X(i, 0) / div[i];
        v[1] += X(i, 1) / div[i];
        v[2] += X(i, 2) / div[i];
    }
    Float rgb[3] = {v[0], v[1], v[2]};
    return Spectrum::FromRGB(rgb);
}
Spectrum Poly_Order4_Dim3_rgb_SGD::g( const Eigen::VectorXd &s) const {
    float v[3] = {X(0, 0), X(0, 1), X(0, 2)};
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
        v[0] += X(i, 0) * c[i];
        v[1] += X(i, 1) * c[i];
        v[2] += X(i, 2) * c[i];
    }
    Float rgb[3] = {v[0], v[1], v[2]};
    return Spectrum::FromRGB(rgb);
}
void Poly_Order4_Dim3_rgb_SGD::sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) {
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
    Spectrum e = (g(s) - v);
    Float rgb[3] = {0,0,0};
    e.ToRGB(rgb);
    for (int i = 0; i < 35; ++i) {
        X(i, 0) -= lr * 2 * c[i] * rgb[0];
        X(i, 1) -= lr * 2 * c[i] * rgb[1];
        X(i, 2) -= lr * 2 * c[i] * rgb[2];
    }
}
Spectrum Poly_Order4_Dim3_rgb_SGD::g_then_sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) {
    Float v_rgb[3] = {0,0,0};
    v.ToRGB(v_rgb);
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
    // Evaluation g
    float g[3] = {X(0, 0), X(0, 1), X(0, 2)};
    for (int i = 1; i < 35; ++i) {
        g[0] += X(i, 0) * c[i];
        g[1] += X(i, 1) * c[i];
        g[2] += X(i, 2) * c[i];
    }
    // Do one step SGD
    float e[4] = {g[0] - v_rgb[0], g[1] - v_rgb[1], g[2] - v_rgb[2]};
    for (int i = 0; i < 35; ++i) {
        X(i, 0) -= lr * 2 * c[i] * e[0];
        X(i, 1) -= lr * 2 * c[i] * e[1];
        X(i, 2) -= lr * 2 * c[i] * e[2];
    }
    Float rgb[3] = {g[0], g[1], g[2]};
    return Spectrum::FromRGB(rgb);
}
Spectrum Poly_Order4_Dim3_rgb_SGD::g_then_sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr, int nSteps) {
    Float v_rgb[3] = {0,0,0};
    v.ToRGB(v_rgb);
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
    float g_first[3];
    float g[3];
    for(int n = 0; n < nSteps; n++) {
        // Evaluation g
        g[0] = X(0, 0);
        g[1] = X(0, 1);
        g[2] = X(0, 2);
        for (int i = 1; i < 35; ++i) {
            g[0] += X(i, 0) * c[i];
            g[1] += X(i, 1) * c[i];
            g[2] += X(i, 2) * c[i];
        }
        if(n == 0) {
            g_first[0] = g[0];
            g_first[1] = g[1];
            g_first[2] = g[2];
        }
        // Do one step SGD
        float e[4] = {g[0] - v_rgb[0], g[1] - v_rgb[1], g[2] - v_rgb[2]};
        for (int i = 0; i < 35; ++i) {
            X(i, 0) -= lr * 2 * c[i] * e[0];
            X(i, 1) -= lr * 2 * c[i] * e[1];
            X(i, 2) -= lr * 2 * c[i] * e[2];
        }
    }
    Float rgb[3] = {g_first[0], g_first[1], g_first[2]};
    return Spectrum::FromRGB(rgb);
}
void Poly_Order5_Dim3_rgb_SGD::reset() {
    X.setZero();
}
Spectrum Poly_Order5_Dim3_rgb_SGD::G() const {
    float v[3] = {X(0, 0), X(0, 1), X(0, 2)};
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
        v[0] += X(i, 0) / div[i];
        v[1] += X(i, 1) / div[i];
        v[2] += X(i, 2) / div[i];
    }
    Float rgb[3] = {v[0], v[1], v[2]};
    return Spectrum::FromRGB(rgb);
}
Spectrum Poly_Order5_Dim3_rgb_SGD::g( const Eigen::VectorXd &s) const {
    float v[3] = {X(0, 0), X(0, 1), X(0, 2)};
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
        v[0] += X(i, 0) * c[i];
        v[1] += X(i, 1) * c[i];
        v[2] += X(i, 2) * c[i];
    }
    Float rgb[3] = {v[0], v[1], v[2]};
    return Spectrum::FromRGB(rgb);
}
void Poly_Order5_Dim3_rgb_SGD::sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) {
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
    Spectrum e = (g(s) - v);
    Float rgb[3] = {0,0,0};
    e.ToRGB(rgb);
    for (int i = 0; i < 56; ++i) {
        X(i, 0) -= lr * 2 * c[i] * rgb[0];
        X(i, 1) -= lr * 2 * c[i] * rgb[1];
        X(i, 2) -= lr * 2 * c[i] * rgb[2];
    }
}
Spectrum Poly_Order5_Dim3_rgb_SGD::g_then_sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) {
    Float v_rgb[3] = {0,0,0};
    v.ToRGB(v_rgb);
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
    // Evaluation g
    float g[3] = {X(0, 0), X(0, 1), X(0, 2)};
    for (int i = 1; i < 56; ++i) {
        g[0] += X(i, 0) * c[i];
        g[1] += X(i, 1) * c[i];
        g[2] += X(i, 2) * c[i];
    }
    // Do one step SGD
    float e[4] = {g[0] - v_rgb[0], g[1] - v_rgb[1], g[2] - v_rgb[2]};
    for (int i = 0; i < 56; ++i) {
        X(i, 0) -= lr * 2 * c[i] * e[0];
        X(i, 1) -= lr * 2 * c[i] * e[1];
        X(i, 2) -= lr * 2 * c[i] * e[2];
    }
    Float rgb[3] = {g[0], g[1], g[2]};
    return Spectrum::FromRGB(rgb);
}
Spectrum Poly_Order5_Dim3_rgb_SGD::g_then_sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr, int nSteps) {
    Float v_rgb[3] = {0,0,0};
    v.ToRGB(v_rgb);
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
    float g_first[3];
    float g[3];
    for(int n = 0; n < nSteps; n++) {
        // Evaluation g
        g[0] = X(0, 0);
        g[1] = X(0, 1);
        g[2] = X(0, 2);
        for (int i = 1; i < 56; ++i) {
            g[0] += X(i, 0) * c[i];
            g[1] += X(i, 1) * c[i];
            g[2] += X(i, 2) * c[i];
        }
        if(n == 0) {
            g_first[0] = g[0];
            g_first[1] = g[1];
            g_first[2] = g[2];
        }
        // Do one step SGD
        float e[4] = {g[0] - v_rgb[0], g[1] - v_rgb[1], g[2] - v_rgb[2]};
        for (int i = 0; i < 56; ++i) {
            X(i, 0) -= lr * 2 * c[i] * e[0];
            X(i, 1) -= lr * 2 * c[i] * e[1];
            X(i, 2) -= lr * 2 * c[i] * e[2];
        }
    }
    Float rgb[3] = {g_first[0], g_first[1], g_first[2]};
    return Spectrum::FromRGB(rgb);
}



}