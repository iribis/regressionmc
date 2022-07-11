#ifndef PBRT_V3_CVLS_POLY_DIM2_H
#define PBRT_V3_CVLS_POLY_DIM2_H

#include "cvls_poly_abstract.h"
#include "ext/eigen/Eigen/Dense"
#include "pbrt.h"
#include "scene.h"

namespace pbrt {

struct Poly_Order0_Dim2_luminance : Poly_luminance {
 Eigen::Matrix<float, 1, 1> A;
 Eigen::Matrix<float, 1, 1> b;
 Eigen::Matrix<float, 1, 1> X; // Solution
 Poly_Order0_Dim2_luminance():
    A(Eigen::Matrix<float, 1, 1>::Zero()),
    b(Eigen::Matrix<float, 1, 1>::Zero()),
    X(Eigen::Matrix<float, 1, 1>::Zero())
 {}
 void reset() override;
 void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
 void solve() override;
 float G() const override;
 float g( const Eigen::VectorXd &s) const override;
 void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
 float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order1_Dim2_luminance : Poly_luminance {
    Eigen::Matrix<float, 3, 3> A;
    Eigen::Matrix<float, 3, 1> b;
    Eigen::Matrix<float, 3, 1> X;  // Solution
    Poly_Order1_Dim2_luminance()
        : A(Eigen::Matrix<float, 3, 3>::Zero()),
          b(Eigen::Matrix<float, 3, 1>::Zero()),
          X(Eigen::Matrix<float, 3, 1>::Zero()) {}
    void reset() override;
    void update(const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float g(const Eigen::VectorXd &s) const override;
    void sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v,
                     const Float lr) override;
};
struct Poly_Order2_Dim2_luminance : Poly_luminance {
    Eigen::Matrix<float, 6, 6> A;
    Eigen::Matrix<float, 6, 1> b;
    Eigen::Matrix<float, 6, 1> X;  // Solution
    Poly_Order2_Dim2_luminance()
        : A(Eigen::Matrix<float, 6, 6>::Zero()),
          b(Eigen::Matrix<float, 6, 1>::Zero()),
          X(Eigen::Matrix<float, 6, 1>::Zero()) {}
    void reset() override;
    void update(const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float g(const Eigen::VectorXd &s) const override;
    void sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v,
                     const Float lr) override;
};
struct Poly_Order3_Dim2_luminance : Poly_luminance {
    Eigen::Matrix<float, 10, 10> A;
    Eigen::Matrix<float, 10, 1> b;
    Eigen::Matrix<float, 10, 1> X;  // Solution
    Poly_Order3_Dim2_luminance()
        : A(Eigen::Matrix<float, 10, 10>::Zero()),
          b(Eigen::Matrix<float, 10, 1>::Zero()),
          X(Eigen::Matrix<float, 10, 1>::Zero()) {}
    void reset() override;
    void update(const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float g(const Eigen::VectorXd &s) const override;
    void sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v,
                     const Float lr) override;
};
struct Poly_Order4_Dim2_luminance : Poly_luminance {
    Eigen::Matrix<float, 15, 15> A;
    Eigen::Matrix<float, 15, 1> b;
    Eigen::Matrix<float, 15, 1> X;  // Solution
    Poly_Order4_Dim2_luminance()
        : A(Eigen::Matrix<float, 15, 15>::Zero()),
          b(Eigen::Matrix<float, 15, 1>::Zero()),
          X(Eigen::Matrix<float, 15, 1>::Zero()) {}
    void reset() override;
    void update(const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float g(const Eigen::VectorXd &s) const override;
    void sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v,
                     const Float lr) override;
};
struct Poly_Order5_Dim2_luminance : Poly_luminance {
    Eigen::Matrix<float, 21, 21> A;
    Eigen::Matrix<float, 21, 1> b;
    Eigen::Matrix<float, 21, 1> X;  // Solution
    Poly_Order5_Dim2_luminance()
        : A(Eigen::Matrix<float, 21, 21>::Zero()),
          b(Eigen::Matrix<float, 21, 1>::Zero()),
          X(Eigen::Matrix<float, 21, 1>::Zero()) {}
    void reset() override;
    void update(const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float g(const Eigen::VectorXd &s) const override;
    void sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v,
                     const Float lr) override;
};
}  // namespace pbrt

#endif  // PBRT_V3_CVLS_TOOLS_H
