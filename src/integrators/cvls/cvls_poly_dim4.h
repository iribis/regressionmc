#ifndef PBRT_V3_CVLS_POLY_DIM4_H
#define PBRT_V3_CVLS_POLY_DIM4_H

#include "pbrt.h"
#include "scene.h"
#include "cvls_poly_abstract.h"

#include "ext/eigen/Eigen/Dense"

namespace pbrt {

struct Poly_Order0_Dim4_luminance : Poly_luminance {
    Eigen::Matrix<float, 1, 1> A;
    Eigen::Matrix<float, 1, 1> b;
    Eigen::Matrix<float, 1, 1> X; // Solution
    Poly_Order0_Dim4_luminance():
            A(Eigen::Matrix<float, 1, 1>::Zero()),
            b(Eigen::Matrix<float, 1, 1>::Zero()),
            X(Eigen::Matrix<float, 1, 1>::Zero())
    {}
    void reset() override;
    void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float Gpix(Float ax, Float ay, Float bx, Float by) const override;
    float g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order1_Dim4_luminance : Poly_luminance {
    Eigen::Matrix<float, 5, 5> A;
    Eigen::Matrix<float, 5, 1> b;
    Eigen::Matrix<float, 5, 1> X; // Solution
    Poly_Order1_Dim4_luminance():
            A(Eigen::Matrix<float, 5, 5>::Zero()),
            b(Eigen::Matrix<float, 5, 1>::Zero()),
            X(Eigen::Matrix<float, 5, 1>::Zero())
    {}
    void reset() override;
    void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float Gpix(Float ax, Float ay, Float bx, Float by) const override;
    float g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order2_Dim4_luminance : Poly_luminance {
    Eigen::Matrix<float, 15, 15> A;
    Eigen::Matrix<float, 15, 1> b;
    Eigen::Matrix<float, 15, 1> X; // Solution
    Poly_Order2_Dim4_luminance():
            A(Eigen::Matrix<float, 15, 15>::Zero()),
            b(Eigen::Matrix<float, 15, 1>::Zero()),
            X(Eigen::Matrix<float, 15, 1>::Zero())
    {}
    void reset() override;
    void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float Gpix(Float ax, Float ay, Float bx, Float by) const override;
    float g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order3_Dim4_luminance : Poly_luminance {
    Eigen::Matrix<float, 35, 35> A;
    Eigen::Matrix<float, 35, 1> b;
    Eigen::Matrix<float, 35, 1> X; // Solution
    Poly_Order3_Dim4_luminance():
            A(Eigen::Matrix<float, 35, 35>::Zero()),
            b(Eigen::Matrix<float, 35, 1>::Zero()),
            X(Eigen::Matrix<float, 35, 1>::Zero())
    {}
    void reset() override;
    void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float Gpix(Float ax, Float ay, Float bx, Float by) const override;
    float g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order4_Dim4_luminance : Poly_luminance {
    Eigen::Matrix<float, 70, 70> A;
    Eigen::Matrix<float, 70, 1> b;
    Eigen::Matrix<float, 70, 1> X; // Solution
    Poly_Order4_Dim4_luminance():
            A(Eigen::Matrix<float, 70, 70>::Zero()),
            b(Eigen::Matrix<float, 70, 1>::Zero()),
            X(Eigen::Matrix<float, 70, 1>::Zero())
    {}
    void reset() override;
    void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float Gpix(Float ax, Float ay, Float bx, Float by) const override;
    float g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order5_Dim4_luminance : Poly_luminance {
    Eigen::Matrix<float, 126, 126> A;
    Eigen::Matrix<float, 126, 1> b;
    Eigen::Matrix<float, 126, 1> X; // Solution
    Poly_Order5_Dim4_luminance():
            A(Eigen::Matrix<float, 126, 126>::Zero()),
            b(Eigen::Matrix<float, 126, 1>::Zero()),
            X(Eigen::Matrix<float, 126, 1>::Zero())
    {}
    void reset() override;
    void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float Gpix(Float ax, Float ay, Float bx, Float by) const override;
    float g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};


}

#endif //PBRT_V3_CVLS_TOOLS_H
