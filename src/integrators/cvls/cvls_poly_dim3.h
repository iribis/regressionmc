#ifndef PBRT_V3_CVLS_POLY_DIM3_H
#define PBRT_V3_CVLS_POLY_DIM3_H

#include "pbrt.h"
#include "scene.h"
#include "cvls_poly_abstract.h"

#include "ext/eigen/Eigen/Dense"

namespace pbrt {

struct Poly_Order0_Dim3_luminance : Poly_luminance {
    Eigen::Matrix<float, 1, 1> A;
    Eigen::Matrix<float, 1, 1> b;
    Eigen::Matrix<float, 1, 1> grads;
    Eigen::Matrix<float, 1, 1> X; // Solution
    Poly_Order0_Dim3_luminance():
            A(Eigen::Matrix<float, 1, 1>::Zero()),
            b(Eigen::Matrix<float, 1, 1>::Zero()),
            grads(Eigen::Matrix<float, 1, 1>::Zero()),
            X(Eigen::Matrix<float, 1, 1>::Zero())
    {}
    void reset() override;
    void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void g_then_sgd_grad(const Eigen::VectorXd &s, Float v) override;
    void apply_grad(const Float lr) override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order1_Dim3_luminance : Poly_luminance {
    Eigen::Matrix<float, 4, 4> A;
    Eigen::Matrix<float, 4, 1> b;
    Eigen::Matrix<float, 4, 1> grads;
    Eigen::Matrix<float, 4, 1> X; // Solution
    Poly_Order1_Dim3_luminance():
            A(Eigen::Matrix<float, 4, 4>::Zero()),
            b(Eigen::Matrix<float, 4, 1>::Zero()),
            grads(Eigen::Matrix<float, 4, 1>::Zero()),
            X(Eigen::Matrix<float, 4, 1>::Zero())
    {}
    void reset() override;
    void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void g_then_sgd_grad(const Eigen::VectorXd &s, Float v) override;
    void apply_grad(const Float lr) override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order2_Dim3_luminance : Poly_luminance {
    Eigen::Matrix<float, 10, 10> A;
    Eigen::Matrix<float, 10, 1> b;
    Eigen::Matrix<float, 10, 1> grads;
    Eigen::Matrix<float, 10, 1> X; // Solution
    Poly_Order2_Dim3_luminance():
            A(Eigen::Matrix<float, 10, 10>::Zero()),
            b(Eigen::Matrix<float, 10, 1>::Zero()),
            grads(Eigen::Matrix<float, 10, 1>::Zero()),
            X(Eigen::Matrix<float, 10, 1>::Zero())
    {}
    void reset() override;
    void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void g_then_sgd_grad(const Eigen::VectorXd &s, Float v) override;
    void apply_grad(const Float lr) override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order3_Dim3_luminance : Poly_luminance {
    Eigen::Matrix<float, 20, 20> A;
    Eigen::Matrix<float, 20, 1> b;
    Eigen::Matrix<float, 20, 1> grads;
    Eigen::Matrix<float, 20, 1> X; // Solution
    Poly_Order3_Dim3_luminance():
            A(Eigen::Matrix<float, 20, 20>::Zero()),
            b(Eigen::Matrix<float, 20, 1>::Zero()),
            grads(Eigen::Matrix<float, 20, 1>::Zero()),
            X(Eigen::Matrix<float, 20, 1>::Zero())
    {}
    void reset() override;
    void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void g_then_sgd_grad(const Eigen::VectorXd &s, Float v) override;
    void apply_grad(const Float lr) override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order4_Dim3_luminance : Poly_luminance {
    Eigen::Matrix<float, 35, 35> A;
    Eigen::Matrix<float, 35, 1> b;
    Eigen::Matrix<float, 35, 1> grads;
    Eigen::Matrix<float, 35, 1> X; // Solution
    Poly_Order4_Dim3_luminance():
            A(Eigen::Matrix<float, 35, 35>::Zero()),
            b(Eigen::Matrix<float, 35, 1>::Zero()),
            grads(Eigen::Matrix<float, 35, 1>::Zero()),
            X(Eigen::Matrix<float, 35, 1>::Zero())
    {}
    void reset() override;
    void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void g_then_sgd_grad(const Eigen::VectorXd &s, Float v) override;
    void apply_grad(const Float lr) override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order5_Dim3_luminance : Poly_luminance {
    Eigen::Matrix<float, 56, 56> A;
    Eigen::Matrix<float, 56, 1> b;
    Eigen::Matrix<float, 56, 1> grads;
    Eigen::Matrix<float, 56, 1> X; // Solution
    Poly_Order5_Dim3_luminance():
            A(Eigen::Matrix<float, 56, 56>::Zero()),
            b(Eigen::Matrix<float, 56, 1>::Zero()),
            grads(Eigen::Matrix<float, 56, 1>::Zero()),
            X(Eigen::Matrix<float, 56, 1>::Zero())
    {}
    void reset() override;
    void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void g_then_sgd_grad(const Eigen::VectorXd &s, Float v) override;
    void apply_grad(const Float lr) override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order6_Dim3_luminance : Poly_luminance {
    Eigen::Matrix<float, 84, 84> A;
    Eigen::Matrix<float, 84, 1> b;
    Eigen::Matrix<float, 84, 1> grads;
    Eigen::Matrix<float, 84, 1> X; // Solution
    Poly_Order6_Dim3_luminance():
            A(Eigen::Matrix<float, 84, 84>::Zero()),
            b(Eigen::Matrix<float, 84, 1>::Zero()),
            grads(Eigen::Matrix<float, 84, 1>::Zero()),
            X(Eigen::Matrix<float, 84, 1>::Zero())
    {}
    void reset() override;
    void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void g_then_sgd_grad(const Eigen::VectorXd &s, Float v) override;
    void apply_grad(const Float lr) override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order7_Dim3_luminance : Poly_luminance {
    Eigen::Matrix<float, 120, 120> A;
    Eigen::Matrix<float, 120, 1> b;
    Eigen::Matrix<float, 120, 1> grads;
    Eigen::Matrix<float, 120, 1> X; // Solution
    Poly_Order7_Dim3_luminance():
            A(Eigen::Matrix<float, 120, 120>::Zero()),
            b(Eigen::Matrix<float, 120, 1>::Zero()),
            grads(Eigen::Matrix<float, 120, 1>::Zero()),
            X(Eigen::Matrix<float, 120, 1>::Zero())
    {}
    void reset() override;
    void update( const Eigen::VectorXd &s, const Spectrum &fx) override;
    void solve() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void g_then_sgd_grad(const Eigen::VectorXd &s, Float v) override;
    void apply_grad(const Float lr) override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};


}

#endif //PBRT_V3_CVLS_TOOLS_H
