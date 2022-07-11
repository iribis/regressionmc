#ifndef PBRT_V3_CVLS_POLY_DIM3_SGD_RGB_H
#define PBRT_V3_CVLS_POLY_DIM3_SGD_RGB_H

#include "pbrt.h"
#include "scene.h"
#include "cvls_poly_abstract.h"
#include "cvls_estimators.h"

namespace pbrt {

struct Poly_Order0_Dim3_rgb_SGD : Poly_RGB_SGD {
    Eigen::Matrix<float, 1, 3> X; // Solution
    Poly_Order0_Dim3_rgb_SGD():
            X(Eigen::Matrix<float, 1, 3>::Zero())
    {}
    void reset() override;
    Spectrum G() const override;
    Spectrum g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) override;
    Spectrum g_then_sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr) override;
    Spectrum g_then_sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr, int nSteps) override;
};
struct Poly_Order1_Dim3_rgb_SGD : Poly_RGB_SGD {
    Eigen::Matrix<float, 4, 3> X; // Solution
    Poly_Order1_Dim3_rgb_SGD():
            X(Eigen::Matrix<float, 4, 3>::Zero())
    {}
    void reset() override;
    Spectrum G() const override;
    Spectrum g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) override;
    Spectrum g_then_sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr) override;
    Spectrum g_then_sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr, int nSteps) override;
};
struct Poly_Order2_Dim3_rgb_SGD : Poly_RGB_SGD {
    Eigen::Matrix<float, 10, 3> X; // Solution
    Poly_Order2_Dim3_rgb_SGD():
            X(Eigen::Matrix<float, 10, 3>::Zero())
    {}
    void reset() override;
    Spectrum G() const override;
    Spectrum g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) override;
    Spectrum g_then_sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr) override;
    Spectrum g_then_sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr, int nSteps) override;
};
struct Poly_Order3_Dim3_rgb_SGD : Poly_RGB_SGD {
    Eigen::Matrix<float, 20, 3> X; // Solution
    Poly_Order3_Dim3_rgb_SGD():
            X(Eigen::Matrix<float, 20, 3>::Zero())
    {}
    void reset() override;
    Spectrum G() const override;
    Spectrum g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) override;
    Spectrum g_then_sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr) override;
    Spectrum g_then_sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr, int nSteps) override;
};
struct Poly_Order4_Dim3_rgb_SGD : Poly_RGB_SGD {
    Eigen::Matrix<float, 35, 3> X; // Solution
    Poly_Order4_Dim3_rgb_SGD():
            X(Eigen::Matrix<float, 35, 3>::Zero())
    {}
    void reset() override;
    Spectrum G() const override;
    Spectrum g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) override;
    Spectrum g_then_sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr) override;
    Spectrum g_then_sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr, int nSteps) override;
};
struct Poly_Order5_Dim3_rgb_SGD : Poly_RGB_SGD {
    Eigen::Matrix<float, 56, 3> X; // Solution
    Poly_Order5_Dim3_rgb_SGD():
            X(Eigen::Matrix<float, 56, 3>::Zero())
    {}
    void reset() override;
    Spectrum G() const override;
    Spectrum g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, const Spectrum& v, const Float lr) override;
    Spectrum g_then_sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr) override;
    Spectrum g_then_sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr, int nSteps) override;
};

inline ProgressiveEstimator constructProgressiveEstimatorDim3(int maxOrder) {
    if(maxOrder == 0) {
        return ProgressiveEstimator(std::make_shared<Poly_Order0_Dim3_rgb_SGD>());
    } else if(maxOrder == 1) {
        return ProgressiveEstimator(std::make_shared<Poly_Order1_Dim3_rgb_SGD>());
    } else if(maxOrder == 2) {
        return ProgressiveEstimator(std::make_shared<Poly_Order2_Dim3_rgb_SGD>());
    } else if(maxOrder == 3) {
        return ProgressiveEstimator(std::make_shared<Poly_Order3_Dim3_rgb_SGD>());
    } else if(maxOrder == 4) {
        return ProgressiveEstimator(std::make_shared<Poly_Order4_Dim3_rgb_SGD>());
    } else if(maxOrder == 5) {
        return ProgressiveEstimator(std::make_shared<Poly_Order5_Dim3_rgb_SGD>());
    } else {
        LOG(ERROR) << "MaxOrder > 5 is not implemented\n";
        return ProgressiveEstimator(std::make_shared<Poly_Order3_Dim3_rgb_SGD>());
    }
};

}

#endif //PBRT_V3_CVLS_TOOLS_H
