#ifndef PBRT_V3_CVLS_POLY_DIM3_SGD_H
#define PBRT_V3_CVLS_POLY_DIM3_SGD_H

#include "pbrt.h"
#include "scene.h"
#include "cvls_poly_abstract.h"
#include "cvls_estimators.h"

namespace pbrt {

struct Poly_Order0_Dim3_luminance_SGD : Poly_luminance_SGD {
    Eigen::Matrix<float, 1, 1> X; // Solution
    Poly_Order0_Dim3_luminance_SGD():
            X(Eigen::Matrix<float, 1, 1>::Zero())
    {}
    void reset() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order1_Dim3_luminance_SGD : Poly_luminance_SGD {
    Eigen::Matrix<float, 4, 1> X; // Solution
    Poly_Order1_Dim3_luminance_SGD():
            X(Eigen::Matrix<float, 4, 1>::Zero())
    {}
    void reset() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order2_Dim3_luminance_SGD : Poly_luminance_SGD {
    Eigen::Matrix<float, 10, 1> X; // Solution
    Poly_Order2_Dim3_luminance_SGD():
            X(Eigen::Matrix<float, 10, 1>::Zero())
    {}
    void reset() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order3_Dim3_luminance_SGD : Poly_luminance_SGD {
    Eigen::Matrix<float, 20, 1> X; // Solution
    Poly_Order3_Dim3_luminance_SGD():
            X(Eigen::Matrix<float, 20, 1>::Zero())
    {}
    void reset() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order4_Dim3_luminance_SGD : Poly_luminance_SGD {
    Eigen::Matrix<float, 35, 1> X; // Solution
    Poly_Order4_Dim3_luminance_SGD():
            X(Eigen::Matrix<float, 35, 1>::Zero())
    {}
    void reset() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order5_Dim3_luminance_SGD : Poly_luminance_SGD {
    Eigen::Matrix<float, 56, 1> X; // Solution
    Poly_Order5_Dim3_luminance_SGD():
            X(Eigen::Matrix<float, 56, 1>::Zero())
    {}
    void reset() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order6_Dim3_luminance_SGD : Poly_luminance_SGD {
    Eigen::Matrix<float, 84, 1> X; // Solution
    Poly_Order6_Dim3_luminance_SGD():
            X(Eigen::Matrix<float, 84, 1>::Zero())
    {}
    void reset() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};
struct Poly_Order7_Dim3_luminance_SGD : Poly_luminance_SGD {
    Eigen::Matrix<float, 120, 1> X; // Solution
    Poly_Order7_Dim3_luminance_SGD():
            X(Eigen::Matrix<float, 120, 1>::Zero())
    {}
    void reset() override;
    float G() const override;
    float g( const Eigen::VectorXd &s) const override;
    void sgd( const Eigen::VectorXd &s, Float v, const Float lr) override;
    float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) override;
};

inline ProgressiveEstimatorRGB constructProgressiveEstimatorRGBDim3(int maxOrder) {
    if(maxOrder == 0) {
        return ProgressiveEstimatorRGB(std::make_shared<Poly_Order0_Dim3_luminance_SGD>(),
                                       std::make_shared<Poly_Order0_Dim3_luminance_SGD>(),
                                       std::make_shared<Poly_Order0_Dim3_luminance_SGD>());
    } else if(maxOrder == 1) {
        return ProgressiveEstimatorRGB(std::make_shared<Poly_Order1_Dim3_luminance_SGD>(),
                                       std::make_shared<Poly_Order1_Dim3_luminance_SGD>(),
                                       std::make_shared<Poly_Order1_Dim3_luminance_SGD>());
    } else if(maxOrder == 2) {
        return ProgressiveEstimatorRGB(std::make_shared<Poly_Order2_Dim3_luminance_SGD>(),
                                       std::make_shared<Poly_Order2_Dim3_luminance_SGD>(),
                                       std::make_shared<Poly_Order2_Dim3_luminance_SGD>());
    } else if(maxOrder == 3) {
        return ProgressiveEstimatorRGB(std::make_shared<Poly_Order3_Dim3_luminance_SGD>(),
                                       std::make_shared<Poly_Order3_Dim3_luminance_SGD>(),
                                       std::make_shared<Poly_Order3_Dim3_luminance_SGD>());
    } else if(maxOrder == 4) {
        return ProgressiveEstimatorRGB(std::make_shared<Poly_Order4_Dim3_luminance_SGD>(),
                                       std::make_shared<Poly_Order4_Dim3_luminance_SGD>(),
                                       std::make_shared<Poly_Order4_Dim3_luminance_SGD>());
    } else if(maxOrder == 5) {
        return ProgressiveEstimatorRGB(std::make_shared<Poly_Order5_Dim3_luminance_SGD>(),
                                       std::make_shared<Poly_Order5_Dim3_luminance_SGD>(),
                                       std::make_shared<Poly_Order5_Dim3_luminance_SGD>());
    } else {
        LOG(ERROR) << "MaxOrder > 5 is not implemented\n";
        return ProgressiveEstimatorRGB(std::make_shared<Poly_Order3_Dim3_luminance_SGD>(),
                                       std::make_shared<Poly_Order3_Dim3_luminance_SGD>(),
                                       std::make_shared<Poly_Order3_Dim3_luminance_SGD>());
    }
};

}

#endif //PBRT_V3_CVLS_TOOLS_H
