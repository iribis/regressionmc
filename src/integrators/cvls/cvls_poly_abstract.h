#ifndef PBRT_V3_CVLS_POLY_ABSTRACT_H
#define PBRT_V3_CVLS_POLY_ABSTRACT_H

#include "pbrt.h"
#include "scene.h"

#include "ext/eigen/Eigen/Dense"

namespace pbrt {

struct Poly_luminance {
    virtual void reset() = 0;
    virtual void sgd(const Eigen::VectorXd &s, Float v, const Float lr) = 0;
    virtual float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) = 0;
    virtual void update( const Eigen::VectorXd &s, const Spectrum &fx) = 0;
    virtual void solve() = 0;
    virtual float G() const = 0;
    virtual float Gpix(Float ax, Float ay, Float bx, Float by) const { assert(false); return 0.0; };
    virtual float g( const Eigen::VectorXd &s) const = 0;
    virtual void g_then_sgd_grad(const Eigen::VectorXd &s, Float v) { assert(false); };
    virtual void apply_grad(const Float lr) { assert(false); };
};

struct Poly_luminance_SGD {
    virtual void reset() = 0;
    virtual void sgd(const Eigen::VectorXd &s, Float v, const Float lr ) = 0;
    virtual float G() const = 0;
    virtual float g( const Eigen::VectorXd &s) const = 0;
    virtual float g_then_sgd(const Eigen::VectorXd &s, Float v, const Float lr) = 0;
};

struct Poly_RGB_SGD {
    virtual void reset() = 0;
    virtual void sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr ) = 0;
    virtual Spectrum G() const = 0;
    virtual Spectrum g( const Eigen::VectorXd &s) const = 0;
    virtual Spectrum g_then_sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr) = 0;
    virtual Spectrum g_then_sgd(const Eigen::VectorXd &s, const Spectrum& v, const Float lr, int nStep) = 0;
};


}

#endif //PBRT_V3_CVLS_POLY_ABSTRACT_H
