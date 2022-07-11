#include "pbrt.h"
#include "scene.h"
#include "cvls_poly_abstract.h"

#include <memory>

#ifndef PBRT_V3_CVLS_ESTIMATORS_H
#define PBRT_V3_CVLS_ESTIMATORS_H

namespace pbrt {

// Adapted from practical path guiding
class AdamOptimizer {
public:
    AdamOptimizer(Float learningRate, Float epsilon = 1e-08f, Float beta1 = 0.9f, Float beta2 = 0.999f) {
        m_hparams = { learningRate, epsilon, beta1, beta2 };
    }

    AdamOptimizer& operator=(const AdamOptimizer& arg) {
        m_state = arg.m_state;
        m_hparams = arg.m_hparams;
        return *this;
    }

    AdamOptimizer(const AdamOptimizer& arg) {
        *this = arg;
    }

    void step(Float gradient) {
        ++m_state.iter;

        Float actualLearningRate = m_hparams.learningRate * std::sqrt(1 - std::pow(m_hparams.beta2, m_state.iter)) / (1 - std::pow(m_hparams.beta1, m_state.iter));
        m_state.firstMoment = m_hparams.beta1 * m_state.firstMoment + (1 - m_hparams.beta1) * gradient;
        m_state.secondMoment = m_hparams.beta2 * m_state.secondMoment + (1 - m_hparams.beta2) * gradient * gradient;
        m_state.variable -= actualLearningRate * m_state.firstMoment / (std::sqrt(m_state.secondMoment) + m_hparams.epsilon);
    }

    Float variable() const {
        return m_state.variable;
    }

private:
    struct State {
        int iter = 0;
        Float firstMoment = 0;
        Float secondMoment = 0;
        Float variable = 0;
    } m_state;

    struct Hyperparameters {
        Float learningRate;
        Float epsilon;
        Float beta1;
        Float beta2;
    } m_hparams;
};

// Compute the CVLS with all the samples
struct DirectEstimator {
    Spectrum accum = 0;
    Float weights = 0;
    std::vector<Eigen::VectorXd> samples;
    std::vector<Float> values;
    std::shared_ptr<Poly_luminance> poly;

    const int nbSGDIter = 0;
    const int batchSize = 0;
    const Float lr = 0.01;
    DirectEstimator(std::shared_ptr<Poly_luminance>&& poly, const int nbSGDIter, const int batchSize, const Float lr):
        poly(poly), nbSGDIter(nbSGDIter), batchSize(batchSize), lr(lr) {}

    void reset() {
        samples.clear();
        values.clear();
        accum = 0.0;
        weights = 0.0;
        poly->reset();
    }

    void update(const Eigen::VectorXd&& s, const Spectrum& v, Float w) {
        accum += w * v;
        weights += w;

        if(nbSGDIter == 0) {
            poly->update(s, w * v);
        }

        samples.emplace_back(s);
        values.push_back(w * v.y());
    }

    Spectrum compute_mean() const {
        if(weights == 0.0) {
            return Spectrum(0.f);
        } else {
            return accum / weights;
        }
    }

    Spectrum compute(bool use_alpha) {
        Spectrum mean_fx = compute_mean();
        if(mean_fx.IsBlack()) {
            return mean_fx;
        }

        if(nbSGDIter>0) {
            int nb_iter = 0;
            for(int i = 0; i < nbSGDIter; i++) {
                for(auto j = 0; j < samples.size(); j++) {
                    poly->g_then_sgd_grad(samples[j], values[j]);
                    nb_iter++;
                    if(nb_iter % batchSize == 0) {
                        poly->apply_grad(lr / batchSize);
                    }
                }
            }
        } else {
            poly->solve(); // Solve LS problem
        }

        // Compute alpha and mean_gx (if needed)
        auto result = computeAlpha(use_alpha);
        Spectrum color = (mean_fx / mean_fx.y());
        Float G_val = poly->G();

        // Main equation
        //return (mean_fx - color * std::get<0>(result) * std::get<1>(result));
        return color * std::get<0>(result) * G_val + (mean_fx - color * std::get<0>(result) * std::get<1>(result));
    }

    std::tuple<Float, Float> computeAlpha(bool use_alpha) const {
        if(!use_alpha) {
            Float mean_gx = 0.0;
            Float norm = 1.0 / samples.size();
            for(const auto& s: samples) {
                mean_gx += poly->g(s) * norm;
            }
            return std::make_tuple(1.0, mean_gx);
        } else {
            std::vector<Float> list_gx;
            list_gx.reserve(samples.size());

            Float mean_fgx = 0.0;
            Float mean_gx = 0.0;
            Float norm = 1.0 / samples.size();
            for(auto i = 0; i < samples.size(); i++) {
                Float gx = poly->g(samples[i]);
                mean_gx += gx * norm;
                mean_fgx += gx * values[i] * norm;
                list_gx.push_back(gx);
            }

            Float var_g = 0.0;
            for(auto gx: list_gx) {
                var_g += std::pow(gx - mean_gx, 2) * norm;
            }

            // Image space control variate alpha
            if(var_g != 0.0) {
                Float alpha = (mean_fgx - compute_mean().y() * mean_gx) / var_g;
                return std::make_tuple(alpha, mean_gx);
            } else {
                return std::make_tuple(1.0, mean_gx);
            }
        }
    }
};

// Estimator that estimate several pixel at a time
struct DirectMultipleEstimator {
    std::vector<Spectrum> accum;
    std::vector<Float> weights;
    std::vector<std::vector<Eigen::VectorXd>> samples;
    std::vector<std::vector<Float>> values;
    std::shared_ptr<Poly_luminance> poly;
    const int nbSGDIter = 0;

    DirectMultipleEstimator(std::shared_ptr<Poly_luminance>&& poly, const int nbSGDIter, int nbPixels):
            poly(poly), nbSGDIter(nbSGDIter), accum(nbPixels), weights(nbPixels), samples(nbPixels), values(nbPixels) {
    }

    void update(Eigen::VectorXd&& s, const Spectrum& v, Float w, int pixelIndex) {
        accum[pixelIndex] += w * v;
        weights[pixelIndex] += w;

        if(nbSGDIter == 0) {
            poly->update(s, w * v);
        }

        samples[pixelIndex].emplace_back(s);
        values[pixelIndex].push_back(w * v.y());
    }

    Spectrum compute_mean(int idPixel) const {
        if(weights[idPixel] == 0.0) {
            return Spectrum(0.f);
        } else {
            return accum[idPixel] / weights[idPixel];
        }
    }

    Spectrum compute_mean_all() const {
        Spectrum res(0.f);
        for(int i = 0; i < samples.size(); i++) {
            res += compute_mean(i);
        }
        return res / samples.size();
    }

    void solve() {
        if(nbSGDIter>0) {
            for (int i = 0; i < nbSGDIter; i++) {
                for(int idpixel = 0; idpixel < samples.size(); idpixel++) {
                    for (auto j = 0; j < samples[idpixel].size(); j++) {
                        poly->sgd(samples[idpixel][j], values[idpixel][j], 0.01);
                    }
                }
            }
        } else {
            poly->solve(); // Solve LS problem
        }
    }

    Spectrum compute(bool use_alpha, int idPixel, Float ax, Float ay, Float bx, Float by) {
        Spectrum mean_fx_all = compute_mean_all();
        if(mean_fx_all.IsBlack()) {
            return mean_fx_all;
        }

        // Compute alpha and mean_gx (if needed)
        auto result = computeAlpha(use_alpha, idPixel);
        Spectrum color = (mean_fx_all / mean_fx_all.y());
        Float G_val = poly->Gpix(ax, ay, bx, by);

        // Main equation
        Spectrum mean_fx = compute_mean(idPixel);
//        return G_val * color;
        return color * std::get<0>(result) * G_val + (mean_fx - color * std::get<0>(result) * std::get<1>(result));
    }

    std::tuple<Float, Float> computeAlpha(bool use_alpha, int idPixel) const {
        if(!use_alpha) {
            Float mean_gx = 0.0;
            Float norm = 1.0 / samples[idPixel].size();
            for(const auto& s: samples[idPixel]) {
                mean_gx += poly->g(s) * norm;
            }
            return std::make_tuple(1.0, mean_gx);
        } else {
            std::vector<Float> list_gx;
            list_gx.reserve(samples.size());

            Float mean_fgx = 0.0;
            Float mean_gx = 0.0;
            Float norm = 1.0 / samples[idPixel].size();
            for(auto i = 0; i < samples[idPixel].size(); i++) {
                Float gx = poly->g(samples[idPixel][i]);
                mean_gx += gx * norm;
                mean_fgx += gx * values[idPixel][i] * norm;
                list_gx.push_back(gx);
            }

            Float var_g = 0.0;
            for(auto gx: list_gx) {
                var_g += std::pow(gx - mean_gx, 2) * norm;
            }

            // Image space control variate alpha
            if(var_g != 0.0) {
                Float alpha = (mean_fgx - compute_mean(idPixel).y() * mean_gx) / var_g;
                return std::make_tuple(alpha, mean_gx);
            } else {
                return std::make_tuple(1.0, mean_gx);
            }
        }
    }
};



// Compute the CVLS on the fly
struct ProgressiveEstimator {
    std::shared_ptr<Poly_RGB_SGD> poly;

    ProgressiveEstimator(std::shared_ptr<Poly_RGB_SGD>&& poly):
            poly(poly) {}

    void reset() {
        poly->reset();
    }

    Spectrum update(Eigen::VectorXd&& s, const Spectrum& v, Float w) {
        Spectrum G_val = poly->G();
        auto g = poly->g_then_sgd(s, v * w, 0.01);
        // If multiple step is needed
//        auto g = poly->g_then_sgd(s, v * w, 0.01, 10);
        return G_val + (v - g);
    }
};

struct ProgressiveEstimatorRGB {
    // TODO: Can modify the polynomial to encode RGB directly
    std::shared_ptr<Poly_luminance_SGD> polyR;
    std::shared_ptr<Poly_luminance_SGD> polyG;
    std::shared_ptr<Poly_luminance_SGD> polyB;

    ProgressiveEstimatorRGB(std::shared_ptr<Poly_luminance_SGD>&& polyR,
                         std::shared_ptr<Poly_luminance_SGD>&& polyG,
                         std::shared_ptr<Poly_luminance_SGD>&& polyB):
            polyR(polyR), polyG(polyG), polyB(polyB) {}

    Spectrum update(Eigen::VectorXd&& s, const Spectrum& v, Float w) {
        Float rgb[3];
        v.ToRGB(rgb);

        Float R_val = polyR->G();
        Float G_val = polyG->G();
        Float B_val = polyB->G();

        Float R_g = polyR->g_then_sgd(s, w * rgb[0], 0.01);
        Float G_g = polyG->g_then_sgd(s, w * rgb[1], 0.01);
        Float B_g = polyB->g_then_sgd(s, w * rgb[2], 0.01);

        Float rgbRes[3] = { R_val + (rgb[0] - R_g),
                            G_val + (rgb[1] - G_g),
                            B_val + (rgb[2] - B_g) };

        return Spectrum::FromRGB(rgbRes);
    }

    void reset() {
        polyR->reset();
        polyG->reset();
        polyB->reset();
    }
};

}

#endif //PBRT_V3_CVLS_TOOLS_H
