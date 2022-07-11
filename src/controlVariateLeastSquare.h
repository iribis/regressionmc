/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_CVLS_H
#define PBRT_INTEGRATORS_CVLS_H

// integrators/directlighting.h*
#include "integrator.h"
#include "pbrt.h"
#include "scene.h"
#include "ext/eigen/Eigen/Dense"
#include <fstream>
#include <algorithm>    // std::sort

namespace pbrt {

// LightStrategyCV Declarations
enum class LightStrategyCV { UniformSampleAll, UniformSampleOne };

// ControleVariateLSDirectLightingIntegrator Declarations
class ControleVariateLSDirectLightingIntegrator : public SamplerIntegrator {
  public:
    // ControleVariateLSDirectLightingIntegrator Public Methods
    ControleVariateLSDirectLightingIntegrator(LightStrategyCV strategy, int maxDepth,
                             std::shared_ptr<const Camera> camera,
                             std::shared_ptr<Sampler> sampler,
                             const Bounds2i &pixelBounds,
                             int maxOrder,int dimension)
        : SamplerIntegrator(camera, sampler, pixelBounds),
          strategy(strategy),
          maxDepth(maxDepth),
          maxOrder(maxOrder),
          dimension(dimension){
        
        std::ifstream seedsFile("./masks/Dither.ppm");

        std::string dummy;
        seedsFile >> dummy;
        seedsFile >> dummy;
        seedsFile >> dummy;
        seedsFile >> dummy;

        int seed = 0;
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                seedsFile >> seed;
                mask[i][j] = seed;
                //std::cout << seed << std::endl;
                seedsFile >> seed;
                seedsFile >> seed;
            }
        }

        std::ifstream refFile("./masks/ref.txt");
        Float val = 0;
        for (int i = 0; i < 380; ++i) {
            for (int j = 0; j < 600; ++j) {
                refFile >> val;
                ref[i][j] = val;
            }
        }
    }
    Spectrum Li(const RayDifferential &ray, const Scene &scene,
                Sampler &sampler, MemoryArena &arena, int depth) const;
    std::pair<Spectrum, Spectrum> Li(const RayDifferential &ray,
                                     const Scene &scene,
                Sampler &sampler, MemoryArena &arena, int depth,
                Eigen::VectorXf& sampleCoordinates1,
                Eigen::VectorXf &sampleCoordinates2) const;
    void Preprocess(const Scene &scene, Sampler &sampler);
    void Render(const Scene &scene) override;
    

  private:
    Float G(Eigen::VectorXf &coeficients, int dimention, Point2i pixel) const;
    Float g(Eigen::VectorXf& coeficients,
            Eigen::VectorXf& sampleCoordinates) const;
    Eigen::VectorXf solveLinearSystem(Eigen::MatrixXf& A,
                                      Eigen::VectorXf& b) const;
    void updateLinearSystem(Eigen::MatrixXf& A, Eigen::VectorXf& b,
                            Eigen::VectorXf& sampleCoordinates, Spectrum& fx);

    void updateLinearSystemDL(Eigen::MatrixXf &A, Eigen::VectorXf &b,
                              Eigen::VectorXf &sampleCoordinates, Spectrum &fx);
    Float G_DL(Eigen::VectorXf &coeficients, int dimention) const;
    Float g_DL(Eigen::VectorXf &coeficients,
            Eigen::VectorXf &sampleCoordinates) const;

    Spectrum selectVal(std::vector<Spectrum> listCVEstimator,
                       Float maskVal);
    static bool comparator(Spectrum i, Spectrum j);

  private:
    // ControleVariateLSDirectLightingIntegrator Private Data
    const LightStrategyCV strategy;
    const int maxDepth;
    std::vector<int> nLightSamples;
    const int maxOrder;
    const int dimension;
    int mask[128][128];
    Float ref[380][600];
};

ControleVariateLSDirectLightingIntegrator *CreateControleVariateLSDirectLightingIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera);

std::pair<Spectrum, Spectrum> UniformSampleOneLight(
    const Interaction &it, const Scene &scene, MemoryArena &arena,
    Sampler &sampler, bool handleMedia, Eigen::VectorXf &sampleCoordinates1,
    Eigen::VectorXf &sampleCoordinates2,
    const Distribution1D *lightDistrib = nullptr);

std::pair<Spectrum, Spectrum> EstimateDirectMIS(
    const Interaction &it, const Point2f &uShading,
                        const Light &light, const Point2f &uLight,
                        const Scene &scene, Sampler &sampler,
                        MemoryArena &arena, bool handleMedia = false,
                        bool specular = false);

}  // namespace pbrt
#endif  // PBRT_INTEGRATORS_DIRECTLIGHTING_H
