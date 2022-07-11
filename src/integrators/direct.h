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

#ifndef PBRT_INTEGRATORS_direct_test_H
#define PBRT_INTEGRATORS_direct_test_H

// integrators/directlighting.h*
#include <algorithm>  // std::sort
#include <fstream>

#include "ext/eigen/Eigen/Dense"
#include "integrator.h"
#include "pbrt.h"
#include "scene.h"

namespace pbrt {

// DirectLightingIntegrator_test Declarations
class DirectLightingIntegrator_test : public SamplerIntegratorEigen {
  public:
    // DirectLightingIntegrator_test Public Methods
    DirectLightingIntegrator_test(
        int maxDepth,
        std::shared_ptr<const Camera> camera, std::shared_ptr<Sampler> sampler,
        const Bounds2i &pixelBounds, bool use_alpha)
        : SamplerIntegratorEigen(camera, sampler, pixelBounds),
          maxDepth(maxDepth),
          use_alpha(use_alpha)
          {}

    //Spectrum Li(const RayDifferential &ray, const Scene &scene,
    //            Sampler &sampler, MemoryArena &arena, int depth) const;
    std::pair<Spectrum, Spectrum> Li(const RayDifferential &ray,
                                     const Scene &scene, Sampler &sampler,
                                     MemoryArena &arena, 
                                     const Eigen::VectorXd &sampleCoordinates1,
                                     const Eigen::VectorXd &sampleCoordinates2,
                                     int depth) const override;
    void Preprocess(const Scene &scene, Sampler &sampler);
    void Render(const Scene &scene) override;
    Float computeAlpha(std::vector<std::tuple<Spectrum, Spectrum>> samples);

  private:
    // DirectLightingIntegrator_test Private Data
    const int maxDepth;
    const bool use_alpha;
};

DirectLightingIntegrator_test *
CreateDirectLightingIntegrator_test(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera);

}  // namespace pbrt
#endif  // PBRT_INTEGRATORS_DIRECTLIGHTING_H
