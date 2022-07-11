
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

#ifndef PBRT_INTEGRATORS_OCV_H
#define PBRT_INTEGRATORS_OCV_H

// integrators/directlighting.h*
#include "pbrt.h"
#include "integrator.h"
#include "scene.h"

namespace pbrt {

struct OCVResult {
    Spectrum LeDirect = Spectrum(0.f);
    Spectrum L = Spectrum(0.f);
    Float pdfBSDF = 0.0;
    Float pdfLight = 0.0;
};

// DirectLightingIntegrator Declarations
class OCVIntegrator : public SamplerIntegrator {
  public:
    // DirectLightingIntegrator Public Methods
    OCVIntegrator(bool excludeLights, std::shared_ptr<const Camera> camera,
                     std::shared_ptr<Sampler> sampler,
                     const Bounds2i &pixelBounds)
        : SamplerIntegrator(camera, sampler, pixelBounds), excludeLights(excludeLights) {}

    // Default Li (not used)
    Spectrum Li(const RayDifferential &ray, const Scene &scene,
                        Sampler &sampler, MemoryArena &arena,
                        int depth = 0) const override {
        assert(false);
        return Spectrum {};
    }

    OCVResult LiLight(const RayDifferential &ray, const Scene &scene,
                Sampler &sampler, MemoryArena &arena,
                const Light &light) const;
    OCVResult LiBSDF(const RayDifferential &ray, const Scene &scene,
                    Sampler &sampler, MemoryArena &arena,
                    const Light& light) const;

    void Render(const Scene &scene) override;
    void Preprocess(const Scene &scene, Sampler &sampler) override;

  private:
    // DirectLightingIntegrator Private Data
    std::vector<int> nLightSamples;
    bool excludeLights = false;
};

OCVIntegrator *CreateOCVIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_OCV_H
