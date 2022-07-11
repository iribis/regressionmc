
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

#ifndef PBRT_CORE_INTEGRATOR_H
#define PBRT_CORE_INTEGRATOR_H

// core/integrator.h*
#include "pbrt.h"
#include "primitive.h"
#include "spectrum.h"
#include "light.h"
#include "reflection.h"
#include "sampler.h"
#include "material.h"
#include "ext/eigen/Eigen/Dense"

namespace pbrt {

// Integrator Declarations
class Integrator {
  public:
    // Integrator Interface
    virtual ~Integrator();
    virtual void Render(const Scene &scene) = 0;
};

Spectrum UniformSampleAllLights(const Interaction &it, const Scene &scene,
                                MemoryArena &arena, Sampler &sampler,
                                const std::vector<int> &nLightSamples,
                                bool handleMedia = false);
Spectrum UniformSampleOneLight(const Interaction &it, const Scene &scene,
                               MemoryArena &arena, Sampler &sampler,
                               bool handleMedia,
                               Eigen::VectorXd& sampleCoordinates,
                               const Distribution1D *lightDistrib = nullptr);
inline Spectrum UniformSampleOneLight(const Interaction &it, const Scene &scene,
                               MemoryArena &arena, Sampler &sampler,
                               bool handleMedia = false,
                               const Distribution1D *lightDistrib = nullptr) {
    Eigen::VectorXd sampleCoordinates = Eigen::VectorXd(5);
    return UniformSampleOneLight(it, scene, arena, sampler, handleMedia, sampleCoordinates, lightDistrib);
}
Spectrum EstimateDirect(const Interaction &it, const Point2f &uShading,
                        const Light &light, const Point2f &uLight,
                        const Scene &scene, Sampler &sampler,
                        MemoryArena &arena, bool handleMedia = false,
                        bool specular = false);
std::unique_ptr<Distribution1D> ComputeLightPowerDistribution(
    const Scene &scene);

// SamplerIntegrator Declarations
class SamplerIntegrator : public Integrator {
  public:
    // SamplerIntegrator Public Methods
    SamplerIntegrator(std::shared_ptr<const Camera> camera,
                      std::shared_ptr<Sampler> sampler,
                      const Bounds2i &pixelBounds)
        : camera(camera), sampler(sampler), pixelBounds(pixelBounds) {}
    virtual void Preprocess(const Scene &scene, Sampler &sampler) {}
    virtual void Render(const Scene &scene);
    virtual Spectrum Li(const RayDifferential &ray, const Scene &scene,
                        Sampler &sampler, MemoryArena &arena,
                        int depth = 0) const = 0;
    Spectrum SpecularReflect(const RayDifferential &ray,
                             const SurfaceInteraction &isect,
                             const Scene &scene, Sampler &sampler,
                             MemoryArena &arena, int depth) const;
    Spectrum SpecularTransmit(const RayDifferential &ray,
                              const SurfaceInteraction &isect,
                              const Scene &scene, Sampler &sampler,
                              MemoryArena &arena, int depth) const;

  protected:
    // SamplerIntegrator Protected Data
    std::shared_ptr<const Camera> camera;
    // SamplerIntegrator Private Data
    std::shared_ptr<Sampler> sampler;
    const Bounds2i pixelBounds;
};

class SamplerIntegratorEigen : public Integrator {
  public:
    // SamplerIntegrator Public Methods
    SamplerIntegratorEigen(std::shared_ptr<const Camera> camera,
                      std::shared_ptr<Sampler> sampler,
                      const Bounds2i &pixelBounds)
        : camera(camera), sampler(sampler), pixelBounds(pixelBounds) {}
    virtual void Preprocess(const Scene &scene, Sampler &sampler) {}
    virtual void Render(const Scene &scene) {
      LOG(ERROR) << "Call render!\n";
      return;  
    }
    virtual std::pair<Spectrum, Spectrum> Li(const RayDifferential &ray, const Scene &scene,
                        Sampler &sampler, 
                        MemoryArena &arena,
                        const Eigen::VectorXd &sampleCoordinatesLight,
                        const Eigen::VectorXd &sampleCoordinatesBSDF,
                        int depth = 0) const = 0;
    std::pair<Spectrum, Spectrum> SpecularReflect(const RayDifferential &ray,
                             const SurfaceInteraction &isect,
                             const Scene &scene, Sampler &sampler,
                             MemoryArena &arena, 
                             const Eigen::VectorXd &sampleCoordinatesLight,
                             const Eigen::VectorXd &sampleCoordinatesBSDF,
                             int depth) const;
    std::pair<Spectrum, Spectrum> SpecularTransmit(const RayDifferential &ray,
                              const SurfaceInteraction &isect,
                              const Scene &scene, Sampler &sampler,
                              MemoryArena &arena, 
                              const Eigen::VectorXd &sampleCoordinatesLight,
                              const Eigen::VectorXd &sampleCoordinatesBSDF,
                              int depth) const;

  protected:
    // SamplerIntegrator Protected Data
    std::shared_ptr<const Camera> camera;
    // SamplerIntegrator Private Data
    std::shared_ptr<Sampler> sampler;
    const Bounds2i pixelBounds;
};

std::pair<Spectrum, Spectrum> UniformSampleOneLight(
    const Interaction &it, const Scene &scene, MemoryArena &arena,
    Sampler &sampler, bool handleMedia, 
    const Eigen::VectorXd &sampleCoordinates1,
    const Eigen::VectorXd &sampleCoordinates2,
    const Distribution1D *lightDistrib = nullptr);

std::pair<Spectrum, Spectrum> EstimateDirectMIS(
    const Interaction &it, const Point2f &uShading, const Light &light,
    const Point2f &uLight, const Scene &scene, Sampler &sampler,
    MemoryArena &arena, bool handleMedia = false, bool specular = false);

class SamplerIntegratorEigenDirect : public SamplerIntegratorEigen {
  public:

    SamplerIntegratorEigenDirect(std::shared_ptr<const Camera> camera,
                      std::shared_ptr<Sampler> sampler,
                      const Bounds2i &pixelBounds,
                      const int maxDepth)
        : SamplerIntegratorEigen(camera, sampler, pixelBounds), maxDepth(maxDepth) {}

    virtual std::pair<Spectrum, Spectrum> Li(const RayDifferential &ray, const Scene &scene,
                        Sampler &sampler, 
                        MemoryArena &arena,
                        const Eigen::VectorXd &sampleCoordinatesLight,
                        const Eigen::VectorXd &sampleCoordinatesBSDF,
                        int depth = 0) const override;
  private:
    const int maxDepth = 5; 
};

}  // namespace pbrt

#endif  // PBRT_CORE_INTEGRATOR_H
