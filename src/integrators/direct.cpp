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

// integrators/directlighting.cpp*
#include <core/sampling.h>

#include <iostream>

#include "camera.h"
#include "film.h"
#include "integrators/direct.h"
#include "interaction.h"
#include "paramset.h"
#include "progressreporter.h"
#include "stats.h"
#include "spectrum.h"
#include <random>

// Include cvls to have sampling routines
#include "integrators/cvls_direct.h"
namespace pbrt {
STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);

// DirectLightingIntegrator_test Method Definitions
void DirectLightingIntegrator_test::Preprocess(const Scene &scene,
                                                           Sampler &sampler) {
    // Nothing
}

/*
Spectrum DirectLightingIntegrator_test::Li(
    const RayDifferential &ray, const Scene &scene, Sampler &sampler,
    MemoryArena &arena, int depth) const {
    auto sampleCoordinatesLight = Eigen::VectorXd(3);
    auto sampleCoordinatesBSDF = Eigen::VectorXd(3);
    return (Li(ray, scene, sampler, arena, depth, sampleCoordinatesLight,
               sampleCoordinatesBSDF)).first;
}
*/
std::pair<Spectrum, Spectrum> DirectLightingIntegrator_test::Li(
    const RayDifferential &ray, const Scene &scene, Sampler &sampler,
    MemoryArena &arena, 
    const Eigen::VectorXd &sampleCoordinatesLight,
    const Eigen::VectorXd &sampleCoordinatesBSDF,
    int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L_light(0.f);
    Spectrum L_bsdf(0.f);
    // Find closest ray intersection or return background radiance
    SurfaceInteraction isect;
    if (!scene.Intersect(ray, &isect)) {
        for (const auto &light : scene.lights) L_light += light->Le(ray);
        return std::pair<Spectrum, Spectrum>(L_light/2, L_light/2);
    }

    // Compute scattering functions for surface interaction
    isect.ComputeScatteringFunctions(ray, arena);
    if (!isect.bsdf)
        return Li(isect.SpawnRay(ray.d), scene, sampler, arena, sampleCoordinatesLight, sampleCoordinatesBSDF, depth);
    Vector3f wo = isect.wo;
    // Compute emitted light if ray hit an area light source
    L_light += isect.Le(wo);
    if (scene.lights.size() > 0) {
        std::pair<Spectrum, Spectrum> L = UniformSampleOneLight(
            isect, scene, arena, sampler, false, sampleCoordinatesLight,
            sampleCoordinatesBSDF, nullptr);
        L_light += L.first;
        L_bsdf += L.second;
    
    }
    if (depth + 1 < maxDepth) {
        // Trace rays for specular reflection and refraction
        std::pair<Spectrum, Spectrum> res1 =  SpecularReflect(ray, isect, scene, sampler, arena, sampleCoordinatesLight,
            sampleCoordinatesBSDF, depth);
        std::pair<Spectrum, Spectrum> res2 = SpecularTransmit(ray, isect, scene, sampler, arena, sampleCoordinatesLight,
            sampleCoordinatesBSDF, depth);

        L_light += res1.first;
        L_bsdf += res1.second;
        L_light += res2.first;
        L_bsdf += res2.second;
    }
    return std::pair<Spectrum, Spectrum>(L_light, L_bsdf);
}

void DirectLightingIntegrator_test::Render(const Scene &scene) {
    Preprocess(scene, *sampler);
    const double error_rate = 0.1;
    std::cout << sampler->samplesPerPixel << std::endl;
    // Compute the number of samples used for the control variate
    // with simpson rules and the number of samples for MC
    const size_t N = 3; // Simpson quadrature rules
    const size_t D = 2; // How many dimensions from the sampler
    size_t nb_samples_mc = sampler->samplesPerPixel;
    //nb_samples_mc = sampler->samplesPerPixel;

    std::cout << "Samples budgets: \n";
    std::cout << "  * MC        : " << nb_samples_mc << "\n";
    std::cout << "  * Total     : " << sampler->samplesPerPixel << "\n";
    // Neasted quadrature rule, nesting is necessary to compute the error
    // and have the adaptive splitting approach

    // Render image tiles in parallel

    // Compute number of tiles, _nTiles_, to use for parallel rendering
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    Vector2i sampleExtent = sampleBounds.Diagonal();
    const int tileSize = 16;
    Point2i nTiles((sampleExtent.x + tileSize - 1) / tileSize,
                   (sampleExtent.y + tileSize - 1) / tileSize);
    ProgressReporter reporter(nTiles.x * nTiles.y, "Rendering");
    {
        ParallelFor2D(
            [&](Point2i tile) {
                // Create an additional random number source
                // for the sampling decisions
                std::random_device seed_rng;
                std::mt19937 rng(seed_rng());
                // Render section of image corresponding to _tile_

                // Allocate _MemoryArena_ for tile
                MemoryArena arena;

                // Get sampler instance for tile
                int seed = tile.y * nTiles.x + tile.x;
                std::unique_ptr<Sampler> tileSampler = sampler->Clone(seed);

                // Compute sample bounds for tile
                int x0 = sampleBounds.pMin.x + tile.x * tileSize;
                int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
                int y0 = sampleBounds.pMin.y + tile.y * tileSize;
                int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
                Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));
                LOG(INFO) << "Starting image tile " << tileBounds;

                // Get _FilmTile_ for tile
                std::unique_ptr<FilmTile> filmTile =
                    camera->film->GetFilmTile(tileBounds);
                
                // Loop over pixels in tile to render them
                for (Point2i pixel : tileBounds) {
                    {
                        ProfilePhase pp(Prof::StartPixel);
                        tileSampler->StartPixel(pixel);
                    }


                    // Do this check after the StartPixel() call; this keeps
                    // the usage of RNG values from (most) Samplers that use
                    // RNGs consistent, which improves reproducability /
                    // debugging.
                    if (!InsideExclusive(pixel, pixelBounds)) continue;


                    // Rendering loop (collect all the samples)
                    do {
                        // Initialize _CameraSample_ for current sample
                        CameraSample cameraSample
                            = tileSampler->GetCameraSample(pixel);

                        // Generate camera ray for current sample
                        RayDifferential ray;
                        //ray.ScaleDifferentials(
                        //    1 / std::sqrt((Float)tileSampler->samplesPerPixel));
                        //++nCameraRays;

                        // Lambda to warp integrator call.
                        // Note that we use same random number for BSDF and light
                        // sampling.
                        Spectrum result = 0;
                        std::uniform_real_distribution<Float> distribution(0.0,1.0);
                        for(std::size_t i = 0; i < nb_samples_mc; i++) {
                            cameraSample = tileSampler->GetCameraSample(pixel);
                            Float rayWeight =
                                camera->GenerateRayDifferential(cameraSample, &ray);
                            ray.ScaleDifferentials(
                                1 / std::sqrt((Float)tileSampler->samplesPerPixel));
                            ++nCameraRays;

                            // Evaluate the function at the random location
                            // and also the fitted quadrature
                            Eigen::VectorXd sampleCoordinatesLight(2);
                            Eigen::VectorXd sampleCoordinatesBSDF(2);
                            for (size_t i = 0; i < D; i++) {
                                Float rnd = distribution(rng);
                                sampleCoordinatesLight[i] = rnd;
                                sampleCoordinatesBSDF[i] = rnd;
                            }
                            //sampleCoordinatesLight[0] = 0.1f;
                            //sampleCoordinatesBSDF[0] = 0.1f;
                            
                            std::pair<Spectrum, Spectrum> L = Li(ray, scene, *tileSampler, arena,
                                   sampleCoordinatesLight, sampleCoordinatesBSDF, 0);
                            result += ((L.first + L.second) * rayWeight)/nb_samples_mc;

                            // Save the sample (for computing alpha and the residual)
                            tileSampler->StartNextSample();
                        }

                        filmTile->AddSample(Point2f(pixel) + Point2f(0.5, 0.5), result, 1.0);
                        // Free _MemoryArena_ memory from computing image sample
                        // value
                        arena.Reset();
                    } while (false); // no AA: tileSampler->StartNextSample()
                }

                LOG(INFO) << "Finished image tile " << tileBounds;

                // Merge image tile into _Film_
                camera->film->MergeFilmTile(std::move(filmTile));
                reporter.Update();
            },
            nTiles);
        reporter.Done();
    }
    LOG(INFO) << "Rendering finished";

    // Save final image after rendering
    camera->film->WriteImage();
}



DirectLightingIntegrator_test *
CreateDirectLightingIntegrator_test(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera) {
    int maxDepth = params.FindOneInt("maxdepth", 5);
    bool use_alpha = params.FindOneBool("useAlpha", true);
    int batchSize = params.FindOneInt("batchSize", 1);
    maxDepth = 5;
    int np;
    const int *pb = params.FindInt("pixelbounds", &np);
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    if (pb) {
        if (np != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  np);
        else {
            pixelBounds = Intersect(pixelBounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixelBounds.Area() == 0)
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }
    return new DirectLightingIntegrator_test(maxDepth, camera, sampler, pixelBounds, use_alpha);
}

}  // namespace pbrt
