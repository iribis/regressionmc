
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
#include "integrators/ocv.h"
#include "interaction.h"
#include "paramset.h"
#include "camera.h"
#include "film.h"
#include "stats.h"
#include "progressreporter.h"

#include "ext/eigen/Eigen/Dense"

namespace pbrt {

STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);

// DirectLightingIntegrator Method Definitions
void OCVIntegrator::Preprocess(const Scene &scene,
                                          Sampler &sampler) {
    // For now, we will sample the light one by one
    // TODO: Check if we need to implement this so QMC is working
//    if (strategy == LightStrategy::UniformSampleAll) {
//        // Compute number of samples to use for each light
//        for (const auto &light : scene.lights)
//            nLightSamples.push_back(sampler.RoundCount(light->nSamples));
//
//        // Request samples for sampling all lights
//        for (int i = 0; i < maxDepth; ++i) {
//            for (size_t j = 0; j < scene.lights.size(); ++j) {
//                sampler.Request2DArray(nLightSamples[j]);
//                sampler.Request2DArray(nLightSamples[j]);
//            }
//        }
//    }
}

void OCVIntegrator::Render(const Scene &scene) {
    Preprocess(scene, *sampler);
    // Render image tiles in parallel

    // Compute number of tiles, _nTiles_, to use for parallel rendering
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    Vector2i sampleExtent = sampleBounds.Diagonal();
    const int tileSize = 1;
    Point2i nTiles((sampleExtent.x + tileSize - 1) / tileSize,
                   (sampleExtent.y + tileSize - 1) / tileSize);
    ProgressReporter reporter(nTiles.x * nTiles.y, "Rendering");
    {
        ParallelFor2D([&](Point2i tile) {
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

                // Matrix: (Row, Col)
                // Note that produce one sample from the light and one from the BSDF
                Eigen::MatrixXf y(2*sampler->samplesPerPixel, 3); // 3 = RGB
                Eigen::MatrixXf A(2*sampler->samplesPerPixel,2); // 2 = intersept and light

                // Do this check after the StartPixel() call; this keeps
                // the usage of RNG values from (most) Samplers that use
                // RNGs consistent, which improves reproducability /
                // debugging.
                if (!InsideExclusive(pixel, pixelBounds))
                    continue;

                Spectrum LeDirect(0.f);
                for(auto id_sample=0;; id_sample++) {
                    // TODO: Sample in the pixel center?
                    // Initialize _CameraSample_ for current sample
                    CameraSample cameraSample =
                            tileSampler->GetCameraSample(pixel);

                    // Generate camera ray for current sample
                    RayDifferential ray;
                    Float rayWeight =
                            camera->GenerateRayDifferential(cameraSample, &ray);
                    ray.ScaleDifferentials(
                            1 / std::sqrt((Float)tileSampler->samplesPerPixel));
                    ++nCameraRays;

                    // Evaluate radiance along camera ray
                    Spectrum L(0.f);

                    // Note that we have 2 samples...
                    A(2*id_sample, 0) = 1; // Intercept term
                    A(2*id_sample+1, 0) = 1; // Intercept term

                    // TODO: For now, we put zero everywhere
                    //  Because I am not sure if the matrix is correctly initialized
                    A(2*id_sample, 1) = 0;
                    A(2*id_sample+1, 1) = 0;
                    y.row(2*id_sample) << 0, 0, 0;
                    y.row(2*id_sample + 1) << 0, 0, 0;

                    // TODO: We only accept box filters..
                    assert(rayWeight == 1.0); // Box filter

                    if (rayWeight > 0) {
                        // TODO: For now, pick a light randomly
                        Float sample1 = tileSampler->Get1D();
                        int nLights = int(scene.lights.size());
                        int lightNum = std::min((int)(sample1 * nLights), nLights - 1);
                        Float lightPdfSel = Float(1) / nLights;
                        const std::shared_ptr<Light> &light = scene.lights[lightNum];

                        // TODO: Do the surface intersection here
                        // Light sampling
                        auto resultLight = LiLight(ray, scene, *tileSampler, arena, *light);
                        // BSDF sampling
                        auto resultBSDF = LiBSDF(ray, scene, *tileSampler, arena, *light);

                        // Add the super direct
                        if(!excludeLights) {
                            LeDirect += resultLight.LeDirect;
                            LeDirect += resultBSDF.LeDirect;
                        }

                        // TODO: Do we need to divide with hte lightPdfSel?
                        A(2*id_sample, 1) = resultLight.pdfLight;
                        A(2*id_sample+1, 1) = resultBSDF.pdfLight;

                        // Rescale estimate with the light picking probability
                        resultLight.L /= lightPdfSel;
                        resultBSDF.L /= lightPdfSel;

                        // Update the y vector
                        Float rgb[3];
                        resultLight.L.ToRGB(rgb);
                        y.row(2*id_sample) << rgb[0], rgb[1], rgb[2];
                        resultBSDF.L.ToRGB(rgb);
                        y.row(2*id_sample + 1) << rgb[0], rgb[1], rgb[2];
                    }

                    // Free _MemoryArena_ memory from computing image sample
                    // value
                    arena.Reset();

                    if(!tileSampler->StartNextSample()) {
                        break; // We are done with this pixel
                    }
                }

                LeDirect /= sampler->samplesPerPixel * 2;

                // Accumulating the result
                {
                    Eigen::MatrixXf reg(2,2);
                    reg(0,0) = 1.0;
                    reg(1,1) = 1.0;
                    reg(1,0) = 0.0;
                    reg(0,1) = 0.0;

                    auto beta = (A.transpose()*A + reg).inverse() * A.transpose() * y;

                    // Add camera ray's contribution to image
                    const auto cameraSample = (Point2f)pixel + Point2f(0.5, 0.5);
                    const auto rayWeight = 1.0;

                    Float rgb[] = {beta.col(0).sum() * 0.5f, beta.col(1).sum() * 0.5f, beta.col(2).sum() * 0.5f};
                    filmTile->AddSample(cameraSample, LeDirect + RGBSpectrum::FromRGB(rgb), rayWeight);
                }
            }
            LOG(INFO) << "Finished image tile " << tileBounds;

            // Merge image tile into _Film_
            camera->film->MergeFilmTile(std::move(filmTile));
            reporter.Update();
        }, nTiles);
        reporter.Done();
    }
    LOG(INFO) << "Rendering finished";

    // Save final image after rendering
    camera->film->WriteImage();
}

OCVResult OCVIntegrator::LiBSDF(const RayDifferential &ray,
                                 const Scene &scene, Sampler &sampler,
                                 MemoryArena &arena, const Light& light) const {
    // TODO: OCV will only works with light that we can intersect
    assert(!IsDeltaLight(light.flags));
    const bool specular = false;
    const bool handleMedia = false;

    ProfilePhase p(Prof::SamplerIntegratorLi);
    OCVResult result = {};

    // Find closest ray intersection or return background radiance
    SurfaceInteraction isect;
    if (!scene.Intersect(ray, &isect)) {
        for (const auto &light : scene.lights) result.LeDirect += light->Le(ray);
        return result;
    }

    // Compute scattering functions for surface interaction
    isect.ComputeScatteringFunctions(ray, arena);
    if (!isect.bsdf) {
        // FIXME: We do not support null BDSF yet (volumes?)
        return result;
    }
    Vector3f wo = isect.wo;

    // Compute emitted light if ray hit an area light source
    result.LeDirect += isect.Le(wo);

    // TODO: Specular reflecting is not handled
    {
        ProfilePhase p(Prof::DirectLighting);
        // Randomly choose a single light to sample, _light_
        Point2f uScattering = sampler.Get2D();
        BxDFType bsdfFlags = specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);

        // Sample light source with explicit sampling
        Vector3f wi;
        Float lightPdf = 0;
        Float scatteringPdf = 0;

        VisibilityTester visibility;
        if (!IsDeltaLight(light.flags)) {
            Spectrum f;
            bool sampledSpecular = false;
            if (isect.IsSurfaceInteraction()) {
                // Sample scattered direction for surface interactions
                BxDFType sampledType;
                const SurfaceInteraction &isect2 = (const SurfaceInteraction &)isect;
                f = isect2.bsdf->Sample_f(isect2.wo, &wi, uScattering, &scatteringPdf,
                                         bsdfFlags, &sampledType);
                f *= AbsDot(wi, isect2.shading.n);
                sampledSpecular = (sampledType & BSDF_SPECULAR) != 0;
            } else {
                // Sample scattered direction for medium interactions
                const MediumInteraction &mi = (const MediumInteraction &)isect;
                Float p = mi.phase->Sample_p(mi.wo, &wi, uScattering);
                f = Spectrum(p);
                scatteringPdf = p;
            }
            VLOG(2) << "  BSDF / phase sampling f: " << f << ", scatteringPdf: " <<
                    scatteringPdf;
            if (!f.IsBlack() && scatteringPdf > 0) {
                // Account for light contributions along sampled direction _wi_
                if (!sampledSpecular) {
                    lightPdf = light.Pdf_Li(isect, wi);
                    if (lightPdf == 0) return result;
                }

                // 1 / p(w, alpha)
                const Float inv_denominator = 1.0 / (lightPdf + scatteringPdf);
                result.pdfLight = lightPdf * inv_denominator;
                result.pdfBSDF = scatteringPdf * inv_denominator;

                // Find intersection and compute transmittance
                SurfaceInteraction lightIsect;
                Ray ray = isect.SpawnRay(wi);
                Spectrum Tr(1.f);
                bool foundSurfaceInteraction =
                        handleMedia ? scene.IntersectTr(ray, sampler, &lightIsect, &Tr)
                                    : scene.Intersect(ray, &lightIsect);

                // Add light contribution from material sampling
                Spectrum Li(0.f);
                if (foundSurfaceInteraction) {
                    if (lightIsect.primitive->GetAreaLight() == &light)
                        Li = lightIsect.Le(-wi);
                } else
                    Li = light.Le(ray);

                if (!Li.IsBlack()) {
                    result.L += f * Li * Tr * inv_denominator;
                }
            }
        }
    }

    return result;
}

OCVResult OCVIntegrator::LiLight(const RayDifferential &ray,
                           const Scene &scene, Sampler &sampler,
                           MemoryArena &arena, const Light& light) const {
    // TODO: OCV will only works with light that we can intersect
    assert(!IsDeltaLight(light.flags));
    const bool specular = false;
    const bool handleMedia = false;

    ProfilePhase p(Prof::SamplerIntegratorLi);
    OCVResult result = {};

    // Find closest ray intersection or return background radiance
    SurfaceInteraction isect;
    if (!scene.Intersect(ray, &isect)) {
        for (const auto &light : scene.lights) result.LeDirect += light->Le(ray);
        return result;
    }

    // Compute scattering functions for surface interaction
    isect.ComputeScatteringFunctions(ray, arena);
    if (!isect.bsdf) {
        // FIXME: We do not support null BDSF yet (volumes?)
        assert(false);
    }
    Vector3f wo = isect.wo;

    // Compute emitted light if ray hit an area light source
    result.LeDirect += isect.Le(wo);

    // TODO: Specular reflecting is not handled
    {
        ProfilePhase p(Prof::DirectLighting);
        // Randomly choose a single light to sample, _light_
        Point2f uLight = sampler.Get2D();

        BxDFType bsdfFlags = specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);

        // Sample light source with explicit sampling
        Vector3f wi;
        Float lightPdf = 0;
        VisibilityTester visibility;
        Spectrum Li = light.Sample_Li(isect, uLight, &wi, &lightPdf, &visibility);

        VLOG(2) << "EstimateDirect uLight:" << uLight << " -> Li: " << Li << ", wi: "
                << wi << ", pdf: " << lightPdf;

        if (lightPdf > 0 && !Li.IsBlack()) {
            // Compute BSDF or phase function's value for light sample
            Spectrum f; //< BDSF value
            Float scatteringPdf = 0;
            if (isect.IsSurfaceInteraction()) {
                // Evaluate BSDF for light sampling strategy
                const SurfaceInteraction &isect2 = (const SurfaceInteraction &)isect;
                f = isect2.bsdf->f(isect2.wo, wi, bsdfFlags) *
                    AbsDot(wi, isect2.shading.n);
                scatteringPdf = isect2.bsdf->Pdf(isect2.wo, wi, bsdfFlags);
                VLOG(2) << "  surf f*dot :" << f << ", scatteringPdf: " << scatteringPdf;
            } else {
                // Evaluate phase function for light sampling strategy
                const MediumInteraction &mi = (const MediumInteraction &)isect;
                Float p = mi.phase->p(mi.wo, wi);
                f = Spectrum(p);
                scatteringPdf = p;
                VLOG(2) << "  medium p: " << p;
            }

            // 1 / p(w, alpha)
            const Float inv_denominator = 1.0 / (lightPdf + scatteringPdf);
            result.pdfLight = lightPdf * inv_denominator;
            result.pdfBSDF = scatteringPdf * inv_denominator;

            if (!f.IsBlack()) {
                // Compute effect of visibility for light source sample
                if (handleMedia) {
                    Li *= visibility.Tr(scene, sampler);
                    VLOG(2) << "  after Tr, Li: " << Li;
                } else {
                    if (!visibility.Unoccluded(scene)) {
                        VLOG(2) << "  shadow ray blocked";
                        Li = Spectrum(0.f);
                    } else
                        VLOG(2) << "  shadow ray unoccluded";
                }

                // Add light's contribution to reflected radiance
                if (!Li.IsBlack()) {
                    if (IsDeltaLight(light.flags)) {
                        assert(false);
                        //Ld += f * Li / lightPdf;
                    } else {
                        result.L += f * Li * inv_denominator;
                    }
                }
            }
        }
    }

    return result;
}

OCVIntegrator *CreateOCVIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera) {
    bool excludeLights = params.FindOneBool("excludelights", false);
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
    return new OCVIntegrator(excludeLights, camera, sampler, pixelBounds);
}

}  // namespace pbrt
