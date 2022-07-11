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
#include "camera.h"
#include "film.h"
#include "integrators/controlVariateLeastSquare.h"
#include "interaction.h"
#include "paramset.h"
#include "stats.h"
#include "progressreporter.h"
#include <iostream>
#include <core\sampling.h>
/**/
namespace pbrt {
STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);
// ControleVariateLSDirectLightingIntegrator Method Definitions
void ControleVariateLSDirectLightingIntegrator::Preprocess(const Scene &scene,
                                          Sampler &sampler) {
    if (strategy == LightStrategyCV::UniformSampleAll) {
        // Compute number of samples to use for each light
        for (const auto &light : scene.lights)
            nLightSamples.push_back(sampler.RoundCount(light->nSamples));

        // Request samples for sampling all lights
        for (int i = 0; i < maxDepth; ++i) {
            for (size_t j = 0; j < scene.lights.size(); ++j) {
                sampler.Request2DArray(nLightSamples[j]);
                sampler.Request2DArray(nLightSamples[j]);
            }
        }
    }
}


Spectrum ControleVariateLSDirectLightingIntegrator::Li(
    const RayDifferential &ray, const Scene &scene, Sampler &sampler,
    MemoryArena &arena, int depth) const {
    auto sampleCoordinates1 = Eigen::VectorXf(dimension);
    auto sampleCoordinates2 = Eigen::VectorXf(dimension);
    return (Li(ray, scene, sampler, arena, depth, sampleCoordinates1, sampleCoordinates2)).first;
}

std::pair<Spectrum,Spectrum> ControleVariateLSDirectLightingIntegrator::Li(
    const RayDifferential &ray,
    const Scene &scene, Sampler &sampler,
    MemoryArena &arena, int depth, Eigen::VectorXf& sampleCoordinates1,
    Eigen::VectorXf& sampleCoordinates2) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L1(0.f);
    Spectrum L2(0.f);
    // Find closest ray intersection or return background radiance
    SurfaceInteraction isect;
    if (!scene.Intersect(ray, &isect)) {
        for (const auto &light : scene.lights) L1 += light->Le(ray);
        return std::pair<Spectrum,Spectrum>(L1/2,L1/2);
    }

    // Compute scattering functions for surface interaction
    isect.ComputeScatteringFunctions(ray, arena);
    if (!isect.bsdf)
        return Li(isect.SpawnRay(ray.d), scene, sampler, arena, depth, sampleCoordinates1,sampleCoordinates2);
    Vector3f wo = isect.wo;
    // Compute emitted light if ray hit an area light source
    L1 += isect.Le(wo);
    if (scene.lights.size() > 0) {
        // Compute direct lighting for _ControleVariateLSDirectLightingIntegrator_ integrator
        if (strategy == LightStrategyCV::UniformSampleAll) {
            L1 += UniformSampleAllLights(isect, scene, arena, sampler,
                                        nLightSamples);
        }
        else {
            std::pair<Spectrum, Spectrum> L =
                UniformSampleOneLight(isect, scene, arena, sampler, false,
                                      sampleCoordinates1, sampleCoordinates2, nullptr);
            L1 += L.first;
            L2 += L.second;
        }
    }
    if (depth + 1 < maxDepth) {
        // Trace rays for specular reflection and refraction
        L1 += SpecularReflect(ray, isect, scene, sampler, arena, depth);
        L1 += SpecularTransmit(ray, isect, scene, sampler, arena, depth);
    }
    return std::pair<Spectrum, Spectrum>(L1,L2);
}

std::pair<Spectrum, Spectrum> UniformSampleOneLight(
    const Interaction& it, const Scene& scene, MemoryArena& arena,
    Sampler& sampler, bool handleMedia, Eigen::VectorXf& sampleCoordinates1,
    Eigen::VectorXf& sampleCoordinates2, const Distribution1D* lightDistrib) {
    ProfilePhase p(Prof::DirectLighting);
    // Randomly choose a single light to sample, _light_
    int nLights = int(scene.lights.size());
    if (nLights == 0)
        return std::pair<Spectrum, Spectrum>(Spectrum(0.f), Spectrum(0.f));
    int lightNum;
    Float lightPdf;
    Float sample1 = sampler.Get1D();
    sampleCoordinates1(0) = sample1;
    sampleCoordinates2(0) = sample1;
    if (lightDistrib) {
        lightNum = lightDistrib->SampleDiscrete(sample1, &lightPdf);
        if (lightPdf == 0)
            return std::pair<Spectrum, Spectrum>(Spectrum(0.f), Spectrum(0.f));
    } else {
        lightNum = std::min((int)(sample1 * nLights), nLights - 1);
        lightPdf = Float(1) / nLights;
    }
    const std::shared_ptr<Light>& light = scene.lights[lightNum];
    Point2f uLight = sampler.Get2D();
    sampleCoordinates1(1) = uLight.x;
    sampleCoordinates1(2) = uLight.y;
    Point2f uScattering = sampler.Get2D();
    sampleCoordinates2(1) = uScattering.x;
    sampleCoordinates2(2) = uScattering.y;
    std::pair<Spectrum, Spectrum> result =
        EstimateDirectMIS(
        it, uScattering, *light, uLight, scene, sampler,
                          arena, handleMedia,false);
    result.first = result.first / lightPdf;
    result.second = result.second / lightPdf;
    return result;
}

std::pair<Spectrum, Spectrum> EstimateDirectMIS(
    const Interaction &it, const Point2f &uScattering,
               const Light &light, const Point2f &uLight, const Scene &scene,
               Sampler &sampler, MemoryArena &arena, bool handleMedia,
               bool specular) {
    BxDFType bsdfFlags =
        specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    std::pair<Spectrum, Spectrum> Ld(0.f,0.f);
    // Sample light source with multiple importance sampling
    Vector3f wi;
    Float lightPdf = 0, scatteringPdf = 0;
    VisibilityTester visibility;
    Spectrum Li = light.Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
    VLOG(2) << "EstimateDirect uLight:" << uLight << " -> Li: " << Li
            << ", wi: " << wi << ", pdf: " << lightPdf;
    if (lightPdf > 0 && !Li.IsBlack()) {
        // Compute BSDF or phase function's value for light sample
        Spectrum f;
        if (it.IsSurfaceInteraction()) {
            // Evaluate BSDF for light sampling strategy
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->f(isect.wo, wi, bsdfFlags) *
                AbsDot(wi, isect.shading.n);
            scatteringPdf = isect.bsdf->Pdf(isect.wo, wi, bsdfFlags);
            VLOG(2) << "  surf f*dot :" << f
                    << ", scatteringPdf: " << scatteringPdf;
        } else {
            // Evaluate phase function for light sampling strategy
            const MediumInteraction &mi = (const MediumInteraction &)it;
            Float p = mi.phase->p(mi.wo, wi);
            f = Spectrum(p);
            scatteringPdf = p;
            VLOG(2) << "  medium p: " << p;
        }
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
                if (IsDeltaLight(light.flags))
                    Ld.first += f * Li / lightPdf;
                else {
                    Float weight =
                        PowerHeuristic(1, lightPdf, 1, scatteringPdf);
                    //weight = 1.0;  // disable MIS
                    Ld.first += f * Li * weight / lightPdf;
                }
            }
        }
    }

    // Sample BSDF with multiple importance sampling
    if (!IsDeltaLight(light.flags)) {
        Spectrum f;
        bool sampledSpecular = false;
        if (it.IsSurfaceInteraction()) {
            // Sample scattered direction for surface interactions
            BxDFType sampledType;
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->Sample_f(isect.wo, &wi, uScattering, &scatteringPdf,
                                     bsdfFlags, &sampledType);
            f *= AbsDot(wi, isect.shading.n);
            sampledSpecular = (sampledType & BSDF_SPECULAR) != 0;
        } else {
            // Sample scattered direction for medium interactions
            const MediumInteraction &mi = (const MediumInteraction &)it;
            Float p = mi.phase->Sample_p(mi.wo, &wi, uScattering);
            f = Spectrum(p);
            scatteringPdf = p;
        }
        VLOG(2) << "  BSDF / phase sampling f: " << f
                << ", scatteringPdf: " << scatteringPdf;
        if (!f.IsBlack() && scatteringPdf > 0) {
            // Account for light contributions along sampled direction _wi_
            Float weight = 1;
            if (!sampledSpecular) {
                lightPdf = light.Pdf_Li(it, wi);
                if (lightPdf == 0) return Ld;
                weight = PowerHeuristic(1, scatteringPdf, 1, lightPdf);
            }
            //weight = 0.0;  // disable MIS
            // Find intersection and compute transmittance
            SurfaceInteraction lightIsect;
            Ray ray = it.SpawnRay(wi);
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
            if (!Li.IsBlack()) Ld.second += f * Li * Tr * weight / scatteringPdf;
        }
    }
    return Ld;
}


void ControleVariateLSDirectLightingIntegrator::Render(const Scene &scene) {
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
        ParallelFor2D(
            [&](Point2i tile) {
                // Render section of image corresponding to _tile_

                // Allocate _MemoryArena_ for tile
                MemoryArena arena;

                // Get sampler instance for tile
                int seed = tile.y *nTiles.x + tile.x;
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

                

                std::vector<std::vector<Eigen::VectorXf>>
                    blocSamplesCoordinates1;
                std::vector<Spectrum> blocMeanFx1;
                std::vector<std::vector<Float>>
                    blocFunctionValues1;
                std::vector<Eigen::VectorXf> blocCoeficients1;

                std::vector<std::vector<Eigen::VectorXf>>
                    blocSamplesCoordinates2;
                std::vector<Spectrum> blocMeanFx2;
                std::vector<std::vector<Float>> blocFunctionValues2;
                std::vector<Eigen::VectorXf> blocCoeficients2;
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
      
                    Spectrum mean_fx_1 = 0;
                    Float sumRayWeights_1 = 0.0;
                    Spectrum mean_fx_2 = 0;
                    Float sumRayWeights_2 = 0.0;
                    std::vector<Eigen::VectorXf> listSamplesCoordinates1;
                    std::vector<Eigen::VectorXf> listSamplesCoordinates2;
                    std::vector<Float> listFunctionValues1;
                    std::vector<Float> listFunctionValues2;
                    
                    // initialise linear system

                    int linear_system_dimension = 1;
                    if (maxOrder == 2 && dimension == 3) {
                        linear_system_dimension = 10;
                    } else {
                        for (int order = 1; order < maxOrder + 1; ++order) {
                            linear_system_dimension +=
                                std::pow(dimension, order);
                        }
                    }

                    Eigen::VectorXf coeficeints1(linear_system_dimension);
                    Eigen::VectorXf coeficeints2(linear_system_dimension);
                    Eigen::MatrixXf A1(linear_system_dimension,
                                      linear_system_dimension);
                    Eigen::MatrixXf A2(linear_system_dimension,
                                      linear_system_dimension);
                    Eigen::VectorXf b1(linear_system_dimension);
                    Eigen::VectorXf b2(linear_system_dimension);
                    for (size_t i = 0; i < linear_system_dimension; i++) {
                        b1(i) = 0;
                        b2(i) = 0;
                        coeficeints1(i) = 0;
                        coeficeints2(i) = 0;
                        for (size_t j = 0; j < linear_system_dimension; j++) {
                            A1(i, j) = 0;
                            A2(i, j) = 0;
                        }
                    }

                    // Rendering loop (collect all the samples)
                    do {
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
                        Eigen::VectorXf sampleCoordinates1(dimension);
                        Eigen::VectorXf sampleCoordinates2(dimension);
                        for (size_t i = 0; i < dimension; i++) {
                            sampleCoordinates1(i) = tileSampler->Get1D();
                            sampleCoordinates2(i) = tileSampler->Get1D();
                        }
                        Spectrum L1(0.f);
                        Spectrum L2(0.f);
                        if (rayWeight > 0) {
                            std::pair<Spectrum, Spectrum> L =
                                Li(ray, scene, *tileSampler, arena, 0,
                                   sampleCoordinates1, sampleCoordinates2);
                            L1 = L.first;
                            L2 = L.second;
                        }
                            

                        // Issue warning if unexpected radiance value returned
                        if (L1.HasNaNs()) {
                            LOG(ERROR) << StringPrintf(
                                "Not-a-number radiance value returned "
                                "for pixel (%d, %d), sample %d. Setting to "
                                "black.",
                                pixel.x, pixel.y,
                                (int)tileSampler->CurrentSampleNumber());
                            L1 = Spectrum(0.f);
                        } else if (L1.y() < -1e-5) {
                            LOG(ERROR) << StringPrintf(
                                "Negative luminance value, %f, returned "
                                "for pixel (%d, %d), sample %d. Setting to "
                                "black.",
                                L1.y(), pixel.x, pixel.y,
                                (int)tileSampler->CurrentSampleNumber());
                            L1 = Spectrum(0.f);
                        } else if (std::isinf(L1.y())) {
                            LOG(ERROR) << StringPrintf(
                                "Infinite luminance value returned "
                                "for pixel (%d, %d), sample %d. Setting to "
                                "black.",
                                pixel.x, pixel.y,
                                (int)tileSampler->CurrentSampleNumber());
                            L1 = Spectrum(0.f);
                        }

                        if (L2.HasNaNs()) {
                            LOG(ERROR) << StringPrintf(
                                "Not-a-number radiance value returned "
                                "for pixel (%d, %d), sample %d. Setting to "
                                "black.",
                                pixel.x, pixel.y,
                                (int)tileSampler->CurrentSampleNumber());
                            L2 = Spectrum(0.f);
                        } else if (L2.y() < -1e-5) {
                            LOG(ERROR) << StringPrintf(
                                "Negative luminance value, %f, returned "
                                "for pixel (%d, %d), sample %d. Setting to "
                                "black.",
                                L2.y(), pixel.x, pixel.y,
                                (int)tileSampler->CurrentSampleNumber());
                            L2 = Spectrum(0.f);
                        } else if (std::isinf(L2.y())) {
                            LOG(ERROR) << StringPrintf(
                                "Infinite luminance value returned "
                                "for pixel (%d, %d), sample %d. Setting to "
                                "black.",
                                pixel.x, pixel.y,
                                (int)tileSampler->CurrentSampleNumber());
                            L2 = Spectrum(0.f);
                        }
                        VLOG(1) << "Camera sample: " << cameraSample
                                << " -> ray: " << ray << " -> L = " << L1;

                        // Add camera ray's contribution to image
                        //filmTile->AddSample(cameraSample.pFilm, L, rayWeight);
                        mean_fx_1 += L1 * rayWeight;
                        sumRayWeights_1 += rayWeight;

                        mean_fx_2 += L2 * rayWeight;
                        sumRayWeights_2 += rayWeight;

                        // Free _MemoryArena_ memory from computing image sample
                        // value
                        arena.Reset();

                        // update linear system 
                        Spectrum f = L1 * rayWeight;

                        if (maxOrder == 2 && dimension == 3) {
                            updateLinearSystemDL(A1, b1, sampleCoordinates1, f);
                        } else {
                            updateLinearSystem(A1, b1, sampleCoordinates1, f);
                        }

                        

                        //updateCoeficients(coeficeints, sampleCoordinates, f);
                        listSamplesCoordinates1.push_back(sampleCoordinates1);
                        listFunctionValues1.push_back(f.y());

                        f = L2 * rayWeight;

                        if (maxOrder == 2 && dimension == 3) {
                            updateLinearSystemDL(A2, b2, sampleCoordinates2, f);
                        } else {
                            updateLinearSystem(A2, b2, sampleCoordinates2, f);
                        }

                        listSamplesCoordinates2.push_back(sampleCoordinates2);
                        listFunctionValues2.push_back(f.y());
                        
                    } while (tileSampler->StartNextSample());
                    blocMeanFx1.push_back(mean_fx_1/sumRayWeights_1);
                    blocSamplesCoordinates1.push_back(listSamplesCoordinates1);
                    blocFunctionValues1.push_back(listFunctionValues1);

                    //coeficeints = solveLinearSystem(A, b);
                    blocCoeficients1.push_back(solveLinearSystem(A1, b1));

                    blocMeanFx2.push_back(mean_fx_2 / sumRayWeights_2);
                    blocSamplesCoordinates2.push_back(listSamplesCoordinates2);
                    blocFunctionValues2.push_back(listFunctionValues2);

                    // coeficeints = solveLinearSystem(A, b);
                    blocCoeficients2.push_back(solveLinearSystem(A2, b2));
                }


                // Do the control variate
                std::vector<std::vector<Spectrum>> blocSpectrums;
                Spectrum sumBlocSpectrum = 0;
                int count = 0;
                int p = 0;
                for (Point2i pixel : tileBounds) {
                    std::vector<Spectrum> listCVEstimator;
                    for (size_t c = 0; c < blocCoeficients1.size(); c++) {
                        {
                            std::vector<Eigen::VectorXf>
                                listSamplesCoordinates =
                                    blocSamplesCoordinates1[p];
                            Spectrum mean_fx = blocMeanFx1[p];
                            // solve linear system
                            Eigen::VectorXf coeficeints = blocCoeficients1[c];

                            // add result to film
                            Float mean_gx = 0.0;
                            Float mean_fgx = 0.0;
                            std::vector<Float> list_gx;
                            for (size_t i = 0;
                                 i < listSamplesCoordinates.size(); i++) {
                                Float gx;
                                if (maxOrder == 2 && dimension == 3) {
                                    gx = g_DL(coeficeints,
                                              listSamplesCoordinates[i]);
                                } else {
                                    gx = g(coeficeints,
                                           listSamplesCoordinates[i]);
                                }
                                mean_gx += gx / listSamplesCoordinates.size();
                                mean_fgx += gx * blocFunctionValues1[p][i] /
                                            listSamplesCoordinates.size();
                                list_gx.push_back(gx);
                            }

                            Float var_g = 0.0;
                            for (size_t i = 0;
                                 i < listSamplesCoordinates.size(); i++) {
                                var_g += std::pow(list_gx[i] - mean_gx, 2) /
                                         listSamplesCoordinates.size();
                            }

                            // Image space control variate alpha
                            Float alpha =
                                (mean_fgx - mean_fx.y() * mean_gx) / var_g;
                            // alpha = 1.0;
                            Spectrum result;
                            if (!mean_fx.IsBlack()) {
                                Spectrum color = (mean_fx / mean_fx.y());
                                Float G_val;
                                if (maxOrder == 2 && dimension == 3) {
                                    G_val = G_DL(coeficeints, dimension);
                                } else {
                                    G_val = G(coeficeints, dimension,
                                              Point2i(0, 0));
                                }

                                // Main equation
                                result = color * alpha * G_val +
                                         (mean_fx - color * alpha * mean_gx);
                                // result = color * G_val;
                                 filmTile->AddSample(
                                    Point2f(pixel) + Point2f(0.5, 0.5),
                                    result*2, 1.0);
                                listCVEstimator.push_back(result);
                                sumBlocSpectrum += result;
                                ++count;
                            } else {
                                 filmTile->AddSample(
                                    Point2f(pixel) + Point2f(0.5, 0.5),
                                    mean_fx*2, 1.0);
                                // listCVEstimator.push_back(mean_fx);
                                sumBlocSpectrum += mean_fx;
                                ++count;
                            }
                        }
                        {
                            std::vector<Eigen::VectorXf>
                                listSamplesCoordinates =
                                    blocSamplesCoordinates2[p];
                            Spectrum mean_fx = blocMeanFx2[p];
                            // solve linear system
                            Eigen::VectorXf coeficeints = blocCoeficients2[c];

                            // add result to film
                            Float mean_gx = 0.0;
                            Float mean_fgx = 0.0;
                            std::vector<Float> list_gx;
                            for (size_t i = 0;
                                 i < listSamplesCoordinates.size(); i++) {
                                Float gx;
                                if (maxOrder == 2 && dimension == 3) {
                                    gx = g_DL(coeficeints,
                                              listSamplesCoordinates[i]);
                                } else {
                                    gx = g(coeficeints,
                                           listSamplesCoordinates[i]);
                                }
                                mean_gx += gx / listSamplesCoordinates.size();
                                mean_fgx += gx * blocFunctionValues2[p][i] /
                                            listSamplesCoordinates.size();
                                list_gx.push_back(gx);
                            }

                            Float var_g = 0.0;
                            for (size_t i = 0;
                                 i < listSamplesCoordinates.size(); i++) {
                                var_g += std::pow(list_gx[i] - mean_gx, 2) /
                                         listSamplesCoordinates.size();
                            }

                            // Image space control variate alpha
                            Float alpha =
                                (mean_fgx - mean_fx.y() * mean_gx) / var_g;
                            // alpha = 1.0;
                            Spectrum result;
                            if (!mean_fx.IsBlack()) {
                                Spectrum color = (mean_fx / mean_fx.y());
                                Float G_val;
                                if (maxOrder == 2 && dimension == 3) {
                                    G_val = G_DL(coeficeints, dimension);
                                } else {
                                    G_val = G(coeficeints, dimension,
                                              Point2i(0, 0));
                                }

                                // Main equation
                                result = color * alpha * G_val +
                                         (mean_fx - color * alpha * mean_gx);
                                // result = color * G_val;
                                 filmTile->AddSample(
                                    Point2f(pixel) + Point2f(0.5, 0.5),
                                    result*2, 1.0);
                                listCVEstimator.push_back(result);
                                sumBlocSpectrum += result;
                                ++count;
                            } else {
                                 filmTile->AddSample(
                                    Point2f(pixel) + Point2f(0.5, 0.5),
                                    mean_fx*2, 1.0);
                                // listCVEstimator.push_back(mean_fx);
                                sumBlocSpectrum += mean_fx;
                                ++count;
                            }
                        }
                    }
                    blocSpectrums.push_back(listCVEstimator);
                    ++p;
                }

                p=0;
                Float blocError[16][16];
                for (size_t i = 0; i < tileSize; i++) {
                    for (size_t j = 0; j < tileSize; j++) {
                        blocError[i][j] = 0.0;
                    }
                }
                for (Point2i pixel : tileBounds) {
                    std::vector<Spectrum> listCVEstimator = blocSpectrums[p];
                    Spectrum mean = sumBlocSpectrum / count;

                    if (listCVEstimator.size() != 0) {
                        //float maskVal = mask[pixel.x%128][pixel.y%128]/256.0;
                        float maskVal = mean.y();
                        //maskVal = ref[pixel.y][pixel.x];
                        Spectrum value =
                            selectVal(listCVEstimator,
                                      maskVal);//
                        //filmTile->AddSample(Point2f(pixel) + Point2f(0.5, 0.5),
                        //                    value, 1.0);
                        /*
                        blocError[(pixel.x) % tileSize][(pixel.y+1) % tileSize] +=
                            (maskVal + blocError[pixel.x % tileSize][pixel.y % tileSize] - value.y()) * (7 / 16.0);
                        blocError[(pixel.x+1) % tileSize][(pixel.y - 1) % tileSize] +=
                            (maskVal + blocError[(pixel.x) % tileSize]
                                      [(pixel.y) % tileSize] - value.y()) * (3 / 16.0);
                        blocError[(pixel.x+1) % tileSize][(pixel.y) % tileSize] +=
                            (maskVal + blocError[(pixel.x) % tileSize]
                                      [(pixel.y) % tileSize] - value.y()) * (5 / 16.0);
                        blocError[(pixel.x+1) % tileSize][(pixel.y + 1) % tileSize] +=
                            (maskVal + blocError[(pixel.x) % tileSize]
                                      [(pixel.y) % tileSize] - value.y()) * (1 / 16.0);
                        */
                    } else {
                        //filmTile->AddSample(Point2f(pixel) + Point2f(0.5, 0.5),
                        //                    0,
                        //                    1.0);
                    }

                    ++p;
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



Float ControleVariateLSDirectLightingIntegrator::G(
    Eigen::VectorXf& coeficients, int dimention, Point2i pixel) const {
    double funtion_value = coeficients[0];
    int count = 1;
    for (int order = 1; order < maxOrder + 1; ++order) {
        std::vector<int> variate_combination(dimention);
        for (int i = 0; i < std::pow(dimention, order); ++i) {
            for (int d = 0; d < dimention; ++d) {
                int val = 0;
                for (int t = 1; t < order + 1; ++t) {
                    if (d == int(i / (t * dimention))) {
                        val += 1;
                    }
                }
                variate_combination[d] = val;
            }
            double val = coeficients[count + i];
            for (int d = 0; d < dimention; ++d) {
                val *= 1.0 / (variate_combination[d] + 1);
                if (d == 0) {
                    val *= (std::pow(pixel.x + 1, variate_combination[d] + 1) -
                            std::pow(pixel.x, variate_combination[d] + 1));
                }
                if (d == 1) {
                    val *= (std::pow(pixel.y + 1, variate_combination[d] + 1) -
                            std::pow(pixel.y, variate_combination[d] + 1));
                }
            }
            funtion_value += val;
        }
        count += std::pow(dimention, order);
    }

    return funtion_value;
}

Float ControleVariateLSDirectLightingIntegrator::g(
    Eigen::VectorXf& coeficients, Eigen::VectorXf& sampleCoordinates) const {
    int dimension = sampleCoordinates.size();
    double funtion_value = coeficients[0];
    int count = 1;
    for (int order = 1; order < maxOrder + 1; ++order) {
        std::vector<int> variate_combination(dimension);
        for (int i = 0; i < std::pow(dimension, order); ++i) {
            for (int d = 0; d < dimension; ++d) {
                int val = 0;
                for (int t = 1; t < order + 1; ++t) {
                    if (d == int(i / (t * dimension))) {
                        val += 1;
                    }
                }
                variate_combination[d] = val;
            }

            double val = coeficients[count + i];
            for (int d = 0; d < dimension; ++d) {
                val *= std::pow(sampleCoordinates[d], variate_combination[d]);
            }
            funtion_value += val;
        }
        count += std::pow(dimension, order);
    }

    return funtion_value;
}

Eigen::VectorXf ControleVariateLSDirectLightingIntegrator::solveLinearSystem(
    Eigen::MatrixXf& A, Eigen::VectorXf& b) const {

    for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }

    Eigen::VectorXf X(A.cols());
    //X = A.completeOrthogonalDecomposition().pseudoInverse() * b;
    //X = A.llt().solve(b);
    X = A.completeOrthogonalDecomposition().solve(b);
    return X;
}

void ControleVariateLSDirectLightingIntegrator::updateLinearSystem(
    Eigen::MatrixXf& A, Eigen::VectorXf& b, Eigen::VectorXf& sampleCoordinates,
    Spectrum& fx) {
    int dimension = sampleCoordinates.size();
    int i = 0;
    int j = 0;
    for (int order_i = 0; order_i < maxOrder + 1; ++order_i) {
        std::vector<int> variate_combination_x1(dimension);
        for (int combi_i = 0; combi_i < std::pow(dimension, order_i);
             ++combi_i) {
            for (int d = 0; d < dimension; ++d) {
                int val = 0;
                for (int t = 1; t < order_i + 1; ++t) {
                    if (d == int(combi_i / (t * dimension))) {
                        val += 1;
                    }
                }
                variate_combination_x1[d] = val;
            }
            Float x1 = 1.0;
            for (int d = 0; d < dimension; ++d) {
                x1 *= std::pow(sampleCoordinates(d),
                               variate_combination_x1[d]);
            }
            b(i) += x1 * fx.y();
            int jbis = j;
            for (int order_j = order_i; order_j < maxOrder + 1; ++order_j) {
                std::vector<int> variate_combination_x2(dimension);
                jbis += combi_i;
                for (int combi_j = combi_i; combi_j < std::pow(dimension, order_j);
                     ++combi_j) {
                    for (int d = 0; d < dimension; ++d) {
                        int val = 0;
                        for (int t = 1; t < order_j + 1; ++t) {
                            if (d == int(combi_j / (t * dimension))) {
                                val += 1;
                            }
                        }
                        variate_combination_x2[d] = val;
                    }
                    Float x2 = 1.0;
                    for (int d = 0; d < dimension; ++d) {
                        x2 *= std::pow(sampleCoordinates(d),
                                       variate_combination_x2[d]);
                    }
                    A(i, jbis) += x1 * x2;
                    
                    ++jbis;
                }
            }
            ++i;

        }
        j += std::pow(dimension, order_i);
    }
}

void ControleVariateLSDirectLightingIntegrator::updateLinearSystemDL(
    Eigen::MatrixXf& A, Eigen::VectorXf& b, Eigen::VectorXf& sampleCoordinates,
    Spectrum& fx) {
    const int dimension = 3;
    const int order = 2;

    // This is the code to update the system
    Float combinations[10];
    combinations[0] = 1.0;
    combinations[1] = sampleCoordinates[0];
    combinations[2] = sampleCoordinates[1];
    combinations[3] = sampleCoordinates[2];
    combinations[4] = sampleCoordinates[0] * sampleCoordinates[0];
    combinations[5] = sampleCoordinates[0] * sampleCoordinates[1];
    combinations[6] = sampleCoordinates[0] * sampleCoordinates[2];
    combinations[7] = sampleCoordinates[1] * sampleCoordinates[1];
    combinations[8] = sampleCoordinates[1] * sampleCoordinates[2];
    combinations[9] = sampleCoordinates[2] * sampleCoordinates[2];

    for (int i = 0; i < 10; ++i) {
        Float x1 = combinations[i];
        b(i) += x1 * fx.y();
        for (int j = i; j < 10; ++j) { 
            Float x2 = combinations[j];
            A(i, j) += x1 * x2;
        }
    }
}

Float ControleVariateLSDirectLightingIntegrator::G_DL(
    Eigen::VectorXf& coeficients, int dimention) const {
    double funtion_value = coeficients[0];
    int div[10];
    div[0] = 1;
    div[1] = 2;
    div[2] = 2;
    div[3] = 2;
    div[4] = 3;
    div[5] = 4;
    div[6] = 4;
    div[7] = 3;
    div[8] = 4;
    div[9] = 3;
    for (int i = 1; i < 10; ++i) {
        Float val = coeficients[i] / div[i];
        funtion_value += val;
    }
    return funtion_value;
}

Float ControleVariateLSDirectLightingIntegrator::g_DL(
    Eigen::VectorXf& coeficients, Eigen::VectorXf& sampleCoordinates) const {
    double funtion_value = coeficients[0];
    Float combinations[10];
    combinations[0] = 1.0;
    combinations[1] = sampleCoordinates[0];
    combinations[2] = sampleCoordinates[1];
    combinations[3] = sampleCoordinates[2];
    combinations[4] = sampleCoordinates[0] * sampleCoordinates[0];
    combinations[5] = sampleCoordinates[0] * sampleCoordinates[1];
    combinations[6] = sampleCoordinates[0] * sampleCoordinates[2];
    combinations[7] = sampleCoordinates[1] * sampleCoordinates[1];
    combinations[8] = sampleCoordinates[1] * sampleCoordinates[2];
    combinations[9] = sampleCoordinates[2] * sampleCoordinates[2];
    for (int i = 1; i < 10; ++i) {
        Float val = coeficients[i] * combinations[i];
        funtion_value += val;
    }
    return funtion_value;
}

Spectrum ControleVariateLSDirectLightingIntegrator::selectVal(
    std::vector<Spectrum> listCVEstimator, Float maskVal) {
    std::sort(listCVEstimator.begin(), listCVEstimator.end(), comparator);
    Spectrum result;
    for (size_t i = 0; i < listCVEstimator.size(); i++) {
        if (abs(listCVEstimator[i].y() - maskVal) < abs(result.y() - maskVal)) {
            result = listCVEstimator[i];
        }
    }
    return result;
    //return listCVEstimator[int(listCVEstimator.size()*maskVal)];
    //return listCVEstimator[listCVEstimator.size() - 1] - listCVEstimator[0];
}

bool ControleVariateLSDirectLightingIntegrator::comparator(
    Spectrum i, Spectrum j){return (i.y() < j.y());}

ControleVariateLSDirectLightingIntegrator *CreateControleVariateLSDirectLightingIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera) {
    int maxDepth = params.FindOneInt("maxdepth", 5);
    int maxOrder = params.FindOneInt("maxOrder", 0);
    int dimension = 3;
    LightStrategyCV strategy;
    std::string st = params.FindOneString("strategy", "all");
    if (st == "one")
        strategy = LightStrategyCV::UniformSampleOne;
    else if (st == "all")
        strategy = LightStrategyCV::UniformSampleAll;
    else {
        Warning(
            "Strategy \"%s\" for direct lighting unknown. "
            "Using \"all\".",
            st.c_str());
        strategy = LightStrategyCV::UniformSampleAll;
    }
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
    return new ControleVariateLSDirectLightingIntegrator(strategy, maxDepth, camera, sampler,
                                        pixelBounds,maxOrder,dimension);
}

}  // namespace pbrt
