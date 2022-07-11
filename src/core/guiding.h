
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

#ifndef PBRT_CORE_GUIDING_H
#define PBRT_CORE_GUIDING_H

// core/guiding.h*
#include "pbrt.h"
#include "interaction.h"
#include "light.h"

namespace pbrt {

class GuidingDistribution {
public:
	enum class SamplingProjection { SphereSimple, SpherePrecise, ParallelPlane, None };

	class ProjectedTriangle {
	private:
		struct {
			Vector3f A, B, C;
			Vector3f normal;
			float area = -1.f;
		} sphereSimple;

		struct {
			Vector3f A, B, C;
			float alpha, beta, gamma;
			float a, b, c;
			float area = -1.f;
		} spherePrecise;

		struct {
			Vector3f A, B, C;
			Vector3f normal;
			float area = -1.f;
		} parallelPlane;

		bool hitsTriangle(const Vector3f &A, const Vector3f &B, const Vector3f &C, const Vector3f &wi, float *dist) const;
	public:
		ProjectedTriangle() {}
		ProjectedTriangle(const Point3f &origin, const Normal3f &normal, const Point3f &v0, const Point3f &v1, const Point3f &v2);

		Vector3f Sample(const Point2f &u, SamplingProjection projection, Float *pdf) const;
		float Pdf(const Vector3f &wi, SamplingProjection projection) const;
		bool CanSample(SamplingProjection projection) const;
	};

	GuidingDistribution(const SurfaceInteraction &it, const Light &light);

	Vector3f Sample_wi(const Point2f &u, SamplingProjection projection, Float *pdf) const;

	float Pdf(const Vector3f &wi, SamplingProjection projection) const;

	bool CanSample(SamplingProjection projection) const;

private:
	bool isTriangle;
	ProjectedTriangle projectedTriangle;
	Vector3f ns, ss, ts;
	float cosMaxAngle;	

	Vector3f WorldToLocal(const Vector3f &v) const {
		return Vector3f(Dot(v, ss), Dot(v, ts), Dot(v, ns));
	}

	Vector3f LocalToWorld(const Vector3f &v) const {
		return Vector3f(ss.x * v.x + ts.x * v.y + ns.x * v.z,
			ss.y * v.x + ts.y * v.y + ns.y * v.z,
			ss.z * v.x + ts.z * v.y + ns.z * v.z);
	}
};

}  // namespace pbrt

#endif  // PBRT_CORE_GUIDING_H
