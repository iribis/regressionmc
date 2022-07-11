
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

// core/guiding.cpp*
#include "guiding.h"

namespace pbrt {

// ProjectedTriangle Method Definitions
GuidingDistribution::ProjectedTriangle::ProjectedTriangle(const Point3f &origin, const Normal3f &normal, const Point3f &v0, const Point3f &v1, const Point3f &v2) {
	sphereSimple.area = -1.f;
	spherePrecise.area = -1.f;
	parallelPlane.area = -1.f;
	
	const Vector3f v0local = v0 - origin;
	const Vector3f v1local = v1 - origin;
	const Vector3f v2local = v2 - origin;

	// SphereSimple projection
	sphereSimple.A = Normalize(v0local);
	sphereSimple.B = Normalize(v1local);
	sphereSimple.C = Normalize(v2local);
	sphereSimple.area = 0.5 * Cross(sphereSimple.B - sphereSimple.A, sphereSimple.C - sphereSimple.A).Length();

    const float minArea = 1e-5f;

    if (sphereSimple.area <= minArea) {
		sphereSimple.area = -1.f;
	} else {
		sphereSimple.normal = Normalize(Cross(sphereSimple.B - sphereSimple.A, sphereSimple.C - sphereSimple.A));

		// SpherePrecise projection
		spherePrecise.A = sphereSimple.A;
		spherePrecise.B = sphereSimple.B;
		spherePrecise.C = sphereSimple.C;

		spherePrecise.a = std::acos(Clamp(Dot(spherePrecise.B, spherePrecise.C), -1, 1));
		spherePrecise.b = std::acos(Clamp(Dot(spherePrecise.C, spherePrecise.A), -1, 1));
		spherePrecise.c = std::acos(Clamp(Dot(spherePrecise.A, spherePrecise.B), -1, 1));

		Vector3f crossAB = Cross(spherePrecise.A, spherePrecise.B);
		Vector3f crossBC = Cross(spherePrecise.B, spherePrecise.C);
		Vector3f crossCA = Cross(spherePrecise.C, spherePrecise.A);

		if (crossAB.LengthSquared() > 0) crossAB = Normalize(crossAB);
		if (crossBC.LengthSquared() > 0) crossBC = Normalize(crossBC);
		if (crossCA.LengthSquared() > 0) crossCA = Normalize(crossCA);

		spherePrecise.alpha = std::acos(Clamp(Dot(crossCA, -crossAB), -1, 1));
		spherePrecise.beta = std::acos(Clamp(Dot(crossAB, -crossBC), -1, 1));
		spherePrecise.gamma = std::acos(Clamp(Dot(crossBC, -crossCA), -1, 1));
		
		spherePrecise.area = spherePrecise.alpha + spherePrecise.beta + spherePrecise.gamma - Pi;
		
        if (spherePrecise.area <= minArea || spherePrecise.alpha <= 0.f || spherePrecise.beta <= 0.f || spherePrecise.gamma <= 0.f || spherePrecise.a <= 0.f || spherePrecise.b <= 0.f || spherePrecise.c <= 0.f) {
			spherePrecise.area = -1.f;
		} 
		
		const float eps = 0.0001f;
		if (spherePrecise.alpha >= (Pi - eps) && spherePrecise.beta >= (Pi - eps) && spherePrecise.gamma >= (Pi - eps)) {
			// Don't sample triangle if it covers almost entire hemisphere
			sphereSimple.area = -1.f;
			spherePrecise.area = -1.f;
		}
	}

	// ParallelPlane projection
	float dot0 = Dot(v0local, normal);
	float dot1 = Dot(v1local, normal);
	float dot2 = Dot(v2local, normal);

	if (dot0 < 0.f && dot1 < 0.f && dot2 < 0.f) {
		dot0 = -dot0;
		dot1 = -dot1;
		dot2 = -dot2;
	}

	const float eps = 0.1f;
	if (dot0 > eps && dot1 > eps && dot2 > eps) {
		parallelPlane.A = v0local / dot0;
		parallelPlane.B = v1local / dot1;
		parallelPlane.C = v2local / dot2;
		parallelPlane.area = 0.5 * Cross(parallelPlane.B - parallelPlane.A, parallelPlane.C - parallelPlane.A).Length();
        if (parallelPlane.area <= minArea) {
			parallelPlane.area = -1.f;
		}
		else {
			parallelPlane.normal = -Normalize(Cross(parallelPlane.B - parallelPlane.A, parallelPlane.C - parallelPlane.A));
		}
	}
}

Vector3f GuidingDistribution::ProjectedTriangle::Sample(const Point2f &r, SamplingProjection projection, Float *pdf) const {
	if (!CanSample(projection)) {
		Error("GuidingDistribution::ProjectedTriangle::Sample: Invalid projection required!");
		return Vector3f();
	}

	switch (projection) {
	case SamplingProjection::SphereSimple:
		{
			const float su0 = std::sqrt(r[0]);
			const Point2f b = Point2f(1 - su0, r[1] * su0);
			const Vector3f pointOnTriangle = b[0] * sphereSimple.A + b[1] * sphereSimple.B + (1 - b[0] - b[1]) * sphereSimple.C;
			const Vector3f wi = Normalize(pointOnTriangle);
			if (pdf) {
                const float coslight = AbsDot(sphereSimple.normal, -wi);
                if (coslight == 0.f) {
                    *pdf = 0.f;
                } else {
                    *pdf = 1.f / sphereSimple.area * pointOnTriangle.LengthSquared() / coslight;
                }
			}
			return wi;
		}
	case SamplingProjection::SpherePrecise:
		{
			// https://www.graphics.cornell.edu/pubs/1995/Arv95c.pdf
			const float ra = r[0] * spherePrecise.area;
			const float s = std::sin(ra - spherePrecise.alpha);
			const float t = std::cos(ra - spherePrecise.alpha);
			const float u = t - std::cos(spherePrecise.alpha);
			const float v = s + std::sin(spherePrecise.alpha) * std::cos(spherePrecise.c);
			const float q = Clamp(((v * t - u * s) * std::cos(spherePrecise.alpha) - v) / ((v * s + u * t) * std::sin(spherePrecise.alpha)), -1, 1);
			const Vector3f C2 = q * spherePrecise.A + std::sqrt(1 - q * q) * Normalize(spherePrecise.C - Dot(spherePrecise.C, spherePrecise.A) * spherePrecise.A);
			const float z = Clamp(1 - r[1] * (1 - Dot(C2, spherePrecise.B)), -1, 1);
			const Vector3f P = ((C2 - Dot(C2, spherePrecise.B) * spherePrecise.B).LengthSquared() == 0.f) ? z * spherePrecise.B : z * spherePrecise.B + std::sqrt(1 - z * z) * Normalize(C2 - Dot(C2, spherePrecise.B) * spherePrecise.B);
			if (pdf) {
				*pdf = 1.f / spherePrecise.area;
			}
			return Normalize(P);
		}
	case SamplingProjection::ParallelPlane:
		{
			const float su0 = std::sqrt(r[0]);
			const Point2f b = Point2f(1 - su0, r[1] * su0);
			const Vector3f pointOnTriangle = b[0] * parallelPlane.A + b[1] * parallelPlane.B + (1 - b[0] - b[1]) * parallelPlane.C;
			const Vector3f wi = Normalize(pointOnTriangle);			
			if (pdf) {
                const float coslight = AbsDot(parallelPlane.normal, -wi);
                if (coslight == 0.f) {
                    *pdf = 0.f;
                } else {
                    *pdf = 1.f / parallelPlane.area * pointOnTriangle.LengthSquared() / coslight;
                }
			}
			return wi;
		}
	default:
		Error("GuidingDistribution::ProjectedTriangle::Sample: Uknown projection required!");
		return Vector3f();
	}
}

float GuidingDistribution::ProjectedTriangle::Pdf(const Vector3f &wi, SamplingProjection projection) const {
	if (!CanSample(projection)) {
		Error("GuidingDistribution::ProjectedTriangle::Pdf: Invalid projection required!");
		return 0.f;
	}
    float coslight;
	float hitDist;
	bool hit;
	switch (projection) {
	case SamplingProjection::SphereSimple:
        coslight = AbsDot(sphereSimple.normal, -wi);
        if (coslight == 0.f) {
            return 0.f;
        }
		hit = hitsTriangle(sphereSimple.A, sphereSimple.B, sphereSimple.C, wi, &hitDist);
		if (hit) {
            return 1.f / sphereSimple.area * hitDist * hitDist / coslight;
		}
		else {
			return 0.f;
		}
	case SamplingProjection::SpherePrecise:
		hit = hitsTriangle(spherePrecise.A, spherePrecise.B, spherePrecise.C, wi, &hitDist);
		if (hit) {
			return 1.f / spherePrecise.area;
		}
		else {
			return 0.f;
		}
	case SamplingProjection::ParallelPlane:
        coslight = AbsDot(parallelPlane.normal, -wi);
        if (coslight == 0.f) {
            return 0.f;
        }
		hit = hitsTriangle(parallelPlane.A, parallelPlane.B, parallelPlane.C, wi, &hitDist);
		if (hit) {
            return 1.f / parallelPlane.area * hitDist * hitDist / coslight;
		}
		else {
			return 0.f;
		}
	default:
		Error("GuidingDistribution::ProjectedTriangle::Pdf: Uknown projection required!");
		return 0.f;
	}
}

bool GuidingDistribution::ProjectedTriangle::CanSample(SamplingProjection projection) const {
	switch (projection) {
	case SamplingProjection::SphereSimple:
		return sphereSimple.area > 0.f;
	case SamplingProjection::SpherePrecise:
		return spherePrecise.area > 0.f;
	case SamplingProjection::ParallelPlane:
		return parallelPlane.area > 0.f;
	default:
		return false;
	}
}

bool GuidingDistribution::ProjectedTriangle::hitsTriangle(const Vector3f &A, const Vector3f &B, const Vector3f &C, const Vector3f &wi, float *dist) const {
	Vector3f p0t = A;
	Vector3f p1t = B;
	Vector3f p2t = C;

	// Permute components of triangle vertices and direction wi
	int kz = MaxDimension(Abs(wi));
	int kx = kz + 1;
	if (kx == 3) kx = 0;
	int ky = kx + 1;
	if (ky == 3) ky = 0;
	Vector3f d = Permute(wi, kx, ky, kz);
	p0t = Permute(p0t, kx, ky, kz);
	p1t = Permute(p1t, kx, ky, kz);
	p2t = Permute(p2t, kx, ky, kz);

	// Apply shear transformation to vertex positions
	Float Sx = -d.x / d.z;
	Float Sy = -d.y / d.z;
	Float Sz = 1.f / d.z;
	p0t.x += Sx * p0t.z;
	p0t.y += Sy * p0t.z;
	p1t.x += Sx * p1t.z;
	p1t.y += Sy * p1t.z;
	p2t.x += Sx * p2t.z;
	p2t.y += Sy * p2t.z;

	// Compute edge function coefficients _e0_, _e1_, and _e2_
	Float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
	Float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
	Float e2 = p0t.x * p1t.y - p0t.y * p1t.x;

	// Fall back to double precision test at triangle edges
	if (sizeof(Float) == sizeof(float) &&
		(e0 == 0.0f || e1 == 0.0f || e2 == 0.0f)) {
		double p2txp1ty = (double)p2t.x * (double)p1t.y;
		double p2typ1tx = (double)p2t.y * (double)p1t.x;
		e0 = (float)(p2typ1tx - p2txp1ty);
		double p0txp2ty = (double)p0t.x * (double)p2t.y;
		double p0typ2tx = (double)p0t.y * (double)p2t.x;
		e1 = (float)(p0typ2tx - p0txp2ty);
		double p1txp0ty = (double)p1t.x * (double)p0t.y;
		double p1typ0tx = (double)p1t.y * (double)p0t.x;
		e2 = (float)(p1typ0tx - p1txp0ty);
	}

	// Perform triangle edge and determinant tests
	if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
		return false;
	Float det = e0 + e1 + e2;
	if (det == 0) return false;

	// Compute scaled hit distance to triangle and test against infinite range
	p0t.z *= Sz;
	p1t.z *= Sz;
	p2t.z *= Sz;
	Float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
	if (det < 0 && tScaled >= 0)
		return false;
	else if (det > 0 && tScaled <= 0)
		return false;

	// Compute barycentric coordinates and $t$ value for triangle intersection
	Float invDet = 1 / det;
	Float b0 = e0 * invDet;
	Float b1 = e1 * invDet;
	Float b2 = e2 * invDet;
	Float t = tScaled * invDet;

	// Ensure that computed triangle $t$ is conservatively greater than zero

	// Compute $\delta_z$ term for triangle $t$ error bounds
	Float maxZt = MaxComponent(Abs(Vector3f(p0t.z, p1t.z, p2t.z)));
	Float deltaZ = gamma(3) * maxZt;

	// Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
	Float maxXt = MaxComponent(Abs(Vector3f(p0t.x, p1t.x, p2t.x)));
	Float maxYt = MaxComponent(Abs(Vector3f(p0t.y, p1t.y, p2t.y)));
	Float deltaX = gamma(5) * (maxXt + maxZt);
	Float deltaY = gamma(5) * (maxYt + maxZt);

	// Compute $\delta_e$ term for triangle $t$ error bounds
	Float deltaE =
		2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

	// Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
	Float maxE = MaxComponent(Abs(Vector3f(e0, e1, e2)));
	Float deltaT = 3 *
		(gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) *
		std::abs(invDet);
	if (t <= deltaT) return false;

	if (dist) *dist = t;
	return true;
}

// GuidingDistribution Method Definitions
GuidingDistribution::GuidingDistribution(const SurfaceInteraction &it, const Light &light) {
	// Fallback solution of uniform sampling of the entire hemisphere
	isTriangle = false;
	cosMaxAngle = 0.f;
	ns = Vector3f(it.shading.n);
	ss = Normalize(it.shading.dpdu);
	ts = Cross(ns, ss);

	// Get light's bounding box
	const Bounds3f lightBBox = light.Bounds();
	
	// For any other than finite area light sample uniformly the hemisphere
	if (lightBBox.SurfaceArea() <= 0.f) {
		return;
	}

	// Try to get light's triangle vertices
	Point3f v0, v1, v2;
	isTriangle = light.GetTriangleVertices(&v0, &v1, &v2);

	// For finite area lights that are not a triangle uniformly sample spherical cap
	// corresponding to projecting their bounding sphere onto unit sphere 
	if (!isTriangle) {
		// Compute bounding sphere from the box
		Point3f sphereCenter;
		Float sphereRadius;
		lightBBox.BoundingSphere(&sphereCenter, &sphereRadius);

		const Vector3f normal = Vector3f(it.shading.n);
		const Vector3f dirToCenter = sphereCenter - it.p;
		const float distToCenter = dirToCenter.Length();

		// If inside the bounding sphere, sample uniformly the hemisphere
		if (distToCenter < sphereRadius) {			
			return;
		}

		// Otherwise sample just the spherical cap corresponding to the projection of the bounding sphere
		cosMaxAngle = std::cos(std::asin(sphereRadius / distToCenter));
		ns = Normalize(dirToCenter);
		if (ns == normal) {
			ss = Normalize(it.shading.dpdu);
		}
		else {
			ss = Normalize(normal - Dot(normal, ns) * ns);
		}
		ts = Cross(ns, ss);
		return;
	}

	// Project the triangle
	projectedTriangle = ProjectedTriangle(it.p, it.shading.n, v0, v1, v2);
}

Vector3f GuidingDistribution::Sample_wi(const Point2f &u, SamplingProjection projection, Float *pdf) const {
	if (isTriangle) {
		if (projectedTriangle.CanSample(projection)) {
			return projectedTriangle.Sample(u, projection, pdf);
		}
		else if (projection == SamplingProjection::SpherePrecise && projectedTriangle.CanSample(SamplingProjection::SphereSimple)) {
			return projectedTriangle.Sample(u, SamplingProjection::SphereSimple, pdf);
		}
	}
	
	// If the triangle can't be sampled, sample the sphere cap
	const float common = std::sqrt(1.f - (1.f - u.y * (1.f - cosMaxAngle)) * (1.f - u.y * (1.f - cosMaxAngle)));
	const Vector3f wi = Vector3f(std::cos(2.f * Pi * u.x) * common, std::sin(2.f * Pi * u.x) * common, 1.f - u.y * (1.f - cosMaxAngle));
	if (pdf) {
		*pdf = 1.f / (2.f * Pi * (1.f - cosMaxAngle));
	}
	return LocalToWorld(wi);
}

float GuidingDistribution::Pdf(const Vector3f& wi, SamplingProjection projection) const {
	if (isTriangle) {
		if (projectedTriangle.CanSample(projection)) {
			return projectedTriangle.Pdf(wi, projection);
		}
		else if (projection == SamplingProjection::SpherePrecise && projectedTriangle.CanSample(SamplingProjection::SphereSimple)) {
			return projectedTriangle.Pdf(wi, SamplingProjection::SphereSimple);
		}
	}
	
	// If the triangle couln't be sampled, the sphere cap was sampled
	const float cosAngle = (Normalize(WorldToLocal(wi))).z;
	if (cosAngle >= cosMaxAngle) {
		return 1.f / (2.f * Pi * (1.f - cosMaxAngle));
	}
	else {
		return 0.f;
	}	
}

bool GuidingDistribution::CanSample(SamplingProjection projection) const {
	if (!isTriangle) {
		return false;
	} else {
		return projectedTriangle.CanSample(projection);
	}
}

}  // namespace pbrt
