#pragma once

#include "vec.h"

template<typename T>
struct Sample
{
	T sample;
	float pdf;
};

// TotalCompendium, (34)
Sample<NormalizedDirection> sampleUniformSphere(const float u, const float v)
{
	const float pi2u = 2 * M_PI * u;
	const float sqrtv1minusv = std::sqrt(v * (1 - v));
	return {
		NormalizedDirection{Vec{2.f * std::cos(pi2u) * sqrtv1minusv, 2.f * std::sin(pi2u) * sqrtv1minusv, 1.f - 2 * v}},
			1 / (4.f * M_PI)
				};
}

// TotalCompendium, (34)
Sample<NormalizedDirection> sampleUniformHemisphere(const float u, const float v)
{
	const float pi2u = 2 * M_PI * u;
	const float sqrt1minusvv = std::sqrt(1 - sqr(v));
	return {
		NormalizedDirection{Vec{std::cos(pi2u) * sqrt1minusvv, std::sin(pi2u) * sqrt1minusvv, v}},
			1 / (2.f * M_PI)
				};
}

// Total compendium (35)
Sample<NormalizedDirection> sampleUniformHemisphereCos(const float u, const float v)
{
	const float pi2u = 2 * M_PI * u;
	const float sqrt1minusv = std::sqrt(1 - v);

	const float sqrtv = std::sqrt(v);
	return {
		NormalizedDirection{Vec{std::cos(pi2u) * sqrt1minusv, std::sin(pi2u) * sqrt1minusv, sqrtv}},
			float(sqrtv / M_PI)
				};
}

// Orthonormal base
// From
// http://jcgt.org/published/0006/01/01/
void branchlessONB(const Vec & n , Vec & b1 , Vec & b2 )
{
	float sign = std::copysign(1.0f, n.z);
	const float a = -1.0f / (sign + n.z);
	const float b = n.x * n.y * a;
	b1 = Vec{1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x};
	b2 = Vec{b, sign + n.y * n.y * a, -n.y};
}

NormalizedDirection RotateAroundBase(const NormalizedDirection &input, const NormalizedDirection &normal)
{
	Vec basex, basey;
	branchlessONB(normal.value, basex, basey);

	return NormalizedDirection{basex * input.value.x + basey * input.value.y + normal.value * input.value.z};
};

