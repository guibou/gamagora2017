#pragma once

#include <optional>

#include "vec.h"
#include "bbox.h"
#include "ray.h"

struct Sphere
{
	Position center;
	float radius;
};

std::ostream& operator<<(std::ostream &stream, const Sphere &s)
{
	stream << "Sphere{" << s.center << "," << s.radius << "}";
	return stream;
}

BBox bboxSphere(const Sphere &sphere)
{
	auto p = sphere.center.value;

	Vec radius = Vec{1, 1, 1} * sphere.radius;

	return {Position{p - radius},
			Position{p + radius}};
}

std::optional<float> intersectSphere(const Ray &ray, const Sphere &sphere)
{
	const Vec op = sphere.center.value - ray.origin.value;

	const float b = dot(op, ray.direction.value);
	const float det2 = b * b - dot(op, op) + sqr(sphere.radius);

	if (det2 < 0)
	{
		return std::nullopt;
	}
	else
	{
		const float det = std::sqrt(det2);

		const float t0 = b - det;
		const float t1 = b + det;

		if(t0 > 0)
		{
			return t0;
		}
		else if(t1 > 0)
			return t1;
		else
		{
			return std::nullopt;
		}
	}
}

float surface(const Sphere &s)
{
	return sqr(s.radius) * 4 * M_PI;
}

NormalizedDirection getNormal(const Sphere &s, const Position &p)
{
	const Vec direction = p.value - s.center.value;
	return NormalizedDirection{direction};
}
