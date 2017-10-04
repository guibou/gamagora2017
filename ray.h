#pragma once

#include "vec.h"

struct Ray
{
	Position origin;
	NormalizedDirection direction;
};

Ray offsetRay(const Ray &ray, const NormalizedDirection &normal)
{
	return Ray{Position{ray.origin.value + 0.001 * normal.value}, ray.direction};
}

Position getIntersectionPosition(const Ray &ray, const float t)
{
	return Position{ray.origin.value + ray.direction.value * t};
}


