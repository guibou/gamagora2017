#pragma once

#include "vec.h"
#include <limits>
#include <optional>
#include "ray.h"

struct BBox
{
	Position pMin;
	Position pMax;
};

bool pointInBox(const Position &p, const BBox &box)
{
	return
		   box.pMin.value.x <= p.value.x
		&& box.pMin.value.y <= p.value.y
		&& box.pMin.value.z <= p.value.z
		&& box.pMax.value.x >= p.value.x
		&& box.pMax.value.y >= p.value.y
		&& box.pMax.value.z >= p.value.z;
}

BBox bboxUnion(const BBox &boxA, const BBox &boxB)
{
	return {Vec{
			std::min(boxA.pMin.value.x, boxB.pMin.value.x),
			std::min(boxA.pMin.value.y, boxB.pMin.value.y),
			std::min(boxA.pMin.value.z, boxB.pMin.value.z)
				},
			Vec{
			std::max(boxA.pMax.value.x, boxB.pMax.value.x),
			std::max(boxA.pMax.value.y, boxB.pMax.value.y),
			std::max(boxA.pMax.value.z, boxB.pMax.value.z)
				}};
};

Vec bboxSize(const BBox &box)
{
	return box.pMax.value - box.pMin.value;
}

BBox empty()
{
	const float inf = std::numeric_limits<float>::infinity();
	return BBox{Vec{inf, inf, inf}, Vec{-inf, -inf, -inf}};
}

std::optional<float> intersectBBox(const Ray &ray, const BBox &box) {
	const float tx1 = (box.pMin.value.x - ray.origin.value.x)/ray.direction.value.x;
	const float tx2 = (box.pMax.value.x - ray.origin.value.x)/ray.direction.value.x;

	float tmin = std::min(tx1, tx2);
	float tmax = std::max(tx1, tx2);

	const float ty1 = (box.pMin.value.y - ray.origin.value.y)/ray.direction.value.y;
	const float ty2 = (box.pMax.value.y - ray.origin.value.y)/ray.direction.value.y;

	tmin = std::max(tmin, std::min(ty1, ty2));
	tmax = std::min(tmax, std::max(ty1, ty2));

	if(tmax >= tmin)
	{
		return tmin;
	}
	return std::nullopt;
}
