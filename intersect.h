#pragma once

#include <optional>
#include "sphere.h"
#include "ray.h"
#include "object.h"

struct Intersection
{
	Intersection(const float _t, const Object *_object)
	{
		t = _t;
		object = _object;
	}
	float t;
	const Object *object;
};

std::ostream& operator<<(std::ostream &stream, const Intersection &i)
{
	stream << "Intersection{" << i.t << "," << i.object << "}";
	return stream;
}
