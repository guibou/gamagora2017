#pragma once

#include "vec.h"
#include "sphere.h"
#include "material.h"

struct Object
{
	Sphere sphere;
	Color albedo;

	Material material;
};

std::ostream& operator<<(std::ostream &stream, const Object &o)
{
	stream << "Object{" << o.sphere << "," << o.albedo << "," << o.material << "}";
	return stream;
}
