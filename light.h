#pragma once

#include <variant>

#include "vec.h"
#include "sphere.h"

using LightShape = std::variant<Sphere, Position>;

struct Light
{
	LightShape shape;
	Color intensity;
};
