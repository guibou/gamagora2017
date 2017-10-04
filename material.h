#pragma once

#include <variant>

#include "vec.h"

struct Diffuse
{};

float FDiffuse(const NormalizedDirection &n, const NormalizedDirection w)
{
	return dot(n.value, w.value) / M_PI;
}

struct Mirror
{};

NormalizedDirection getMirrorDirection(const NormalizedDirection &I, const NormalizedDirection &N)
{
	const Vec direction = {I.value - N.value * (dot(N.value, I.value) * 2)};
	return NormalizedDirection{direction};
}


using Material = std::variant<Diffuse, Mirror>;

struct MaterialStream
{
	std::ostream& stream;

	std::ostream& operator()(const Diffuse &) const
	{
		stream << "Diffuse{}";
		return stream;
	}

	std::ostream& operator()(const Mirror &) const
	{
		stream << "Mirror{}";
		return stream;
	}
};

std::ostream& operator<<(std::ostream &stream, const Material &m)
{
	stream << "Material{";
	std::visit(MaterialStream{stream}, m);
	stream << "}";
	return stream;
}
