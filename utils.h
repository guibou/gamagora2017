#pragma once

#include <iostream>
#include <optional>
#include <cmath>

float sqr(const float v)
{
	return v * v;
}

float clamp(const float v)
{
	return std::min(1.f, std::max(0.f, v));
}

int toInt(const float v)
{
	return int(std::pow(clamp(v), 1.f / 2.2f) * 255);
}

template<typename T>
std::ostream& operator<<(std::ostream &stream, const std::optional<T> &o)
{
	stream << "Optional: ";
	if(o)
	{
		stream << "{" << *o << "}";
	}
	else
	{
		stream << "EMPTY";
	}

	return stream;
}
