#pragma once

#include <cmath>
#include <iostream>

#include "utils.h"

struct Vec
{
	float x, y, z;
};

Vec operator-(const Vec &a, const Vec &b)
{
	return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec operator+(const Vec &a, const Vec &b)
{
	return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec operator*(const Vec &a, const float f)
{
	return {a.x * f, a.y * f, a.z * f};
}

Vec operator*(const float f, const Vec &a)
{
	return {a.x * f, a.y * f, a.z * f};
}

Vec operator/(const Vec &a, const float f)
{
	return {a.x / f, a.y / f, a.z / f};
}

Vec operator/(const float f, const Vec &a)
{
	return {a.x / f, a.y / f, a.z / f};
}

float dot(const Vec &a, const Vec &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

float norm2(const Vec &v)
{
	return sqr(v.x) + sqr(v.y) + sqr(v.z);
}

Vec normalize(const Vec &v)
{
	return v * (1.f / std::sqrt(norm2(v)));
}

struct Position
{
	Vec value;
};

struct Color
{
	Vec value;
};

struct NormalizedDirection
{
	explicit NormalizedDirection(const Vec &v): value(normalize(v))
	{
	}

	Vec value;
};

Color operator-(const Color &a, const Color &b)
{
	return {a.value.x - b.value.x, a.value.y - b.value.y, a.value.z - b.value.z};
}

Color operator+(const Color &a, const Color &b)
{
	return {a.value.x + b.value.x, a.value.y + b.value.y, a.value.z + b.value.z};
}

Color operator*(const Color &a, const Color &b)
{
	return {a.value.x * b.value.x, a.value.y * b.value.y, a.value.z * b.value.z};
}

Color operator*(const Color &a, const float f)
{
	return {a.value.x * f, a.value.y * f, a.value.z * f};
}

Color operator*(const float f, const Color &a)
{
	return {a.value.x * f, a.value.y * f, a.value.z * f};
}

Color operator/(const Color &a, const float f)
{
	return {a.value.x / f, a.value.y / f, a.value.z / f};
}

Color operator/(const float f, const Color &a)
{
	return {a.value.x / f, a.value.y / f, a.value.z / f};
}

NormalizedDirection invert(const NormalizedDirection &n)
{
	return NormalizedDirection{n.value * -1};
}

std::ostream& operator<<(std::ostream &stream, const Vec &d)
{
	stream << "Vec{" << d.x << "," << d.y << "," << d.z << "}";
	return stream;
}

std::ostream& operator<<(std::ostream &stream, const NormalizedDirection &d)
{
	stream << "NormalizedDirection{" << d.value << "}";
	return stream;
}

std::ostream& operator<<(std::ostream &stream, const Position &d)
{
	stream << "Position{" << d.value << "}";
	return stream;
}

std::ostream& operator<<(std::ostream &stream, const Color &d)
{
	stream << "Color{" << d.value << "}";
	return stream;
}

