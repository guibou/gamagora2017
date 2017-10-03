#include <optional>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>

float sqr(const float v)
{
	return v * v;
}

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


struct Position
{
	Vec value;
};

struct Color
{
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

float norm2(const Vec &v)
{
	return sqr(v.x) + sqr(v.y) + sqr(v.z);
}

Vec normalize(const Vec &v)
{
	return v * (1.f / std::sqrt(norm2(v)));
}

struct NormalizedDirection
{
	explicit NormalizedDirection(const Vec &v): value(normalize(v))
	{
	}

	Vec value;
};

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

struct Ray
{
	Position origin;
	NormalizedDirection direction;
};

struct Sphere
{
	Position center;
	float radius;
};

struct Object
{
	Sphere sphere;
	Color color;
	bool isMirror;
};

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

int toInt(const float v)
{
	return int(v * 255);
}

struct Intersection
{
	float t;
	const Object *object;
};

std::ostream& operator<<(std::ostream &stream, const Intersection &i)
{
	stream << "Intersection{" << i.t << "," << i.object << "}";
	return stream;
}

std::ostream& operator<<(std::ostream &stream, const Sphere &s)
{
	stream << "Sphere{" << s.center << "," << s.radius << "}";
	return stream;
}

std::ostream& operator<<(std::ostream &stream, const Object &o)
{
	stream << "Object{" << o.sphere << "," << o.color << "," << o.isMirror << "}";
	return stream;
}

std::optional<Intersection> intersectScene(const Ray &ray, const std::vector<Object> &scene)
{
	std::optional<Intersection> result = std::nullopt;

	for(auto &object : scene)
	{
		auto it = intersectSphere(ray, object.sphere);

		if(it && (!result || *it < result->t))
		{
			result = Intersection{*it, &object};
		}
	}

	return result;
}

Position getIntersectionPosition(const Ray &ray, const float t)
{
	return Position{ray.origin.value + ray.direction.value * t};
}

NormalizedDirection getNormal(const Sphere &s, const Position &p)
{
	const Vec direction = p.value - s.center.value;
	return NormalizedDirection{direction};
}

NormalizedDirection getMirrorDirection(const NormalizedDirection &I, const NormalizedDirection &N)
{
	const Vec direction = {I.value - N.value * (dot(N.value, I.value) * 2)};
	return NormalizedDirection{direction};
}

Color radiance(const Ray &ray, const std::vector<Object> &scene, const int depth)
{
	if(depth == 10)
		return Color{Vec{0, 0, 0}};

	auto it = intersectScene(ray, scene);

	if(it)
	{
		if(it->object->isMirror)
		{
			Position origin = getIntersectionPosition(ray, it->t);
			NormalizedDirection normal = getNormal(it->object->sphere, origin);
			NormalizedDirection direction = getMirrorDirection(ray.direction, normal);
			const Ray newRay{origin, direction};
			return radiance(newRay, scene, depth + 1);
		}
		else
		{
			return it->object->color;
		}
	}
	else
	{
		return Color{Vec{0, 0, 0}};
	}
}

struct Camera
{
	int pixelSize;
	float sceneSize;

	float zPos;
	float opening;
};

float scaleCoordinate(const Camera &camera, const float v)
{
	return ((v / camera.pixelSize) - 0.5) * camera.sceneSize;
}

Ray sampleCamera(const Camera &camera, const int x, const int y)
{
	// we sample a point on the camera plane
	const Position posCameraA{Vec{scaleCoordinate(camera, x), scaleCoordinate(camera, y), camera.zPos}};

	// we sample a point on a bigger plane away from the camera
	const Position posCameraB {Vec{posCameraA.value.x * camera.opening, posCameraA.value.y * camera.opening, camera.zPos + 10}};

	return {posCameraA, NormalizedDirection{posCameraB.value - posCameraA.value}};
}

int main()
{
	// TESTS
	std::cout << sqr(2) << " should be " << 4 << std::endl;
	std::cout << NormalizedDirection{Vec{10, 0, 0}} << " should be " << "{1, 0, 0}" << std::endl;
	std::cout << NormalizedDirection{Vec{0, 10, 0}} << " should be " << "{0, 1, 0}" << std::endl;
	std::cout << NormalizedDirection{Vec{0, 0, 10}} << " should be " << "{0, 0, 1}" << std::endl;
	std::cout << NormalizedDirection{Vec{10, 10, 10}} << " should be " << "{0.57, 0.57, 0.57}" << std::endl;

	std::cout << Vec{2, 3, 5} - Vec{1, 8, 10} << " should be " << "{1, -5, -5}" << std::endl;

	std::cout << dot(Vec{2, 0, 0}, Vec{2, 0, 0}) << " should be " << 4 << std::endl;
	std::cout << dot(Vec{10, 0, 0}, Vec{0, 10, 0}) << " should be " << 0 << std::endl;
	std::cout << dot(Vec{2, 2, 3}, Vec{4, 5, 6}) << " should be " << 36 << std::endl;

	std::cout << intersectSphere(Ray{Position{Vec{2, 0, 0}}, NormalizedDirection{Vec{1, 0, 0}}},
								 Sphere{Position{Vec{10, 0, 0}}, 2}) << " should be " << 6 << std::endl;
	std::cout << intersectSphere(Ray{Position{Vec{0, 0, 0}}, NormalizedDirection{Vec{1, 0, 0}}},
								 Sphere{Position{Vec{0, 0, 10}}, 2}) << " should be " << "EMPTY" << std::endl;

	std::cout << intersectSphere(Ray{Position{Vec{15, 0, 0}}, NormalizedDirection{Vec{1, 0, 0}}},
								 Sphere{Position{Vec{10, 0, 0}}, 2}) << " should be " << "EMPTY" << std::endl;

	std::cout << intersectSphere(Ray{Position{Vec{10, 0, 0}}, NormalizedDirection{Vec{1, 1, 1}}},
								 Sphere{Position{Vec{10, 0, 0}}, 2}) << " should be " << 2 << std::endl;

	std::cout << intersectScene(Ray{Position{Vec{0, 0, 0}}, NormalizedDirection{Vec{1, 0, 0}}},
								std::vector<Object>{
									Object{Sphere{Position{Vec{10, 0, 0}}, 9}, Color{Vec{1, 0, 0}}, false},
									Object{Sphere{Position{Vec{4, 0, 0}}, 2}, Color{Vec{1, 1, 1}}, false}}) << " should be " << 1 << std::endl;

	const Camera camera{1024, 40, -10, 1.15}; // 1024x1024 pixels, with the screen between [-20 and 20]

	std::cout << scaleCoordinate(camera, 0) << " should be " << -20 << std::endl;
	std::cout << scaleCoordinate(camera, 1024) << " should be " << 20 << std::endl;
	std::cout << scaleCoordinate(camera, 512) << " should be " << 0 << std::endl;
	std::cout << toInt(0) << " should be " << 0 << std::endl;
	std::cout << toInt(1) << " should be " << 255 << std::endl;

	const int w = camera.pixelSize;
	const int h = camera.pixelSize;

	FILE * const f = fopen ("image.ppm", "w");
	fprintf (f, "P3\n%d %d\n%d\n", w, h, 255);

	float R = 100;
	std::vector<Object> scene{
		{{Position{Vec{0, 0, 0}}, 5}, Color{Vec{1, 1, 1}}, true}, // Center
		{{Position{Vec{15 + R, 0, 0}}, R}, Color{Vec{1, 0, 0}}, false}, // left
		{{Position{Vec{-15 - R, 0, 0}}, R}, Color{Vec{0, 0, 1}}, false}, // right
		{{Position{Vec{0, 15 + R, 0}}, R}, Color{Vec{0.5, 0.5, 0.5}}, false}, // top
		{{Position{Vec{0, - 15 - R, 0}}, R}, Color{Vec{0.5, 0.5, 0.5}}, false}, // bottom
		{{Position{Vec{0, 0, 15 + R}}, R}, Color{Vec{0.5, 0.5, 0.5}}, false} // back
	};

	for(unsigned y = 0; y < h; ++y)
	{
		for(unsigned x = 0; x < w; ++x)
		{
			const Ray ray = sampleCamera(camera, x, h - y);

			Color color = radiance(ray, scene, 0);

			fprintf (f, "%d %d %d ", toInt(color.value.x), toInt(color.value.y), toInt(color.value.z));
		}
	}

	fclose(f);
}
