#include <optional>
#include <variant>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

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

struct Diffuse
{};

float FDiffuse(const NormalizedDirection &n, const NormalizedDirection w)
{
	return dot(n.value, w.value) / M_PI;
}

struct Mirror
{};

using Material = std::variant<Diffuse, Mirror>;

struct Object
{
	Sphere sphere;
	Color albedo;

	Material material;
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

float clamp(const float v)
{
	return std::min(1.f, std::max(0.f, v));
}

int toInt(const float v)
{
	return int(std::pow(clamp(v), 1.f / 2.2f) * 255);
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

std::ostream& operator<<(std::ostream &stream, const Object &o)
{
	stream << "Object{" << o.sphere << "," << o.albedo << "," << o.material << "}";
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

using LightShape = std::variant<Sphere, Position>;

struct Light
{
	LightShape shape;
	Color intensity;
};

// Sampling
template<typename T>
struct Sample
{
	T sample;
	float pdf;
};

Ray offsetRay(const Ray &ray, const NormalizedDirection &normal)
{
	return Ray{Position{ray.origin.value + 0.001 * normal.value}, ray.direction};
}

// TotalCompendium, (34)
Sample<NormalizedDirection> sampleUniformSphere(const float u, const float v)
{
	const float pi2u = 2 * M_PI * u;
	const float sqrtv1minusv = std::sqrt(v * (1 - v));
	return {
		NormalizedDirection{Vec{2.f * std::cos(pi2u) * sqrtv1minusv, 2.f * std::sin(pi2u) * sqrtv1minusv, 1.f - 2 * v}},
			1 / (4.f * M_PI)
				};
}

// TotalCompendium, (34)
Sample<NormalizedDirection> sampleUniformHemisphere(const float u, const float v)
{
	const float pi2u = 2 * M_PI * u;
	const float sqrt1minusvv = std::sqrt(1 - sqr(v));
	return {
		NormalizedDirection{Vec{std::cos(pi2u) * sqrt1minusvv, std::sin(pi2u) * sqrt1minusvv, v}},
			1 / (2.f * M_PI)
				};
}

// Total compendium (35)
Sample<NormalizedDirection> sampleUniformHemisphereCos(const float u, const float v)
{
	const float pi2u = 2 * M_PI * u;
	const float sqrt1minusv = std::sqrt(1 - v);

	const float sqrtv = std::sqrt(v);
	return {
		NormalizedDirection{Vec{std::cos(pi2u) * sqrt1minusv, std::sin(pi2u) * sqrt1minusv, sqrtv}},
			float(sqrtv / M_PI)
				};
}

// Orthonormal base
// From
// http://jcgt.org/published/0006/01/01/
void branchlessONB(const Vec & n , Vec & b1 , Vec & b2 )
{
	float sign = std::copysign(1.0f, n.z);
	const float a = -1.0f / (sign + n.z);
	const float b = n.x * n.y * a;
	b1 = Vec{1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x};
	b2 = Vec{b, sign + n.y * n.y * a, -n.y};
}

NormalizedDirection RotateAroundBase(const NormalizedDirection &input, const NormalizedDirection &normal)
{
	Vec basex, basey;
	branchlessONB(normal.value, basex, basey);

	return NormalizedDirection{basex * input.value.x + basey * input.value.y + normal.value * input.value.z};
};

thread_local std::random_device r;

// Choose a random mean between 1 and 6
thread_local std::default_random_engine randomGenerator(r());
thread_local std::uniform_real_distribution<float> uniformRandom(0, 1);

struct SelectPointOnLight
{
	Sample<Position> operator()(const Position &p) const
	{
		return {p, 1.f};
	}

	Sample<Position> operator()(const Sphere &s) const
	{
		float u = uniformRandom(randomGenerator);
		float v = uniformRandom(randomGenerator);

		// select a point on an unit sphere
		const auto sample = sampleUniformSphere(u, v);

		// move it and scale it
		const auto p{s.center.value + s.radius * sample.sample.value};

		// pdf is modified by the light radius
		return {p, sample.pdf / sqr(s.radius)};
	}
};

float surface(const Sphere &s)
{
	return sqr(s.radius) * 4 * M_PI;
}

struct LightIllumination
{
	const Position &illuminationPoint;
	const NormalizedDirection &illuminationDirection;

	float operator()(const Position &/*p*/) const
	{
		return 1.f / (4 * M_PI);
	}

	float operator()(const Sphere &s) const
	{
		const NormalizedDirection nAtLightSurface = getNormal(s, illuminationPoint);

		/*
		  Note: on éclaire avec une lampe un peu bizarre, c'est une surface sphérique, mais qui n'existe pas dans la scene en tant que géometrie et donc ne fait pas d'ombrage.

		  ainsi, l'intégrale de la lampe, qui doit être égale à light.intensity est tel que :

		  =\int \Omega  \int _ S light.intensity * cos(\theta) dx
		  = light.intensity * Surface * 2 * pi

		  il faut donc normaliser par 1 / (surface * 2 * PI)

		  le 2 pi vient du fait que la surface éclaire des deux cotés, d'ou la valeur absolue dans le dot ui suit
		*/

		return std::abs(dot(nAtLightSurface.value, illuminationDirection.value)) / (surface(s) * 2 * M_PI);
	}
};

struct ComputeDirectLighting
{
	NormalizedDirection n;
	NormalizedDirection illuminationDirection;

	float operator()(const Diffuse &) const
	{
		return FDiffuse(n, illuminationDirection);
	}

	float operator()(const Mirror &) const
	{
		return 0.f;
	}
};

Color getLo(const Position &p, const NormalizedDirection &n, const std::vector<Light> &lights, const std::vector<Object> &scene, const Material &material)
{
	// stochastically select a light
	const float uLight = uniformRandom(randomGenerator);
	const int lightIdx = int(uLight * lights.size());
	const float pdfLight = 1.f / lights.size();

	const auto &light = lights[lightIdx];

	{
		const auto illuminationSample = std::visit(SelectPointOnLight(), light.shape);

		const Vec illuminationEdge = illuminationSample.sample.value - p.value;

		const float distanceSquared = norm2(illuminationEdge);

		const NormalizedDirection illuminationDirection(normalize(illuminationEdge));

		// evaluation de la fonction de surface
		const float f = std::visit(ComputeDirectLighting{n, illuminationDirection}, material);

		if(f > 0)
		{
			// check occlusion
			const Ray ray{p, illuminationDirection};

			const auto it = intersectScene(offsetRay(ray, n), scene);

			if(!it || sqr(it->t) > distanceSquared)
			{
				const float illuminationFactor = std::visit(LightIllumination{illuminationSample.sample, illuminationDirection}, light.shape);

				return light.intensity * (f * illuminationFactor / (distanceSquared * illuminationSample.pdf * pdfLight));
			}
		}
	}

	return {0.f, 0.f, 0.f};;
}

NormalizedDirection invert(const NormalizedDirection &n)
{
	return NormalizedDirection{n.value * -1};
}

struct SampleIndirect
{
	float contrib;
	NormalizedDirection direction;
};

struct IndirectSampling
{
	NormalizedDirection normal;
	NormalizedDirection rayDirection;

	SampleIndirect operator()(const Diffuse &) const
	{
		float u = uniformRandom(randomGenerator);
		float v = uniformRandom(randomGenerator);
		const auto sample = sampleUniformHemisphereCos(u, v);
		const auto direction = RotateAroundBase(sample.sample, normal);
		const float f = FDiffuse(normal, direction);

		return {f / sample.pdf, direction};
	}

	SampleIndirect operator()(const Mirror &) const
	{
		return {1.f, getMirrorDirection(rayDirection, normal)};
	}
};

Color radiance(const Ray &ray, const std::vector<Object> &scene, const std::vector<Light> &lights, const int depth)
{
	if(depth == 5)
		return Color{Vec{0, 0, 0}};

	auto it = intersectScene(ray, scene);

	if(it)
	{
		Position origin = getIntersectionPosition(ray, it->t);
		NormalizedDirection normalGeom = getNormal(it->object->sphere, origin);
		NormalizedDirection normal = dot(normalGeom.value, ray.direction.value) < 0 ? normalGeom : invert(normalGeom);

		Color indirectLighting{Vec{0, 0, 0}};

		const auto sampleIndirect = std::visit(IndirectSampling{normal, ray.direction}, it->object->material);

		if(sampleIndirect.contrib > 0.f)
		{
			const Ray newRay{origin, sampleIndirect.direction};
			const auto Li = radiance(offsetRay(newRay, normal), scene, lights, depth + 1);
			indirectLighting = it->object->albedo * sampleIndirect.contrib * Li;
		}

		const Color directLighting = it->object->albedo * getLo(origin, normal, lights, scene, it->object->material);

		return directLighting + indirectLighting;
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

Ray sampleCamera(const Camera &camera, const float x, const float y)
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
									Object{Sphere{Position{Vec{10, 0, 0}}, 9}, Color{Vec{1, 0, 0}}, Diffuse{}},
									Object{Sphere{Position{Vec{4, 0, 0}}, 2}, Color{Vec{1, 1, 1}}, Diffuse{}}}) << " should be " << 1 << std::endl;

	const Camera camera{1024, 40, -10, 1.17}; // 1024x1024 pixels, with the screen between [-20 and 20]

	std::cout << scaleCoordinate(camera, 0) << " should be " << -20 << std::endl;
	std::cout << scaleCoordinate(camera, 1024) << " should be " << 20 << std::endl;
	std::cout << scaleCoordinate(camera, 512) << " should be " << 0 << std::endl;
	std::cout << toInt(0) << " should be " << 0 << std::endl;
	std::cout << toInt(1) << " should be " << 255 << std::endl;

	const int w = camera.pixelSize;
	const int h = camera.pixelSize;

	float R = 1000;
	std::vector<Object> scene{
		{{Position{Vec{-8, 0, 0}}, 5}, Color{Vec{1, 1, 1}}, Mirror{}}, // Center Mirror
		{{Position{Vec{8, 0, 0}}, 5}, Color{Vec{1, 1, 1}}, Diffuse{}}, // Center Diffuse
		{{Position{Vec{15 + R, 0, 0}}, R}, Color{Vec{1, 0, 0}}, Diffuse{}}, // left
		{{Position{Vec{-15 - R, 0, 0}}, R}, Color{Vec{0, 0, 1}}, Diffuse{}}, // right
		{{Position{Vec{0, 15 + R, 0}}, R}, Color{Vec{0.5, 0.5, 0.5}}, Diffuse{}}, // top
		{{Position{Vec{0, - 15 - R, 0}}, R}, Color{Vec{0.5, 0.5, 0.5}}, Diffuse{}}, // bottom
		{{Position{Vec{0, 0, 15 + R}}, R}, Color{Vec{0.5, 0.5, 0.5}}, Diffuse{}} // back
	};

	std::vector<Light> lights{
		{LightShape{Sphere{Position{Vec{0, 10, 0}}, 2}}, Color{Vec{10000, 10000, 10000}}}
		// {LightShape{Position{Vec{0, 10, 0}}}, Color{Vec{10000, 10000, 10000}}}
	};

	const int nSamples = 64;

	std::vector<Color> output(w * h, Color{Vec{0, 0, 0}});
	
#pragma omp parallel for
	for(unsigned y = 0; y < h; ++y)
	{
		for(unsigned x = 0; x < w; ++x)
		{
			Color color{Vec{0, 0, 0}};

			for(uint sample = 0; sample < nSamples; sample++)
			{
				const float u = uniformRandom(randomGenerator);
				const float v = uniformRandom(randomGenerator);

				const Ray ray = sampleCamera(camera, x + u - 0.5f, h - y + v - 0.5f);

				color = color + radiance(ray, scene, lights, 0);
			}

			output[y * w + x] = color / nSamples;
		}
	}

	FILE * const f = fopen ("image.ppm", "w");
	fprintf (f, "P3\n%d %d\n%d\n", w, h, 255);

	for(const auto &color : output)
	{
		fprintf (f, "%d %d %d ", toInt(color.value.x), toInt(color.value.y), toInt(color.value.z));
	}

	fclose(f);
}
