#include <optional>
#include <variant>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

#include "vec.h"
#include "utils.h"
#include "material.h"
#include "bbox.h"
#include "ray.h"
#include "camera.h"
#include "sampling.h"
#include "sphere.h"
#include "object.h"
#include "light.h"
#include "intersect.h"
#include "sceneTree.h"

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

Color getLo(const Position &p, const NormalizedDirection &n, const Scene &scene, const Material &material)
{
	// stochastically select a light
	const float uLight = uniformRandom(randomGenerator);
	const int lightIdx = int(uLight * scene.lights.size());
	const float pdfLight = 1.f / scene.lights.size();

	const auto &light = scene.lights[lightIdx];

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

			const auto it = intersectScene(offsetRay(ray, n), *scene.tree);

			if(!it || sqr(it->t) > distanceSquared)
			{
				const float illuminationFactor = std::visit(LightIllumination{illuminationSample.sample, illuminationDirection}, light.shape);

				return light.intensity * (f * illuminationFactor / (distanceSquared * illuminationSample.pdf * pdfLight));
			}
		}
	}

	return {0.f, 0.f, 0.f};
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

Color radiance(const Ray &ray, const Scene &scene, const int depth)
{
	if(depth == 5)
		return Color{Vec{0, 0, 0}};

	auto it = intersectScene(ray, *scene.tree);

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
			const auto Li = radiance(offsetRay(newRay, normal), scene, depth + 1);
			indirectLighting = it->object->albedo * sampleIndirect.contrib * Li;
		}

		const Color directLighting = it->object->albedo * getLo(origin, normal, scene, it->object->material);

		return directLighting + indirectLighting;
	}
	else
	{
		return Color{Vec{0, 0, 0}};
	}
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


	const Camera camera{1024, 40, -10, 1.17}; // 1024x1024 pixels, with the screen between [-20 and 20]

	std::cout << scaleCoordinate(camera, 0) << " should be " << -20 << std::endl;
	std::cout << scaleCoordinate(camera, 1024) << " should be " << 20 << std::endl;
	std::cout << scaleCoordinate(camera, 512) << " should be " << 0 << std::endl;
	std::cout << toInt(0) << " should be " << 0 << std::endl;
	std::cout << toInt(1) << " should be " << 255 << std::endl;

	const int w = camera.pixelSize;
	const int h = camera.pixelSize;

	float R = 1000;

	std::vector<Object> spheres
	{
		{{Position{Vec{-8, 0, 0}}, 5}, Color{Vec{1, 1, 1}}, Mirror{}}, // Center Mirror
		{{Position{Vec{8, 0, 0}}, 5}, Color{Vec{1, 1, 1}}, Diffuse{}}, // Center Diffuse
		{{Position{Vec{15 + R, 0, 0}}, R}, Color{Vec{1, 0, 0}}, Diffuse{}}, // left
		{{Position{Vec{-15 - R, 0, 0}}, R}, Color{Vec{0, 0, 1}}, Diffuse{}}, // right
		{{Position{Vec{0, 15 + R, 0}}, R}, Color{Vec{0.5, 0.5, 0.5}}, Diffuse{}}, // top
		{{Position{Vec{0, - 15 - R, 0}}, R}, Color{Vec{0.5, 0.5, 0.5}}, Diffuse{}}, // bottom
		{{Position{Vec{0, 0, 15 + R}}, R}, Color{Vec{0.5, 0.5, 0.5}}, Diffuse{}} // back
	};

	// This is a big scene description, comment it out
	/*
	R = 1;
	for(unsigned int i = 0; i < 10000; i++)
	{
		float x = uniformRandom(randomGenerator) - 0.5;
		float y = uniformRandom(randomGenerator) - 0.5;
		float z = uniformRandom(randomGenerator)- 0.5;

		spheres.push_back({{Position{Vec{x, y, z} * 20 }, 0.1}, Color{Vec{1, 1, 1}}, Diffuse{}}); // Center Mirror
	}
	*/

	const Scene scene{spheres,
		{
			{LightShape{Sphere{Position{Vec{0, 10, 0}}, 2}}, Color{Vec{10000, 10000, 10000}}}
			// {LightShape{Position{Vec{0, 10, 0}}}, Color{Vec{10000, 10000, 10000}}}
		}
	};

	const int nSamples = 1;

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

				color = color + radiance(ray, scene, 0);
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
