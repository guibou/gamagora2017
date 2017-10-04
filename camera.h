#pragma once

#include "ray.h"

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
