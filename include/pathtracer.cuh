#pragma once

class PathTracer
{
public:
	void Render(class Camera& camera, class BVH* bvh);
};