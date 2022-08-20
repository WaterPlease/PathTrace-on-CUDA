#pragma once

#include "mesh.h"

const int K = 8;
const int ITER = 10;

struct Primitive
{
	Vertex v1;
	Vertex v2;
	Vertex v3;
};

struct BVHNode
{
	glm::vec3 bMin;
	glm::vec3 bMax;
};

struct Cluster
{
	glm::vec3 Centroid;
	std::vector<unsigned int> primitives;
	Cluster* Children[K];

	GLuint VAO;
	GLuint VBO;
};

class BVH
{
public:
	std::vector<Primitive> primitives;
	Cluster* rootCluster;

	void draw(class Shader* shader);
	void drawCluster(class Shader* shader, Cluster* cluster, int depth);
	void GenClusterMesh(Cluster* cluster);

	void Cluster_K_MEAS_Recursive(Cluster* cluster, int depth);
	void Cluster_K_MEAS(Cluster* cluster);

	void AddModel(class Model* model);

private:
	inline void AddPrimitives(Primitive primitive) { primitives.push_back(primitive); }
	glm::vec3 GetCentroid(const Primitive& prim);

	void Cluster_Select_K(Cluster* cluster);
	void Cluster_Assign_K(Cluster* cluster);
	void Cluster_Calc_Centroid(Cluster* cluster);
};