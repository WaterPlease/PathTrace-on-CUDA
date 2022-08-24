#pragma once

#include "mesh.h"

const int K = 2;
const int ITER = 100;

struct Primitive
{
	Vertex v1;
	Vertex v2;
	Vertex v3;
};

struct BVHNode
{
	glm::vec3 Centroid;
	glm::vec3 bMin;
	glm::vec3 bMax;
	GLuint primCnt;

	BVHNode* Child[2];
	std::vector<unsigned int>* ptr_primitives = nullptr;
};

struct Cluster
{
	glm::vec3 Centroid;
	glm::vec3 Color;
	std::vector<unsigned int> primitives;
	Cluster* Children[K];

	GLuint VAO;
	GLuint VBO;

	bool bLeaf = false;
};

class BVH
{
public:
	std::vector<Primitive> primitives;
	Cluster* rootCluster;
	BVHNode* rootBVH;

	GLuint VAO;
	GLuint VBO;

	class Shader* BoxShader;

	BVH() = default;
	void Init();

	void draw(class Shader* shader);
	void drawCluster(class Shader* shader, Cluster* cluster, int depth);
	void drawBVHTree(BVHNode* node);
	void drawBVHNode(BVHNode* node);
	virtual BVHNode* GenBVHTree(Cluster* cluster);
	void GenClusterMesh(Cluster* cluster);

	void Cluster_K_MEAS_Recursive(Cluster* cluster, int depth);
	void Cluster_K_MEAS(Cluster* cluster);

	void AddModel(class Model* model);

protected:
	inline void AddPrimitives(Primitive primitive) { primitives.push_back(primitive); }
	glm::vec3 GetCentroid(const Primitive& prim);

	BVHNode* IntoBVHNode(Cluster* cluster);
private:
	void Cluster_Select_K(Cluster* cluster);
	void Cluster_Assign_K(Cluster* cluster);
	void Cluster_Calc_Centroid(Cluster* cluster);

	inline static bool isLeafCluster(Cluster* cluster);
};

static const float BoxVertices[] = {
		-0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
		 0.5f,  0.5f, -0.5f,
		 0.5f,  0.5f, -0.5f,
		-0.5f,  0.5f, -0.5f,
		-0.5f, -0.5f, -0.5f,

		-0.5f, -0.5f,  0.5f,
		 0.5f, -0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,
		-0.5f,  0.5f,  0.5f,
		-0.5f, -0.5f,  0.5f,

		-0.5f,  0.5f,  0.5f,
		-0.5f,  0.5f, -0.5f,
		-0.5f, -0.5f, -0.5f,
		-0.5f, -0.5f, -0.5f,
		-0.5f, -0.5f,  0.5f,
		-0.5f,  0.5f,  0.5f,

		 0.5f,  0.5f,  0.5f,
		 0.5f,  0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,

		-0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f,  0.5f,
		 0.5f, -0.5f,  0.5f,
		-0.5f, -0.5f,  0.5f,
		-0.5f, -0.5f, -0.5f,

		-0.5f,  0.5f, -0.5f,
		 0.5f,  0.5f, -0.5f,
		 0.5f,  0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,
		-0.5f,  0.5f,  0.5f,
		-0.5f,  0.5f, -0.5f,
};

class SAHBVH : public BVH
{
public:
	virtual BVHNode* GenBVHTree(Cluster* cluster) override;
private:
	int stopNumber = 4;
	void SplitNode(Cluster* cluster, int axis);
	BVHNode* ConvertToBVH(Cluster* cluster);
};