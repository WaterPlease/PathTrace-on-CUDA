#include "bvh.h"

#include <random>
#include <set>

#include <glad/glad.h>

#include "shader.h"
#include "model.h"
#include "mesh.h"

void BVH::draw(Shader* shader)
{
	
}

void BVH::drawCluster(Shader* shader, Cluster* cluster, int depth)
{
	if (cluster == nullptr)
	{
		return;
	}

	if (cluster->primitives.empty())
	{
		for (int i = 0; i < K; i++)
		{
			drawCluster(shader, cluster->Children[i], depth+1);
		}
		return;
	}

	shader->setVec3("ClusterColor", cluster->Centroid);

	glBindVertexArray(cluster->VAO);
	glDrawArrays(GL_TRIANGLES, 0, 3 * cluster->primitives.size());
}

glm::vec3 BVH::GetCentroid(const Primitive& prim)
{
	return (prim.v1.Position + prim.v2.Position + prim.v3.Position) * 0.333333f;
}

void BVH::Cluster_K_MEAS_Recursive(Cluster* cluster, int depth)
{
	if (depth == 0 || cluster->primitives.size() < K) return;

	depth -= 1;

	Cluster_K_MEAS(cluster);
	for (int i = 0; i < K; i++)
	{
		Cluster_K_MEAS_Recursive(cluster->Children[i], depth);
	}
}

void BVH::Cluster_K_MEAS(Cluster* cluster)
{
	for (int i = 0; i < K; i++)
	{
		cluster->Children[i] = new Cluster();
	}

	Cluster_Select_K(cluster);
	float DiffSum;
	for (int i = 0; i < ITER; i++)
	{
		DiffSum = 0.f;
		glm::vec3 PrevCentroids[K];
		for (int i = 0; i < K; i++)
		{
			PrevCentroids[i] = cluster->Children[i]->Centroid;
			cluster->Children[i]->primitives.clear();
		}

		Cluster_Assign_K(cluster);
		for (int i = 0; i < K; i++)
		{
			Cluster_Calc_Centroid(cluster->Children[i]);
		}
	}
	/*
	do
	{
		DiffSum = 0.f;
		glm::vec3 PrevCentroids[K];
		for (int i = 0; i < K; i++)
		{
			PrevCentroids[i] = cluster->Children[i]->Centroid;
			cluster->Children[i]->primitives.clear();
		}

		Cluster_Assign_K(cluster);
		for (int i = 0; i < K; i++)
		{
			Cluster_Calc_Centroid(cluster->Children[i]);
		}

		for (int i = 0; i < K; i++)
		{
			DiffSum += glm::distance(PrevCentroids[i],cluster->Children[i]->Centroid);
		}
	} while (DiffSum > (K * std::numeric_limits<float>::epsilon()));
	*/

	cluster->primitives.clear();
}

void BVH::AddModel(Model* model)
{
	auto modelMat = model->GetModelMat();
	for (auto mesh : model->meshes)
	{
		if (mesh.indices.size() % 3 != 0)
		{
			std::cout << "WTF??? : " << mesh.indices.size();
			exit(0);
		}
		for (int i = 0; i < mesh.indices.size(); i += 3)
		{
			Primitive prim;
			prim.v1 = mesh.vertices[mesh.indices[i]];
			prim.v2 = mesh.vertices[mesh.indices[i+1]];
			prim.v3 = mesh.vertices[mesh.indices[i+2]];

			prim.v1.Position = modelMat * glm::vec4(prim.v1.Position, 1.0);
			prim.v2.Position = modelMat * glm::vec4(prim.v2.Position, 1.0);
			prim.v3.Position = modelMat * glm::vec4(prim.v3.Position, 1.0);

			prim.v1.Normal = modelMat * glm::vec4(prim.v1.Normal, 0.0);
			prim.v2.Normal = modelMat * glm::vec4(prim.v2.Normal, 0.0);
			prim.v3.Normal = modelMat * glm::vec4(prim.v3.Normal, 0.0);

			prim.v1.Tangent = modelMat * glm::vec4(prim.v1.Tangent, 0.0);
			prim.v2.Tangent = modelMat * glm::vec4(prim.v2.Tangent, 0.0);
			prim.v3.Tangent = modelMat * glm::vec4(prim.v3.Tangent, 0.0);

			prim.v1.Bitangent = modelMat * glm::vec4(prim.v1.Bitangent, 0.0);
			prim.v2.Bitangent = modelMat * glm::vec4(prim.v2.Bitangent, 0.0);
			prim.v3.Bitangent = modelMat * glm::vec4(prim.v3.Bitangent, 0.0);

			AddPrimitives(prim);
		}
	}
}

void BVH::Cluster_Select_K(Cluster* cluster)
{
	std::random_device rd;
	std::mt19937 gen(rd()); // Create a mersenne twister, seeded using the random device

	// Create a reusable random number generator that generates uniform numbers between 1 and 6
	std::uniform_int_distribution<> uDist(0, cluster->primitives.size() - 1);
	std::set<int> DupHash;

	for (int i = 0; i < K; i++)
	{
		int rInt;
		do
		{
			rInt = uDist(gen);
		} while (DupHash.find(rInt) != DupHash.end());
		DupHash.insert(rInt);
		int primID = cluster->primitives[rInt];
		cluster->Children[i]->Centroid = GetCentroid(primitives[primID]);
	}
}

void BVH::Cluster_Assign_K(Cluster* cluster)
{
	for (auto primID : cluster->primitives)
	{
		float minDist = std::numeric_limits<float>::max();
		int minK = -1;
		glm::vec3 centroid = GetCentroid(primitives[primID]);
		for (int i = 0; i < K; i++)
		{
			float dist = glm::distance(centroid, cluster->Children[i]->Centroid);
			if (dist < minDist)
			{
				minDist = dist;
				minK = i;
			}
		}

		cluster->Children[minK]->primitives.push_back(primID);
	}
}

void BVH::Cluster_Calc_Centroid(Cluster* cluster)
{
	glm::vec3 Centroid(0.f);
	for (auto primID : cluster->primitives)
	{
		Centroid += GetCentroid(primitives[primID]);
	}
	Centroid /= (float)(cluster->primitives.size());
}

void BVH::GenClusterMesh(Cluster* cluster)
{
	if (cluster == nullptr)
	{
		std::cout << "CP nullptr" << std::endl;
		return;
	}

	if (cluster->primitives.empty())
	{
		for (int i = 0; i < K; i++)
		{
			GenClusterMesh(cluster->Children[i]);
		}
		return;
	}

	std::random_device rd;
	std::mt19937 gen(rd()); // Create a mersenne twister, seeded using the random device

	// Create a reusable random number generator that generates uniform numbers between 1 and 6
	std::uniform_real_distribution<> uDist(0.0,1.0);

	cluster->Centroid.x = uDist(gen);
	cluster->Centroid.y = uDist(gen);
	cluster->Centroid.z = uDist(gen);
	glGenVertexArrays(1, &cluster->VAO);
	glGenBuffers(1, &cluster->VBO);

	glBindVertexArray(cluster->VAO);

	glBindBuffer(GL_ARRAY_BUFFER, cluster->VBO);

	int N = cluster->primitives.size();

	glBufferData(GL_ARRAY_BUFFER, N * sizeof(Primitive), NULL, GL_STATIC_DRAW);
	for (int i = 0; i < N; i++)
	{
		Primitive prim = primitives[cluster->primitives[i]];
		glBufferSubData(GL_ARRAY_BUFFER, i * sizeof(Primitive) , sizeof(Primitive), (void*) & prim);
	}

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Bitangent));

	glBindVertexArray(0);
}
