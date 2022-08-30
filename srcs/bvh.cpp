#include "bvh.h"

#include <random>
#include <set>

#include <glad/glad.h>

#include "shader.h"
#include "model.h"
#include "mesh.h"

void BVH::Init()
{
	BoxShader = new Shader("C:\\Users\\kwonh\\Desktop\\study\\Graphics\\PathTrace_GPGPU\\resources\\BoxVertex.glsl", "C:\\Users\\kwonh\\Desktop\\study\\Graphics\\PathTrace_GPGPU\\resources\\BoxFrag.glsl");

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(BoxVertices), &BoxVertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void*)0);
	glBindVertexArray(0);

	rootBVH = nullptr;
	rootCluster = nullptr;
}

void BVH::draw(Shader* shader)
{

	glm::vec3 CenterPos(0.f);
	glm::vec3 Scale(10.f);

	glm::mat4 model = glm::translate(glm::identity<glm::mat4>(), CenterPos)
		* glm::scale(glm::identity<glm::mat4>(), Scale);
	BoxShader->setMat4("model", model);

	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, 36);
}

void BVH::drawCluster(Shader* shader, Cluster* cluster, int depth)
{
	if (cluster == nullptr)
	{
		return;
	}

	if (!cluster->bLeaf)
	{
		for (int i = 0; i < K; i++)
		{
			drawCluster(shader, cluster->Children[i], depth+1);
		}
		return;
	}

	shader->setVec3("ClusterColor", cluster->Color);

	glBindVertexArray(cluster->VAO);
	glDrawArrays(GL_TRIANGLES, 0, 3 * cluster->primitives.size());
}

void BVH::drawBVHTree(BVHNode* node)
{
	if (node == nullptr)
	{
		return;
	}

	if (node->Child[0] == nullptr && node->Child[1] == nullptr)
	{
		drawBVHNode(node);
		return;
	}

	drawBVHTree(node->Child[0]);
	drawBVHTree(node->Child[1]);
}

void BVH::drawBVHNode(BVHNode* node)
{
	glm::vec3 CenterPos = (node->bMin + node->bMax) * 0.5f;
	glm::vec3 Scale = (node->bMax - node->bMin);

	glm::mat4 model = glm::translate(glm::identity<glm::mat4>(), CenterPos)
                    * glm::scale(glm::identity<glm::mat4>(), Scale);

	BoxShader->setMat4("model", model);

	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, 36);
}

glm::vec3 BVH::GetCentroid(const Primitive& prim)
{
	return (prim.v1.Position + prim.v2.Position + prim.v3.Position) * 0.333333f;
}

void BVH::Cluster_K_MEAS_Recursive(Cluster* cluster, int depth)
{
	if (depth == 0 || cluster->primitives.size() < K)
	//if (cluster->primitives.size() < K)
	{
		cluster->bLeaf = true;
		return;
	}

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
	//float DiffSum;
	for (int i = 0; i < ITER; i++)
	{
		//DiffSum = 0.f;
		//glm::vec3 PrevCentroids[K];
		for (int i = 0; i < K; i++)
		{
			//PrevCentroids[i] = cluster->Children[i]->Centroid;
			cluster->Children[i]->primitives.clear();
		}

		Cluster_Assign_K(cluster);
		for (int i = 0; i < K; i++)
		{
			Cluster_Calc_Centroid(cluster->Children[i]);
		}
	}

	cluster->primitives.clear();
	cluster->bLeaf = false;
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
		std::set<int> candidates;
		for (int j = 0; j < min((unsigned long long)(2 * K), cluster->primitives.size() - i -j); j++)
		{
			do
			{
				rInt = uDist(gen);
			} while (DupHash.find(rInt) != DupHash.end() || candidates.find(rInt) != candidates.end());
			candidates.insert(rInt);
		}

		float minDist = std::numeric_limits<float>::max();
		int minDistClusterIdx = -1;

		for (auto idx : candidates)
		{
			float dist = 0.f;
			for (int j = 0; j <= i; j++)
			{
				dist += (cluster->Children[idx]->Centroid - cluster->Children[j]->Centroid).length();
			}

			if (dist < minDist)
			{
				minDist = dist;
				minDistClusterIdx = idx;
			}
		}

		candidates.clear();

		DupHash.insert(minDistClusterIdx);
		int primID = cluster->primitives[minDistClusterIdx];
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

	if (!cluster->bLeaf)
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

	cluster->Color.x = uDist(gen);
	cluster->Color.y = uDist(gen);
	cluster->Color.z = uDist(gen);
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

BVHNode* BVH::GenBVHTree(Cluster* cluster)
{
	if (cluster == nullptr)
	{
		std::cout << "NULLPTR WARNING" << std::endl;
		return nullptr;
		//exit(0);
	}
	std::vector<BVHNode*> nodes;
	for (int i = 0; i < K; i++)
	{
		if (cluster->Children[i] == nullptr)
		{
			continue;
		}
		else if (cluster->Children[i]->bLeaf)
		{
			if (cluster->Children[i]->primitives.size() > 0)
				nodes.push_back(IntoBVHNode(cluster->Children[i]));
			continue;
		}
		auto childTree = GenBVHTree(cluster->Children[i]);
		if (childTree)
			nodes.push_back(childTree);
	}
	while (nodes.size() > 1)
	{
		int NNN = nodes.size();
		BVHNode* minNode1 = nullptr;
		BVHNode* minNode2 = nullptr;
		float minDist = std::numeric_limits<float>::max();
		int _i = -1;
		int _j = -1;
		for (int i = 0; i < nodes.size(); i++)
		{
			for (int j = i + 1; j < nodes.size(); j++)
			{
				float dist = glm::distance(nodes[i]->Centroid, nodes[j]->Centroid);
				_i = i;
				_j = j;
				if (dist < minDist)
				{
					minDist = dist;
					minNode1 = nodes[i];
					minNode2 = nodes[j];
				}
			}
		}

		BVHNode* newNode = new BVHNode;
		newNode->bMin = glm::min(minNode1->bMin, minNode2->bMin);
		newNode->bMax = glm::max(minNode1->bMax, minNode2->bMax);
		newNode->primCnt = minNode1->primCnt + minNode2->primCnt;
		//newNode->Centroid = ((minNode1->Centroid * (float)(minNode1->primCnt)) + (minNode2->Centroid * (float)(minNode2->primCnt))) / (float)(newNode->primCnt);
		newNode->Centroid = (newNode->bMin + newNode->bMax) * 0.5f;
		newNode->Child[0] = minNode1; newNode->Child[1] = minNode2;
		auto iter = std::find(nodes.begin(), nodes.end(), minNode1);
		nodes.erase(iter);
		iter = std::find(nodes.begin(), nodes.end(), minNode2);
		nodes.erase(iter);
		nodes.push_back(newNode);
	}

	return nodes[0];
}

BVHNode* BVH::IntoBVHNode(Cluster* cluster)
{
	glm::vec3 bbMin = glm::vec3(std::numeric_limits<float>::max());
	glm::vec3 bbMax = glm::vec3(std::numeric_limits<float>::lowest());
	glm::vec3 Centroid(0.f);
	GLuint primCnt = cluster->primitives.size();

	for (auto primID : cluster->primitives)
	{
		Primitive& prim = primitives[primID];
		bbMin = glm::min(bbMin,glm::min(prim.v1.Position, glm::min(prim.v2.Position, prim.v3.Position)));
		bbMax = glm::max(bbMax,glm::max(prim.v1.Position, glm::max(prim.v2.Position, prim.v3.Position)));
		//Centroid += prim.v1.Position + prim.v2.Position + prim.v3.Position;
	}
	//Centroid /= (3.f * (float)primCnt);
	Centroid = (bbMax + bbMin) * 0.5f;
	BVHNode* node = new BVHNode();

	node->bMin = bbMin;
	node->bMax = bbMax;
	node->Centroid = Centroid;
	node->primCnt = primCnt;
	node->Child[0] = nullptr;
	node->Child[1] = nullptr;
	node->ptr_primitives = &(cluster->primitives);

	return node;
}

inline bool BVH::isLeafCluster(Cluster* cluster)
{
	return !cluster->primitives.empty();
}

BVHNode* SAHBVH::GenBVHTree(Cluster* cluster)
{
	int axis = 0;
	for (int i = 0; i < primitives.size(); i++)
		cluster->primitives.push_back(i);
	SplitNode(cluster, axis);

	rootCluster = cluster;
	rootBVH = ConvertToBVH(cluster);

	return rootBVH;
}

void SAHBVH::SplitNode(Cluster* cluster, int axis)
{
	if (cluster->primitives.size() <= 4)
	{
		cluster->bLeaf = true;
		return;
	}
	cluster->bLeaf = false;


	std::vector<Primitive>& prims = primitives;
	SAHBVH* _this = this;
	sort(cluster->primitives.begin(), cluster->primitives.end(),
		[&_this, &axis, &prims](const unsigned int A, const unsigned int B) {
			return  _this->GetCentroid(prims[A])[axis] > _this->GetCentroid(prims[B])[axis];
	});
	int primNum = cluster->primitives.size();
	GLfloat* CSA = new float[primNum];
	
	for (int i = 0; i < primNum; i++)
	{
		Primitive prim = prims[cluster->primitives[i]];
		float area = glm::length(glm::cross((prim.v2.Position - prim.v1.Position), (prim.v3.Position - prim.v1.Position)));
		if (i > 0)
			CSA[i] = CSA[i - 1] + area;
		else
			CSA[i] = area;
	}
	float minValue = std::numeric_limits<float>::max();
	int idx=0;
	for (int i = 1; i < primNum; i++)
	{
		float fi = CSA[i - 1] * i + (CSA[primNum - 1] - CSA[i - 1]) * (primNum - i);
		if (fi < minValue)
		{
			minValue = fi;
			idx = i;
		}
	}
	delete[] CSA;

	cluster->Children[0] = new Cluster();
	cluster->Children[1] = new Cluster();
	for (int i = 0; i < idx; i++)
	{
		cluster->Children[0]->primitives.push_back(cluster->primitives[i]);
	}
	for (int i = idx; i < primNum; i++)
	{
		cluster->Children[1]->primitives.push_back(cluster->primitives[i]);
	}
	cluster->primitives.clear();

	SplitNode(cluster->Children[0], (axis + 1) % 3);
	SplitNode(cluster->Children[1], (axis + 1) % 3);
}

BVHNode* SAHBVH::ConvertToBVH(Cluster* cluster)
{
	if (cluster->bLeaf)
		return IntoBVHNode(cluster);
	BVHNode* node = new BVHNode;

	node->Child[0] = ConvertToBVH(cluster->Children[0]);
	node->Child[1] = ConvertToBVH(cluster->Children[1]);

	node->bMin = glm::min(node->Child[0]->bMin, node->Child[1]->bMin);
	node->bMax = glm::max(node->Child[0]->bMax, node->Child[1]->bMax);
	node->Centroid = (node->bMin + node->bMax) * 0.5f;
	node->primCnt = node->Child[0]->primCnt + node->Child[1]->primCnt;

	return node;
}
