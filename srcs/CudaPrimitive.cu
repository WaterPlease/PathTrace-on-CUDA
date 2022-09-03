#include "CudaPrimitive.cuh"

std::vector<CudaBVHNode> CudaBVH;
std::vector<Triangle>    CudaPrims;
std::vector<Sphere>    CudaSpheres;


void LoadFromBVH(BVH* bvh)
{
    CudaBVH.clear();
    CudaPrims.clear();

    BVHNode* root = bvh->rootBVH;
    std::vector<BVHNode*> stack;
    stack.push_back(root);
    stack.push_back(root);
    stack.push_back((BVHNode*)0);
    stack.push_back((BVHNode*)0);

    int maxDepth = -1;

    while (!stack.empty())
    {
        int depth = (int)stack.back();
        stack.pop_back();
        int parentIdx = (int)stack.back();
        stack.pop_back();
        BVHNode* parent = stack.back();
        stack.pop_back();
        BVHNode* node = stack.back();
        stack.pop_back();

        int idx = CudaBVH.size();

        if (maxDepth < depth)
        {
            maxDepth = depth;
        }

        CudaBVHNode cudaNode;
        cudaNode.bMax = ConvertToCudaVec(node->bMax);
        cudaNode.bMin = ConvertToCudaVec(node->bMin);
        cudaNode.childL = cudaNode.childR = -1;
        cudaNode.primStart = -1;
        cudaNode.primEnd = -1;

        if (node->ptr_primitives != nullptr)
        {
            std::vector<unsigned int>& primInBB = *(node->ptr_primitives);
            cudaNode.primStart = CudaPrims.size();
            cudaNode.primEnd = CudaPrims.size() + primInBB.size() - 1;
            for (auto primID : primInBB)
            {
                Primitive prim = bvh->primitives[primID];
                Triangle tr;

                tr.V0 = ConvertToCudaVec(prim.v1.Position);
                tr.V1 = ConvertToCudaVec(prim.v2.Position);
                tr.V2 = ConvertToCudaVec(prim.v3.Position);

                tr.E1 = tr.V1 - tr.V0;
                tr.E2 = tr.V2 - tr.V0;

                tr.normal = Normalize(cross(tr.E1, tr.E2));

                tr.T0 = Normalize(ConvertToCudaVec(prim.v1.Tangent));
                tr.T1 = Normalize(ConvertToCudaVec(prim.v2.Tangent));
                tr.T2 = Normalize(ConvertToCudaVec(prim.v3.Tangent));

                tr.B0 = Normalize(ConvertToCudaVec(prim.v1.Bitangent));
                tr.B1 = Normalize(ConvertToCudaVec(prim.v2.Bitangent));
                tr.B2 = Normalize(ConvertToCudaVec(prim.v3.Bitangent));

                tr.N0 = Normalize(ConvertToCudaVec(prim.v1.Normal));
                tr.N1 = Normalize(ConvertToCudaVec(prim.v2.Normal));
                tr.N2 = Normalize(ConvertToCudaVec(prim.v3.Normal));

                tr.u0 = prim.v1.u;
                tr.v0 = prim.v1.v;

                tr.u1 = prim.v2.u;
                tr.v1 = prim.v2.v;

                tr.u2 = prim.v3.u;
                tr.v2 = prim.v3.v;

                tr.mat0.albedo = ConvertToCudaVec(prim.v1.mat.albedo);
                tr.mat0.emittance = ConvertToCudaVec(prim.v1.mat.emittance);
                tr.mat0.specular = ConvertToCudaVec(prim.v1.mat.specular);
                tr.mat0.roughness = prim.v1.mat.roughness;
                tr.mat0.metallic = prim.v1.mat.metallic;
                tr.mat0.opacity = prim.v1.mat.opacity;

                tr.mat1.albedo = ConvertToCudaVec(prim.v2.mat.albedo);
                tr.mat1.emittance = ConvertToCudaVec(prim.v2.mat.emittance);
                tr.mat1.specular = ConvertToCudaVec(prim.v2.mat.specular);
                tr.mat1.roughness = prim.v2.mat.roughness;
                tr.mat1.metallic = prim.v2.mat.metallic;
                tr.mat1.opacity = prim.v2.mat.opacity;

                tr.mat2.albedo = ConvertToCudaVec(prim.v3.mat.albedo);
                tr.mat2.emittance = ConvertToCudaVec(prim.v3.mat.emittance);
                tr.mat2.specular = ConvertToCudaVec(prim.v3.mat.specular);
                tr.mat2.roughness = prim.v3.mat.roughness;
                tr.mat2.metallic = prim.v3.mat.metallic;
                tr.mat2.opacity = prim.v3.mat.opacity;

                CudaPrims.push_back(tr);
            }
        }

        if (idx != parentIdx)
        {
            CudaBVHNode& pNode = CudaBVH[parentIdx];
            if (pNode.childL == -1)
            {
                pNode.childL = idx;
            }
            else if (pNode.childR == -1)
            {
                pNode.childR = idx;
            }
        }

        CudaBVH.push_back(cudaNode);


        if (node->Child[0] != nullptr)
        {
            stack.push_back(node->Child[0]);
            stack.push_back(node);
            stack.push_back((BVHNode*)idx);
            stack.push_back((BVHNode*)(depth + 1));
        }
        if (node->Child[1] != nullptr)
        {
            stack.push_back(node->Child[1]);
            stack.push_back(node);
            stack.push_back((BVHNode*)idx);
            stack.push_back((BVHNode*)(depth + 1));
        }
    }

    std::cout << "Maximum depth of tree : " << maxDepth << std::endl;
}