#pragma once

#include <glad/glad.h> 

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "mesh.h"
#include "shader.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>

using namespace std;

unsigned int TextureFromFile(const char* path, const string& directory, bool gamma = false);

class Model
{
public:
    // model data 
    vector<Texture> textures_loaded;	// stores all the textures loaded so far, optimization to make sure textures aren't loaded more than once.
    vector<Mesh>    meshes;
    string directory;
    bool gammaCorrection;

    glm::vec3 translation;
    glm::vec4 rotation;
    float scale;

    // constructor, expects a filepath to a 3D model.
    Model(string const& path, bool gamma = false) : gammaCorrection(gamma)
    {
        loadModel(path);
    }

    // draws the model, and thus all its meshes
    void Draw(Shader* shader)
    {
        shader->setMat4("model", GetModelMat());
        for (unsigned int i = 0; i < meshes.size(); i++)
            meshes[i].Draw(shader);
    }

    glm::mat4 GetModelMat()
    {
        glm::mat4 Identity = glm::identity<glm::mat4>();

        glm::mat4 translateMat = glm::translate(Identity, translation);

        glm::mat4 rotationMat;
        glm::vec3 axis(rotation.x, rotation.y, rotation.z);
        float angle = rotation.w;
        if (glm::length(axis) < 1e-6)
        {
            rotationMat = Identity;
        }
        else
        {
            rotationMat = glm::rotate(Identity, angle, axis);
        }

        glm::mat4 scaleMat = glm::scale(Identity, glm::vec3(scale));

        return translateMat * rotationMat * scaleMat;
    }

private:
    // loads a model with supported ASSIMP extensions from file and stores the resulting meshes in the meshes vector.
    void loadModel(string const& path)
    {
        // read file via ASSIMP
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
        // check for errors
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
        {
            cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << endl;
            return;
        }
        // retrieve the directory path of the filepath
        directory = path.substr(0, path.find_last_of('\\'));

        // process ASSIMP's root node recursively
        processNode(scene->mRootNode, scene);
    }

    // processes a node in a recursive fashion. Processes each individual mesh located at the node and repeats this process on its children nodes (if any).
    void processNode(aiNode* node, const aiScene* scene)
    {
        // process each mesh located at the current node
        for (unsigned int i = 0; i < node->mNumMeshes; i++)
        {
            // the node object only contains indices to index the actual objects in the scene. 
            // the scene contains all the data, node is just to keep stuff organized (like relations between nodes).
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            meshes.push_back(processMesh(mesh, scene));
        }
        // after we've processed all of the meshes (if any) we then recursively process each of the children nodes
        for (unsigned int i = 0; i < node->mNumChildren; i++)
        {
            processNode(node->mChildren[i], scene);
        }

    }

    Mesh processMesh(aiMesh* mesh, const aiScene* scene)
    {
        // data to fill
        vector<Vertex> vertices;
        vector<unsigned int> indices;
        vector<Texture> textures;

        // walk through each of the mesh's vertices
        for (unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            Vertex vertex;
            glm::vec3 vector; // we declare a placeholder vector since assimp uses its own vector class that doesn't directly convert to glm's vec3 class so we transfer the data to this placeholder glm::vec3 first.
            // positions
            vector.x = mesh->mVertices[i].x;
            vector.y = mesh->mVertices[i].y;
            vector.z = mesh->mVertices[i].z;
            vertex.Position = vector;
            // normals
            if (mesh->HasNormals())
            {
                vector.x = mesh->mNormals[i].x;
                vector.y = mesh->mNormals[i].y;
                vector.z = mesh->mNormals[i].z;
                vertex.Normal = vector;
            }
            // texture coordinates
            if (mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
            {
                glm::vec2 vec;
                // a vertex can contain up to 8 different texture coordinates. We thus make the assumption that we won't 
                // use models where a vertex can have multiple texture coordinates so we always take the first set (0).
                vec.x = mesh->mTextureCoords[0][i].x;
                vec.y = mesh->mTextureCoords[0][i].y;
                vertex.TexCoords = vec;
                // tangent
                vector.x = mesh->mTangents[i].x;
                vector.y = mesh->mTangents[i].y;
                vector.z = mesh->mTangents[i].z;
                vertex.Tangent = vector;
                // bitangent
                vector.x = mesh->mBitangents[i].x;
                vector.y = mesh->mBitangents[i].y;
                vector.z = mesh->mBitangents[i].z;
                vertex.Bitangent = vector;
            }
            else
            {
                vertex.TexCoords = glm::vec2(0.0f, 0.0f);
                glm::vec3 v2,v3;

                if (std::abs(vertex.Normal.x) > std::abs(vertex.Normal.y))
                    v2 = glm::normalize(glm::vec3(-vertex.Normal.z, 0, vertex.Normal.x));
                else
                    v2 = glm::normalize(glm::vec3(0, vertex.Normal.z, -vertex.Normal.y));
                v3 = glm::cross(vertex.Normal, v2);
                vertex.Tangent = v2;
                vertex.Bitangent = v3;
            }

            aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
            aiColor3D basecolor(0.f, 0.f, 0.f);
            aiColor3D emissive(0.f, 0.f, 0.f);
            aiColor3D specular(0.04f, 0.04f, 0.04f);
            ai_real metallic = 0.0f;
            ai_real roughness = 0.0f;
            ai_real opacity = 1.0f;

            material->Get(AI_MATKEY_COLOR_DIFFUSE, basecolor);
            material->Get(AI_MATKEY_COLOR_EMISSIVE, emissive);
            material->Get(AI_MATKEY_COLOR_SPECULAR, specular);
            material->Get(AI_MATKEY_METALLIC_FACTOR, metallic);
            material->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
            material->Get(AI_MATKEY_OPACITY, opacity);

            //std::cout << "[" << basecolor.r << ", " << basecolor.g << ", " << basecolor.b << "]1" << std::endl;
            //std::cout << "[" << specular.r << ", " << specular.g  << ", " << specular.b << "]2" << std::endl;
            //std::cout << "OPACITY : " << opacity <<"  => Test result = " << (opacity < (1.f - 0.0001f)) << std::endl;

            vertex.mat.albedo.r = basecolor.r;
            vertex.mat.albedo.g = basecolor.g;
            vertex.mat.albedo.b = basecolor.b;

            vertex.mat.emittance.r = emissive.r;
            vertex.mat.emittance.g = emissive.g;
            vertex.mat.emittance.b = emissive.b;

            vertex.mat.specular.r = specular.r;
            vertex.mat.specular.g = specular.g;
            vertex.mat.specular.b = specular.b;

            vertex.mat.metallic = metallic; 
            vertex.mat.roughness = roughness;
            //vertex.mat.roughness = glm::clamp(roughness,5e-3f,0.999f);
            vertex.mat.opacity = opacity;

            vertices.push_back(vertex);
        }
        // now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
        for (unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            aiFace face = mesh->mFaces[i];
            // retrieve all indices of the face and store them in the indices vector
            if (face.mNumIndices % 3)
                continue;
            for (unsigned int j = 0; j < face.mNumIndices; j++)
                indices.push_back(face.mIndices[j]);
        }
        // process materials
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        // we assume a convention for sampler names in the shaders. Each diffuse texture should be named
        // as 'texture_diffuseN' where N is a sequential number ranging from 1 to MAX_SAMPLER_NUMBER. 
        // Same applies to other texture as the following list summarizes:
        // diffuse: texture_diffuseN
        // specular: texture_specularN
        // normal: texture_normalN
        // 1. diffuse maps
        // aiTextureType_DIFFUSE
        vector<Texture> basecolorMaps = loadMaterialTextures(material, aiTextureType_BASE_COLOR, "texture_basecolor");
        textures.insert(textures.end(), basecolorMaps.begin(), basecolorMaps.end());
        // 2. specular maps
        vector<Texture> metallicMaps = loadMaterialTextures(material, aiTextureType_METALNESS, "texture_metallic");
        textures.insert(textures.end(), metallicMaps.begin(), metallicMaps.end());
        // 3. normal maps
        std::vector<Texture> roughnessMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE_ROUGHNESS, "texture_roughness");
        textures.insert(textures.end(), roughnessMaps.begin(), roughnessMaps.end());
        // 4. height maps
        std::vector<Texture> normalMaps = loadMaterialTextures(material, aiTextureType_NORMALS, "texture_normal");
        textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());

        // return a mesh object created from the extracted mesh data
        return Mesh(vertices, indices, textures);
    }

    // checks all material textures of a given type and loads the textures if they're not loaded yet.
    // the required info is returned as a Texture struct.
    vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, string typeName)
    {
        vector<Texture> textures;
        for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
        {
            aiString str;
            mat->GetTexture(type, i, &str);
            // check if texture was loaded before and if so, continue to next iteration: skip loading a new texture
            bool skip = false;
            for (unsigned int j = 0; j < textures_loaded.size(); j++)
            {
                if (std::strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0)
                {
                    textures.push_back(textures_loaded[j]);
                    skip = true; // a texture with the same filepath has already been loaded, continue to next one. (optimization)
                    break;
                }
            }
            if (!skip)
            {   // if texture hasn't been loaded already, load it
                Texture texture;
                texture.SetID(TextureFromFile(str.C_Str(), this->directory));
                texture.type = typeName;
                texture.path = str.C_Str();
                textures.push_back(texture);
                textures_loaded.push_back(texture);  // store it as texture loaded for entire model, to ensure we won't unnecesery load duplicate textures.
                std::cout << "From material : " << mat->GetName().C_Str() << " => ";
                switch (type)
                {
                case aiTextureType_BASE_COLOR:
                    std::cout << "[aiTextureType_BASE_COLOR] ";
                    break;
                case aiTextureType_METALNESS:
                    std::cout << "[aiTextureType_METALNESS] ";
                    break;
                case aiTextureType_DIFFUSE_ROUGHNESS:
                    std::cout << "[aiTextureType_DIFFUSE_ROUGHNESS] ";
                    break;
                case aiTextureType_NORMALS:
                    std::cout << "[aiTextureType_NORMALS] ";
                    break;
                }
                std::cout << str.C_Str() << std::endl;
            }
        }
        return textures;
    }
};