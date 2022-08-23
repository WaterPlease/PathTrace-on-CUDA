#pragma once

#include <vector>

#include <glad/glad.h> // holds all OpenGL type declarations

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "texture.h"

struct MaterialOnCPU
{
    glm::vec3 emittance;
    glm::vec3 albedo;
    float metallic;
    float roughness;
};

struct Vertex {
    // position
    glm::vec3 Position;
    // normal
    glm::vec3 Normal;
    // texCoords
    glm::vec2 TexCoords;
    // tangent
    glm::vec3 Tangent;
    // bitangent
    glm::vec3 Bitangent;

    MaterialOnCPU mat;

    float u;
    float v;
};

class Mesh
{
public:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;

    unsigned int VAO;

    Mesh(decltype(vertices) _vertices, decltype(indices) _indicies, decltype(textures) _textures) :vertices(_vertices), indices(_indicies), textures(_textures)
    {
        SetupMesh();
    }

    void Draw(class Shader* shader);
private:
    unsigned int VBO, EBO;

    void SetupMesh();
};