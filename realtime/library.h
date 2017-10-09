#include <vector>
#include <string>

#include "glad/glad.h"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

// read a file content into a string
std::string fileGetContents(const std::string &path);

// build a program with a vertex shader and a fragment shader
GLuint buildProgram(const std::string &vertexFile, const std::string &fragmentFile);

struct PN
{
	glm::vec3 p; // position
	glm::vec3 n; // normal
};

struct Triangle
{
	PN pn[3]; // 3 points make a triangl
};

// Read an stl file and return a list of triangles
std::vector<Triangle> readStl(const std::string &filename);

void runGL(void (*init)(), void (*render)(const int width, const int height));
