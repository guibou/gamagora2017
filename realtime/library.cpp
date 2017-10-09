#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>

#include "glad/glad.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <GLFW/glfw3.h>

#define glInfo(a) std::cout << #a << ": " << glGetString(a) << std::endl

std::set<std::string> messages;

// This function is called on any openGL API error
void debug(GLenum, // source
		   GLenum, // type
		   GLuint, // id
		   GLenum, // severity
		   GLsizei, // length
		   const GLchar *message,
		   const void *) // userParam
{
	if(messages.count(std::string(message)) == 0)
	{
		messages.insert(std::string(message));
		std::cout << "DEBUG: " << message << std::endl;
	}
}

GLFWwindow* openContext()
{
	GLFWwindow* window;

	/* Initialize the library */
	if (!glfwInit())
	{
		std::cerr << "Could not init glfw" << std::endl;
		return nullptr;
	}

	// This is a debug context, this is slow, but debugs, which is interesting
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
	if (!window)
	{
		std::cerr << "Could not init window" << std::endl;
		glfwTerminate();
		return nullptr;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	if(!gladLoadGL()) {
		printf("Something went wrong!\n");
		exit(-1);
	}

	// Now that the context is initialised, print some informations
	glInfo(GL_VENDOR);
	glInfo(GL_RENDERER);
	glInfo(GL_VERSION);
	glInfo(GL_SHADING_LANGUAGE_VERSION);

	// And enable debug
	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

	glDebugMessageCallback(debug, nullptr);

	return window;
}

void runGL(void (*init)(),void (*render)(const int width, const int height))
{
	auto window = openContext();

	init();

	/* Loop until the user closes the window */
	while (!glfwWindowShouldClose(window))
	{
		/* Render here */
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		render(width, height);

		/* Swap front and back buffers */
		glfwSwapBuffers(window);

		/* Poll for and process events */
		glfwPollEvents();
	}

	glfwTerminate();
}

// Build a shader from a string
GLuint buildShader(GLenum const &shaderType, std::string const &src)
{
	GLuint shader = glCreateShader(shaderType);

	const char* ptr = src.c_str();
	GLint length = src.length();

	glShaderSource(shader, 1, &ptr, &length);

	glCompileShader(shader);

	GLint res;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &res);
	if(!res)
	{
		std::cerr << "shader compilation error" << std::endl;

		char message[1000];

		GLsizei readSize;
		glGetShaderInfoLog(shader, 1000, &readSize, message);
		message[999] = '\0';

		std::cerr << message << std::endl;

		glfwTerminate();
		exit(-1);
	}

	return shader;
}

// read a file content into a string
std::string fileGetContents(const std::string &path)
{
	std::ifstream t(path);

	assert(t.is_open());

	std::stringstream buffer;
	buffer << t.rdbuf();

	return buffer.str();
}

// build a program with a vertex shader and a fragment shader
GLuint buildProgram(const std::string &vertexFile, const std::string &fragmentFile)
{
	auto vshader = buildShader(GL_VERTEX_SHADER, fileGetContents(vertexFile));
	auto fshader = buildShader(GL_FRAGMENT_SHADER, fileGetContents(fragmentFile));

	GLuint program = glCreateProgram();

	glAttachShader(program, vshader);
	glAttachShader(program, fshader);

	glLinkProgram(program);

	GLint res;
	glGetProgramiv(program, GL_LINK_STATUS, &res);
	if(!res)
	{
		std::cerr << "program link error" << std::endl;

		char message[1000];

		GLsizei readSize;
		glGetProgramInfoLog(program, 1000, &readSize, message);
		message[999] = '\0';

		std::cerr << message << std::endl;

		glfwTerminate();
		exit(-1);
	}

	return program;
}

struct PN
{
	glm::vec3 p;
	glm::vec3 n;
};

struct Triangle
{
	PN pn[3];
};

std::vector<Triangle> readStl(const std::string &filename)
{
	std::vector<Triangle> tris;

	static_assert(sizeof(Triangle) == (3 * 3 * 4 * 2), "Triangle is not tighly packed");

	std::ifstream file(filename);
	assert(file.is_open());

	std::string dummy;

	// skip header
	std::getline(file, dummy);

	do
	{
		Triangle t;

		// facet
		file >> dummy;
		// normal
		file >> dummy;

		// skip normal
		file >> dummy;
		file >> dummy;
		file >> dummy;

		// skip outer loop
		file >> dummy;
		file >> dummy;

		// vertex
		for(int i = 0; i < 3; i++)
		{
			file >> dummy;
			for(int j = 0; j < 3; j++)
			{
				file >> t.pn[i].p[j];
			}
		}

		// skip endloop
		file >> dummy;
		// skip endfacet
		file >> dummy;

		if(file.good())
		{
			glm::vec3 normal = glm::normalize(glm::cross(t.pn[2].p - t.pn[0].p, t.pn[1].p - t.pn[0].p));
			for(int i = 0; i < 3; i++)
				t.pn[i].n = normal;

			tris.push_back(t);
		}
	} while(file.good());

	return tris;
}
