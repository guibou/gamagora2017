#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "glad/glad.h"

#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "library.h"

// Store the global state of your program
struct
{
	GLuint program; // a shader
	GLuint vao; // a vertex array object
	size_t nTris;
} gs;

const GLuint attribPosition = 0;

void init();

void init()
{
	// load the stl mesh
	auto tris = readStl("monkey.stl");
	gs.nTris = tris.size();
	std::cout << tris.size() << std::endl;

	// Build our program and an empty VAO
	gs.program = buildProgram("basic.vsl", "basic.fsl");

	glGenVertexArrays(1, &gs.vao);
	glBindVertexArray(gs.vao);

	GLuint buffer;
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);

	// fill the buffer
	int size = sizeof(Triangle) * tris.size();
	glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Triangle) * tris.size(), tris.data());

	// set the VAO
	/*
	  3: 3 floats
	  GL_FLOAT: c'est des floats
	  GL_FALSE: on ne veut pas les normaliser
	  3 * 4 * 2: c'est l'espace entre chaque nombre
	      3 float
		  3 sizeof(float)
		  2 (il y a les normals à passer)
	 */
	glVertexAttribPointer(attribPosition, 3, GL_FLOAT, GL_FALSE, 3 * 4 * 2, 0);
	glEnableVertexAttribArray(attribPosition);

	glBindVertexArray(0);
}

void render(const int width, const int height)
{
	glViewport(0, 0, width, height);

	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	glUseProgram(gs.program);
	glBindVertexArray(gs.vao);

	// double time = glfwGetTime();

	glDrawArrays(GL_TRIANGLES, 0, gs.nTris * 3);

	glBindVertexArray(0);
	glUseProgram(0);
	glDisable(GL_DEPTH_TEST);
}

int main(void)
{
	runGL(init, render);
}
