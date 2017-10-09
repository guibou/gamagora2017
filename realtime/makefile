CXXFLAGS=-Wall -std=c++17 -Wextra -lGL -lglfw -g -Iglad/include/

test: test.cpp library.cpp glad/glad.c
	$(CXX) $(CXXFLAGS) test.cpp library.cpp glad/glad.c -o test
