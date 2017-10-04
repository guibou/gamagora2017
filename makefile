CXXFLAGS=-Wall -Wextra -g -Ofast -std=c++17 -fopenmp
CC=c++

rt: rt.cpp *.h
	${CXX} ${CXXFLAGS} rt.cpp -o rt
