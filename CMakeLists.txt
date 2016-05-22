cmake_minimum_required(VERSION 2.8)

set(SRC terrainrecovery.cpp
	rasterize.cpp
	rotater.cpp
	implementations.cpp
)

find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})

endif()
set(CMAKE_CXX_FLAGS "-std=c++11")
add_executable(terrainrecovery ${SRC})
	
