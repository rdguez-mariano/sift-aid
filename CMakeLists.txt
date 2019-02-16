project(DeepAsift)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)



FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

ADD_SUBDIRECTORY(cpp_libs/libNumerics)
ADD_SUBDIRECTORY(cpp_libs/libOrsa)

set(src
    libDA.cpp

    #ORSA
    cpp_libs/libOrsa/orsa_fundamental.cpp cpp_libs/libOrsa/conditioning.cpp
    cpp_libs/libOrsa/orsa_model.cpp cpp_libs/libOrsa/fundamental_model.cpp
    cpp_libs/libOrsa/homography_model.hpp cpp_libs/libOrsa/homography_model.cpp
    cpp_libs/libOrsa/orsa_homography.cpp cpp_libs/libOrsa/orsa_homography.hpp
)

include_directories(. ./cpp_libs/libOrsa)
add_library(DA MODULE ${src})
TARGET_LINK_LIBRARIES(DA Numerics)
