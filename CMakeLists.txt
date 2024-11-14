cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# specify the C++ standard
# set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0120 NEW)

# Set the project name
project(MAIClient VERSION 0.1)

# Packages
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)
# find_package(Torch REQUIRED)

include_directories(src)

#
configure_file(src/config.hpp.in src/config.hpp @ONLY)

# Adding all .cpp-s
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Add the executable
add_executable(MAIClient ${SOURCES})

target_include_directories(MAIClient PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src)
target_include_directories(MAIClient PRIVATE src)

# Link SFML libraries
target_link_libraries(MAIClient sfml-graphics sfml-window sfml-system)
# target_link_libraries(MAIClient "${TORCH_LIBRARIES}")
