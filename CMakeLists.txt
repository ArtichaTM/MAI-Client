cmake_minimum_required(VERSION 3.10)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the project name
project(MAIClient VERSION 0.1)

# Find SFML
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

include_directories(src)

configure_file(src/config.hpp.in src/config.hpp @ONLY)

file(GLOB_RECURSE SOURCES "src/*.cpp")

# Add the source files
# set(SOURCES
#     src/main.cpp
#     src/elements/button.cpp
#     src/elements/tab_system.cpp
# )

# Add the executable
add_executable(MAIClient ${SOURCES})

target_include_directories(MAIClient PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src)
target_include_directories(MAIClient PRIVATE src)

# Link SFML libraries
target_link_libraries(MAIClient sfml-graphics sfml-window sfml-system)