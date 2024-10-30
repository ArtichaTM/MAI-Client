cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MAIClient VERSION 0.1)

# Find SFML
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# target_include_directories(MAIClient PUBLIC "${PROJECT_BINARY_DIR}")

include_directories(src)
include_directories(static)

# Add the source files
set(SOURCES
    src/main.cpp
    src/elements/button.cpp
)

# Add the executable
add_executable(MAIClient ${SOURCES})

# Link SFML libraries
target_link_libraries(MAIClient sfml-graphics sfml-window sfml-system)