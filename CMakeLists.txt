cmake_minimum_required(VERSION 3.10)
project(GroupkCLIPPER)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include the gen_max_clique library
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

# GkCLIPPER include directories
include_directories(include)

# Add the executable
add_executable(GroupkCLIPPER src/main.cpp src/GkCLIPPER.cpp)
add_executable(gkcm_brendon src/gkcm_test.cpp src/GkCLIPPER.cpp)

# Link the gen_max_clique library
target_link_libraries(gkcm_brendon PRIVATE gen_max_clique pthread)
