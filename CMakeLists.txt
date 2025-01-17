cmake_minimum_required(VERSION 3.10)
project(GroupkCLIPPER)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include the gen_max_clique library
include_directories(/usr/local/include include)
link_directories(/usr/local/lib /usr/lib/x86_64-linux-gnu/)

# Add clipper to project
set(CLIPPER_DIR /home/kalliyanlay/Documents/BYU/research/clipper CACHE INTERNAL "CLIPPER build dir" FORCE)
set(BUILD_BINDINGS_MATLAB OFF CACHE BOOL "")
set(BUILD_TESTS OFF CACHE BOOL "")
set(ENABLE_MKL OFF CACHE BOOL "")
set(ENABLE_BLAS OFF CACHE BOOL "")
configure_file(cmake/clipper.cmake.in ${CLIPPER_DIR}/CMakeLists.txt IMMEDIATE @ONLY)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" . WORKING_DIRECTORY ${CLIPPER_DIR})
execute_process(COMMAND "${CMAKE_COMMAND}" --build . WORKING_DIRECTORY ${CLIPPER_DIR})
add_subdirectory(${CLIPPER_DIR}/src ${CLIPPER_DIR}/build)

# # Get openblas to work
# find_library(CBLAS_LIBRARY NAMES cblas PATHS /usr/local/lib)
# message(STATUS "CBLAS LIBRARIES:    " ${CBLAS_LIBRARY})
# add_definitions(-DHAVE_CBLAS=1)


# find_package(xtensor-blas REQUIRED)
# find_library(OPENBLAS_LIBRARY NAMES openblas PATHS /usr/local/lib)

# set(BLAS_VENDOR OpenBLAS)
# find_package(BLAS REQUIRED)

# message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
# message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})
# # message(STATUS "BLAS HEADERS: " ${BLAS_INCLUDE_DIRS})

# find_package(LAPACK REQUIRED)
# message(STATUS "LAPACK LIBRARIES: " ${LAPACK_LIBRARIES})

# Add the executable
add_executable(GroupkCLIPPER src/main.cpp src/GkCLIPPER.cpp)
add_executable(gkcm_brendon src/gkcm_test.cpp src/GkCLIPPER.cpp)

# Link the gen_max_clique library and openblas libraries
target_link_libraries(gkcm_brendon PRIVATE gen_max_clique pthread clipper)
# target_link_libraries(gkcm_brendon PRIVATE gen_max_clique pthread ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# Optional: Force xtensor-blas to use OpenBLAS explicitly
set(XTENSOR_BLAS_BACKEND OpenBLAS)