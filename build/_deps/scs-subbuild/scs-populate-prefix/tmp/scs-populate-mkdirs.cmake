# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/scs-src")
  file(MAKE_DIRECTORY "/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/scs-src")
endif()
file(MAKE_DIRECTORY
  "/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/scs-build"
  "/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/scs-subbuild/scs-populate-prefix"
  "/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/scs-subbuild/scs-populate-prefix/tmp"
  "/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/scs-subbuild/scs-populate-prefix/src/scs-populate-stamp"
  "/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/scs-subbuild/scs-populate-prefix/src"
  "/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/scs-subbuild/scs-populate-prefix/src/scs-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/scs-subbuild/scs-populate-prefix/src/scs-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/scs-subbuild/scs-populate-prefix/src/scs-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
