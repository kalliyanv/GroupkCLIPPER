# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build

# Include any dependencies generated for this target.
include CMakeFiles/gkcm_brendon.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/gkcm_brendon.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gkcm_brendon.dir/flags.make

CMakeFiles/gkcm_brendon.dir/src/gkcm_test.cpp.o: CMakeFiles/gkcm_brendon.dir/flags.make
CMakeFiles/gkcm_brendon.dir/src/gkcm_test.cpp.o: ../src/gkcm_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/gkcm_brendon.dir/src/gkcm_test.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gkcm_brendon.dir/src/gkcm_test.cpp.o -c /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/src/gkcm_test.cpp

CMakeFiles/gkcm_brendon.dir/src/gkcm_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gkcm_brendon.dir/src/gkcm_test.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/src/gkcm_test.cpp > CMakeFiles/gkcm_brendon.dir/src/gkcm_test.cpp.i

CMakeFiles/gkcm_brendon.dir/src/gkcm_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gkcm_brendon.dir/src/gkcm_test.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/src/gkcm_test.cpp -o CMakeFiles/gkcm_brendon.dir/src/gkcm_test.cpp.s

CMakeFiles/gkcm_brendon.dir/src/GkCLIPPER.cpp.o: CMakeFiles/gkcm_brendon.dir/flags.make
CMakeFiles/gkcm_brendon.dir/src/GkCLIPPER.cpp.o: ../src/GkCLIPPER.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/gkcm_brendon.dir/src/GkCLIPPER.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gkcm_brendon.dir/src/GkCLIPPER.cpp.o -c /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/src/GkCLIPPER.cpp

CMakeFiles/gkcm_brendon.dir/src/GkCLIPPER.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gkcm_brendon.dir/src/GkCLIPPER.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/src/GkCLIPPER.cpp > CMakeFiles/gkcm_brendon.dir/src/GkCLIPPER.cpp.i

CMakeFiles/gkcm_brendon.dir/src/GkCLIPPER.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gkcm_brendon.dir/src/GkCLIPPER.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/src/GkCLIPPER.cpp -o CMakeFiles/gkcm_brendon.dir/src/GkCLIPPER.cpp.s

# Object files for target gkcm_brendon
gkcm_brendon_OBJECTS = \
"CMakeFiles/gkcm_brendon.dir/src/gkcm_test.cpp.o" \
"CMakeFiles/gkcm_brendon.dir/src/GkCLIPPER.cpp.o"

# External object files for target gkcm_brendon
gkcm_brendon_EXTERNAL_OBJECTS =

gkcm_brendon: CMakeFiles/gkcm_brendon.dir/src/gkcm_test.cpp.o
gkcm_brendon: CMakeFiles/gkcm_brendon.dir/src/GkCLIPPER.cpp.o
gkcm_brendon: CMakeFiles/gkcm_brendon.dir/build.make
gkcm_brendon: CMakeFiles/gkcm_brendon.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable gkcm_brendon"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gkcm_brendon.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gkcm_brendon.dir/build: gkcm_brendon

.PHONY : CMakeFiles/gkcm_brendon.dir/build

CMakeFiles/gkcm_brendon.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gkcm_brendon.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gkcm_brendon.dir/clean

CMakeFiles/gkcm_brendon.dir/depend:
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/CMakeFiles/gkcm_brendon.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gkcm_brendon.dir/depend

