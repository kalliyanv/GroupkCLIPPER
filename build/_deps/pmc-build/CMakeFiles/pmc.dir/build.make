# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build

# Include any dependencies generated for this target.
include _deps/pmc-build/CMakeFiles/pmc.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/pmc-build/CMakeFiles/pmc.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/pmc-build/CMakeFiles/pmc.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/pmc-build/CMakeFiles/pmc.dir/flags.make

_deps/pmc-build/CMakeFiles/pmc.dir/codegen:
.PHONY : _deps/pmc-build/CMakeFiles/pmc.dir/codegen

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_heu.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/flags.make
_deps/pmc-build/CMakeFiles/pmc.dir/pmc_heu.cpp.o: _deps/pmc-src/pmc_heu.cpp
_deps/pmc-build/CMakeFiles/pmc.dir/pmc_heu.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/pmc-build/CMakeFiles/pmc.dir/pmc_heu.cpp.o"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/pmc-build/CMakeFiles/pmc.dir/pmc_heu.cpp.o -MF CMakeFiles/pmc.dir/pmc_heu.cpp.o.d -o CMakeFiles/pmc.dir/pmc_heu.cpp.o -c /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_heu.cpp

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_heu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pmc.dir/pmc_heu.cpp.i"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_heu.cpp > CMakeFiles/pmc.dir/pmc_heu.cpp.i

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_heu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pmc.dir/pmc_heu.cpp.s"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_heu.cpp -o CMakeFiles/pmc.dir/pmc_heu.cpp.s

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_maxclique.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/flags.make
_deps/pmc-build/CMakeFiles/pmc.dir/pmc_maxclique.cpp.o: _deps/pmc-src/pmc_maxclique.cpp
_deps/pmc-build/CMakeFiles/pmc.dir/pmc_maxclique.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object _deps/pmc-build/CMakeFiles/pmc.dir/pmc_maxclique.cpp.o"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/pmc-build/CMakeFiles/pmc.dir/pmc_maxclique.cpp.o -MF CMakeFiles/pmc.dir/pmc_maxclique.cpp.o.d -o CMakeFiles/pmc.dir/pmc_maxclique.cpp.o -c /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_maxclique.cpp

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_maxclique.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pmc.dir/pmc_maxclique.cpp.i"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_maxclique.cpp > CMakeFiles/pmc.dir/pmc_maxclique.cpp.i

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_maxclique.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pmc.dir/pmc_maxclique.cpp.s"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_maxclique.cpp -o CMakeFiles/pmc.dir/pmc_maxclique.cpp.s

_deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/flags.make
_deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique.cpp.o: _deps/pmc-src/pmcx_maxclique.cpp
_deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object _deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique.cpp.o"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique.cpp.o -MF CMakeFiles/pmc.dir/pmcx_maxclique.cpp.o.d -o CMakeFiles/pmc.dir/pmcx_maxclique.cpp.o -c /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmcx_maxclique.cpp

_deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pmc.dir/pmcx_maxclique.cpp.i"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmcx_maxclique.cpp > CMakeFiles/pmc.dir/pmcx_maxclique.cpp.i

_deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pmc.dir/pmcx_maxclique.cpp.s"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmcx_maxclique.cpp -o CMakeFiles/pmc.dir/pmcx_maxclique.cpp.s

_deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/flags.make
_deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.o: _deps/pmc-src/pmcx_maxclique_basic.cpp
_deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object _deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.o"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.o -MF CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.o.d -o CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.o -c /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmcx_maxclique_basic.cpp

_deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.i"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmcx_maxclique_basic.cpp > CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.i

_deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.s"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmcx_maxclique_basic.cpp -o CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.s

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_cores.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/flags.make
_deps/pmc-build/CMakeFiles/pmc.dir/pmc_cores.cpp.o: _deps/pmc-src/pmc_cores.cpp
_deps/pmc-build/CMakeFiles/pmc.dir/pmc_cores.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object _deps/pmc-build/CMakeFiles/pmc.dir/pmc_cores.cpp.o"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/pmc-build/CMakeFiles/pmc.dir/pmc_cores.cpp.o -MF CMakeFiles/pmc.dir/pmc_cores.cpp.o.d -o CMakeFiles/pmc.dir/pmc_cores.cpp.o -c /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_cores.cpp

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_cores.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pmc.dir/pmc_cores.cpp.i"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_cores.cpp > CMakeFiles/pmc.dir/pmc_cores.cpp.i

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_cores.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pmc.dir/pmc_cores.cpp.s"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_cores.cpp -o CMakeFiles/pmc.dir/pmc_cores.cpp.s

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_utils.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/flags.make
_deps/pmc-build/CMakeFiles/pmc.dir/pmc_utils.cpp.o: _deps/pmc-src/pmc_utils.cpp
_deps/pmc-build/CMakeFiles/pmc.dir/pmc_utils.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object _deps/pmc-build/CMakeFiles/pmc.dir/pmc_utils.cpp.o"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/pmc-build/CMakeFiles/pmc.dir/pmc_utils.cpp.o -MF CMakeFiles/pmc.dir/pmc_utils.cpp.o.d -o CMakeFiles/pmc.dir/pmc_utils.cpp.o -c /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_utils.cpp

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pmc.dir/pmc_utils.cpp.i"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_utils.cpp > CMakeFiles/pmc.dir/pmc_utils.cpp.i

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pmc.dir/pmc_utils.cpp.s"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_utils.cpp -o CMakeFiles/pmc.dir/pmc_utils.cpp.s

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_graph.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/flags.make
_deps/pmc-build/CMakeFiles/pmc.dir/pmc_graph.cpp.o: _deps/pmc-src/pmc_graph.cpp
_deps/pmc-build/CMakeFiles/pmc.dir/pmc_graph.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object _deps/pmc-build/CMakeFiles/pmc.dir/pmc_graph.cpp.o"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/pmc-build/CMakeFiles/pmc.dir/pmc_graph.cpp.o -MF CMakeFiles/pmc.dir/pmc_graph.cpp.o.d -o CMakeFiles/pmc.dir/pmc_graph.cpp.o -c /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_graph.cpp

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_graph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pmc.dir/pmc_graph.cpp.i"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_graph.cpp > CMakeFiles/pmc.dir/pmc_graph.cpp.i

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_graph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pmc.dir/pmc_graph.cpp.s"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_graph.cpp -o CMakeFiles/pmc.dir/pmc_graph.cpp.s

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_clique_utils.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/flags.make
_deps/pmc-build/CMakeFiles/pmc.dir/pmc_clique_utils.cpp.o: _deps/pmc-src/pmc_clique_utils.cpp
_deps/pmc-build/CMakeFiles/pmc.dir/pmc_clique_utils.cpp.o: _deps/pmc-build/CMakeFiles/pmc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object _deps/pmc-build/CMakeFiles/pmc.dir/pmc_clique_utils.cpp.o"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/pmc-build/CMakeFiles/pmc.dir/pmc_clique_utils.cpp.o -MF CMakeFiles/pmc.dir/pmc_clique_utils.cpp.o.d -o CMakeFiles/pmc.dir/pmc_clique_utils.cpp.o -c /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_clique_utils.cpp

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_clique_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pmc.dir/pmc_clique_utils.cpp.i"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_clique_utils.cpp > CMakeFiles/pmc.dir/pmc_clique_utils.cpp.i

_deps/pmc-build/CMakeFiles/pmc.dir/pmc_clique_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pmc.dir/pmc_clique_utils.cpp.s"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src/pmc_clique_utils.cpp -o CMakeFiles/pmc.dir/pmc_clique_utils.cpp.s

# Object files for target pmc
pmc_OBJECTS = \
"CMakeFiles/pmc.dir/pmc_heu.cpp.o" \
"CMakeFiles/pmc.dir/pmc_maxclique.cpp.o" \
"CMakeFiles/pmc.dir/pmcx_maxclique.cpp.o" \
"CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.o" \
"CMakeFiles/pmc.dir/pmc_cores.cpp.o" \
"CMakeFiles/pmc.dir/pmc_utils.cpp.o" \
"CMakeFiles/pmc.dir/pmc_graph.cpp.o" \
"CMakeFiles/pmc.dir/pmc_clique_utils.cpp.o"

# External object files for target pmc
pmc_EXTERNAL_OBJECTS =

_deps/pmc-build/libpmc.so: _deps/pmc-build/CMakeFiles/pmc.dir/pmc_heu.cpp.o
_deps/pmc-build/libpmc.so: _deps/pmc-build/CMakeFiles/pmc.dir/pmc_maxclique.cpp.o
_deps/pmc-build/libpmc.so: _deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique.cpp.o
_deps/pmc-build/libpmc.so: _deps/pmc-build/CMakeFiles/pmc.dir/pmcx_maxclique_basic.cpp.o
_deps/pmc-build/libpmc.so: _deps/pmc-build/CMakeFiles/pmc.dir/pmc_cores.cpp.o
_deps/pmc-build/libpmc.so: _deps/pmc-build/CMakeFiles/pmc.dir/pmc_utils.cpp.o
_deps/pmc-build/libpmc.so: _deps/pmc-build/CMakeFiles/pmc.dir/pmc_graph.cpp.o
_deps/pmc-build/libpmc.so: _deps/pmc-build/CMakeFiles/pmc.dir/pmc_clique_utils.cpp.o
_deps/pmc-build/libpmc.so: _deps/pmc-build/CMakeFiles/pmc.dir/build.make
_deps/pmc-build/libpmc.so: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
_deps/pmc-build/libpmc.so: /usr/lib/x86_64-linux-gnu/libpthread.so
_deps/pmc-build/libpmc.so: _deps/pmc-build/CMakeFiles/pmc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX shared library libpmc.so"
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pmc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/pmc-build/CMakeFiles/pmc.dir/build: _deps/pmc-build/libpmc.so
.PHONY : _deps/pmc-build/CMakeFiles/pmc.dir/build

_deps/pmc-build/CMakeFiles/pmc.dir/clean:
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build && $(CMAKE_COMMAND) -P CMakeFiles/pmc.dir/cmake_clean.cmake
.PHONY : _deps/pmc-build/CMakeFiles/pmc.dir/clean

_deps/pmc-build/CMakeFiles/pmc.dir/depend:
	cd /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-src /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build /home/kalliyanlay/Documents/BYU/research/CAAMS/GroupkCLIPPER/build/_deps/pmc-build/CMakeFiles/pmc.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : _deps/pmc-build/CMakeFiles/pmc.dir/depend

