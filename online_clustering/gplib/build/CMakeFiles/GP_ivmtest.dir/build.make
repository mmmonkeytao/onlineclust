# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jimmy/workspace/thesis/thesis_code/online_clustering/clustering/gplib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jimmy/workspace/thesis/thesis_code/online_clustering/clustering/gplib/build

# Include any dependencies generated for this target.
include CMakeFiles/GP_ivmtest.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/GP_ivmtest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GP_ivmtest.dir/flags.make

CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o: CMakeFiles/GP_ivmtest.dir/flags.make
CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o: ../test/gp_ivmtest.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jimmy/workspace/thesis/thesis_code/online_clustering/clustering/gplib/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o -c /home/jimmy/workspace/thesis/thesis_code/online_clustering/clustering/gplib/test/gp_ivmtest.cc

CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jimmy/workspace/thesis/thesis_code/online_clustering/clustering/gplib/test/gp_ivmtest.cc > CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.i

CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jimmy/workspace/thesis/thesis_code/online_clustering/clustering/gplib/test/gp_ivmtest.cc -o CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.s

CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o.requires:
.PHONY : CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o.requires

CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o.provides: CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o.requires
	$(MAKE) -f CMakeFiles/GP_ivmtest.dir/build.make CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o.provides.build
.PHONY : CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o.provides

CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o.provides.build: CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o

# Object files for target GP_ivmtest
GP_ivmtest_OBJECTS = \
"CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o"

# External object files for target GP_ivmtest
GP_ivmtest_EXTERNAL_OBJECTS =

GP_ivmtest: CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o
GP_ivmtest: CMakeFiles/GP_ivmtest.dir/build.make
GP_ivmtest: libgplib.a
GP_ivmtest: /usr/local/lib/libgsl.so
GP_ivmtest: /usr/local/lib/libgslcblas.so
GP_ivmtest: CMakeFiles/GP_ivmtest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable GP_ivmtest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GP_ivmtest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GP_ivmtest.dir/build: GP_ivmtest
.PHONY : CMakeFiles/GP_ivmtest.dir/build

CMakeFiles/GP_ivmtest.dir/requires: CMakeFiles/GP_ivmtest.dir/test/gp_ivmtest.cc.o.requires
.PHONY : CMakeFiles/GP_ivmtest.dir/requires

CMakeFiles/GP_ivmtest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GP_ivmtest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GP_ivmtest.dir/clean

CMakeFiles/GP_ivmtest.dir/depend:
	cd /home/jimmy/workspace/thesis/thesis_code/online_clustering/clustering/gplib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jimmy/workspace/thesis/thesis_code/online_clustering/clustering/gplib /home/jimmy/workspace/thesis/thesis_code/online_clustering/clustering/gplib /home/jimmy/workspace/thesis/thesis_code/online_clustering/clustering/gplib/build /home/jimmy/workspace/thesis/thesis_code/online_clustering/clustering/gplib/build /home/jimmy/workspace/thesis/thesis_code/online_clustering/clustering/gplib/build/CMakeFiles/GP_ivmtest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/GP_ivmtest.dir/depend
