# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.0.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.0.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/build

# Include any dependencies generated for this target.
include CMakeFiles/BayesOpt.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/BayesOpt.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/BayesOpt.dir/flags.make

CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o: CMakeFiles/BayesOpt.dir/flags.make
CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o: ../test/bayes_opt.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o -c /Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/test/bayes_opt.cc

CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/test/bayes_opt.cc > CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.i

CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/test/bayes_opt.cc -o CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.s

CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o.requires:
.PHONY : CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o.requires

CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o.provides: CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o.requires
	$(MAKE) -f CMakeFiles/BayesOpt.dir/build.make CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o.provides.build
.PHONY : CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o.provides

CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o.provides.build: CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o

# Object files for target BayesOpt
BayesOpt_OBJECTS = \
"CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o"

# External object files for target BayesOpt
BayesOpt_EXTERNAL_OBJECTS =

BayesOpt: CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o
BayesOpt: CMakeFiles/BayesOpt.dir/build.make
BayesOpt: libgplib.a
BayesOpt: /usr/local/lib/libgsl.dylib
BayesOpt: /usr/local/lib/libgslcblas.dylib
BayesOpt: CMakeFiles/BayesOpt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable BayesOpt"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/BayesOpt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/BayesOpt.dir/build: BayesOpt
.PHONY : CMakeFiles/BayesOpt.dir/build

CMakeFiles/BayesOpt.dir/requires: CMakeFiles/BayesOpt.dir/test/bayes_opt.cc.o.requires
.PHONY : CMakeFiles/BayesOpt.dir/requires

CMakeFiles/BayesOpt.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/BayesOpt.dir/cmake_clean.cmake
.PHONY : CMakeFiles/BayesOpt.dir/clean

CMakeFiles/BayesOpt.dir/depend:
	cd /Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib /Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib /Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/build /Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/build /Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/build/CMakeFiles/BayesOpt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/BayesOpt.dir/depend

