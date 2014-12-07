# Configuration file for the SpInspeC library.

# Pick up the auto-generated file which knows how to add the library targets
# This will mean that we don't have to supply full paths for the libraries
set(exports_file "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/build/UseSpInspeC.cmake")
if (EXISTS ${exports_file})
  include(${exports_file})
endif ()

set(SPINSPEC_INCLUDE_DIRS )
set(SPINSPEC_LIB_DIR    )
set(SPINSPEC_LIBRARIES    )
