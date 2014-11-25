# A CMake script to find the start clustering library
# and any required components
#
# Usage:
# find_package(StarClust)
#
# Cache Variables:
# StarClust_LIBRARY
# StarClust_INCLUDE_DIR
#

set(StarClust_ROOT_DIR
    "${StarClust_ROOT_DIR}"
    CACHE
    PATH
    "Root directory to search for StarClust"
)


###
# Configure StarClust
###

find_path(StarClust_HEADER
    NAMES
    onlineStarClusterer.h
    HINTS
    "${StarClust_ROOT_DIR}" "$ENV{MRG_BASE_ROOT}/../pOnlineStarClustering"
    PATHS
    "${_progfiles}/StarClust"
)

find_library(StarClust_LIBRARY
    NAMES
    pOSC
    HINTS
    "${StarClust_ROOT_DIR}" "$ENV{MRG_BASE_ROOT}/../pOnlineStarClustering/build"
    PATHS
    "${_progfiles}/StarClust"
)

if(${StarClust_HEADER} EQUAL "StarClust_HEADER-NOTFOUND" OR 
   ${StarClust_LIBRARY} EQUAL "StarClust_LIBRARY-NOTFOUND")
   message("Could not find Star Clustering library")
else()
set(StarClust_INCLUDE_DIR ${StarClust_HEADER})
get_filename_component(StarClust_LIBRARY_DIR ${StarClust_LIBRARY} PATH)
get_filename_component(StarClust_LIB ${StarClust_LIBRARY} NAME)
set(StarClust_FOUND true)
endif()

mark_as_advanced(StarClust_LIBRARY StarClust_INCLUDE_DIR)