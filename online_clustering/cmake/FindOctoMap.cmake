# Find the ZeroC ICE includes and libraries for every module (Ice, IceStorm, IceUtil, etc)

#
# OCTOMAP_INCLUDE_DIR - Where the includes are. If everything is all right, OctoMapXXXX_INCLUDE_DIR is always the same. You usually will use this.
# OCTOMAP_LIBRARIES - List of *all* the libraries. You usually will not use this but only OctoMapUtil_LIBRARY or alike
# ZerocCIce_FOUND - True if the core Ice was found

# OCTOMAP_FOUND
# OCTOMAP_INCLUDE_DIR
# OCTOMAP_LIBRARY


if("${CMAKE_SIZEOF_VOID_P}" MATCHES "8")
     set(_libsuffixes lib64 lib)
else()
     set(_libsuffixes lib)
endif()

###
# Configure OctoMap
###



find_path(OCTOMAP_INCLUDE_DIR octomap/OcTree.h)

if( OCTOMAP_INCLUDE_DIR )

  find_library(OCTOMAP_MATH_LIBRARY
     octomath
     PATH_SUFFIXES
     ${_libsuffixes})

  find_library(OCTOMAP_CORE_LIBRARY
     octomap
     PATH_SUFFIXES
     ${_libsuffixes})

  find_library(OCTOMAP_VIS_LIBRARY
     octovis
     PATH_SUFFIXES
     ${_libsuffixes})

  set (OCTOMAP_LIBRARY ${OCTOMAP_MATH_LIBRARY} ${OCTOMAP_CORE_LIBRARY}
   ${OCTOMAP_VIS_LIBRARY})
	
	if( OCTOMAP_CORE_LIBRARY )
	if( OCTOMAP_MATH_LIBRARY )
	if( OCTOMAP_VIS_LIBRARY )
		SET( OCTOMAP_FOUND TRUE )
	endif(OCTOMAP_VIS_LIBRARY)
	endif(OCTOMAP_MATH_LIBRARY )
	endif(OCTOMAP_CORE_LIBRARY )

	if(OCTOMAP_FOUND)
	  message(STATUS "Found the OctoMap library at ${OCTOMAP_LIBRARY}")
	  message(STATUS "Found the OctoMap headers at ${OCTOMAP_INCLUDE_DIR}")

	else(OCTOMAP_FOUND)
	  if(OCTOMAP_FIND_REQUIRED)
	    message(FATAL_ERROR "Could not find OctoMap")
          endif(OCTOMAP_FIND_REQUIRED)
	endif(OCTOMAP_FOUND)

endif( OCTOMAP_INCLUDE_DIR )

