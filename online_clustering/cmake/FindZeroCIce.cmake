# Find the ZeroC ICE includes and libraries for every module (Ice, IceStorm, IceUtil, etc)

#
# ZeroCIce_INCLUDE_DIR - Where the includes are. If everything is all right, ZeroCIceXXXX_INCLUDE_DIR is always the same. You usually will use this.
# ZeroCIce_LIBRARIES - List of *all* the libraries. You usually will not use this but only ZeroCIceUtil_LIBRARY or alike
# ZerocCIce_FOUND - True if the core Ice was found

# ZeroCIce_FOUND
# ZeroCIce_INCLUDE_DIR
# ZeroCIce_LIBRARY


set(ICE_ROOT_DIR "/opt/Ice-3.4" CACHE PATH  "Root directory to search for Ice")

if("${CMAKE_SIZEOF_VOID_P}" MATCHES "8")
     set(_libsuffixes lib64 lib)
else()
     set(_libsuffixes lib)
endif()

###
# Configure ICE
###



find_path(ZeroCIce_INCLUDE_DIR
     IceUtil/IceUtil.h PATH_SUFFIXES include PATHS ${ICE_ROOT_DIR})

IF( ZeroCIce_INCLUDE_DIR )


find_library(ZeroCIceCore_LIBRARY
     NAMES
     Ice
     PATH_SUFFIXES
     ${_libsuffixes}
     HINTS
     "${ICE_ROOT_DIR}")

find_library(ZeroCIceUtil_LIBRARY
     NAMES
     IceUtil
     PATH_SUFFIXES
     ${_libsuffixes}
     HINTS
     "${ICE_ROOT_DIR}")

	SET (ZeroCIce_LIBRARY ${ZeroCIceCore_LIBRARY} ${ZeroCIceUtil_LIBRARY})
	
	IF( ZeroCIceCore_LIBRARY )
	IF( ZeroCIceUtil_LIBRARY )
		SET( ZeroCIce_FOUND TRUE )
	ENDIF(ZeroCIceUtil_LIBRARY)
	ENDIF(ZeroCIceCore_LIBRARY )

	IF(ZeroCIce_FOUND)
			MESSAGE(STATUS "Found the ZeroC Ice library at ${ZeroCIce_LIBRARY}")
			MESSAGE(STATUS "Found the ZeroC Ice headers at ${ZeroCIce_INCLUDE_DIR}")
			#mark_as_advanced(ZeroCIceUtil_LIBRARY      ZeroCIce_LIBRARY      ZeroCIceCore_LIBRARY      ZeroCIce_INCLUDE_DIR      ICE_ROOT_DIR)
	ELSE(ZeroCIce_FOUND)
		IF(ZeroCIce_FIND_REQUIRED)
			MESSAGE(FATAL_ERROR "Could NOT find ZeroC Ice")
		ENDIF(ZeroCIce_FIND_REQUIRED)
	ENDIF(ZeroCIce_FOUND)

ENDIF( ZeroCIce_INCLUDE_DIR )

