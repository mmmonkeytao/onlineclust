# Find RemoteVR header and library

#
# RemoteVR_INCLUDE_DIR - Where the includes are. If everything is all right, RemoteVRXXXX_INCLUDE_DIR is always the same. You usually will use this.
# RemoteVR_LIBRARIES - List of *all* the libraries. You usually will not use this but only RemoteVRUtil_LIBRARY or alike
# ZerocCIce_FOUND - True if the core Ice was found

# RemoteVR_FOUND
# RemoteVR_INCLUDE_DIR
# RemoteVR_LIBRARY



#find_path(RemoteVR_INCLUDE_DIR RemoteVR.h
#          PATHS $ENV{HOME}/Software/ARFWork/RemoteVR/cpp/src)

### I HAVE NO IDEA HOW TO PROPERLY FIND THAT STUPID LIBRARY ###
set(RemoteVR_INCLUDE_DIR "$ENV{HOME}/Software/ARFWork/ARF/RemoteVR/cpp/build/RemoteVRLib")

if( RemoteVR_INCLUDE_DIR )


find_library(RemoteVR_LIBRARY
     NAMES
     RemoteVR
     PATH_SUFFIXES
     bin/lib/
     HINTS
     "${RemoteVR_INCLUDE_DIR}/../..")

#set (RemoteVR_LIB_DIR "${RemoteVR_INCLUDE_DIR}/../../bin/lib")
set (RemoteVR_LIB_DIR "$ENV{HOME}/Software/ARFWork/ARF/RemoteVR/cpp/build/lib")
set( RemoteVR_FOUND TRUE )

if(RemoteVR_FOUND)
  message(STATUS "Found the RemoteVR library at ${RemoteVR_LIBRARY}")
else(RemoteVR_FOUND)
	if(RemoteVR_FIND_REQUIRED)
			message(FATAL_ERROR "Could NOT find RemoteVR")
  endif(RemoteVR_FIND_REQUIRED)
endif(RemoteVR_FOUND)

endif( RemoteVR_INCLUDE_DIR )

