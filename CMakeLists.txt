##
#  CMake script
##
MESSAGE("========================================================")
MESSAGE("========= Configuring BoundaryTopographySolver  ========")
MESSAGE("========= An adaptive flow solver ======================")
MESSAGE("========= for stationary problems with boundary ========")
MESSAGE("========= topography. ==================================")
##
# Is the cmake version sufficient?
##
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
SET(CMAKE_CXX_STANDARD 20)

IF (EXISTS ${CMAKE_SOURCE_DIR}/CMakeCache.txt)
  MESSAGE(FATAL_ERROR  "Detected the file\n"
  "${CMAKE_SOURCE_DIR}/CMakeCache.txt\n"
  "in your source directory, which is a left-over from an in-source build. "
  "Please delete the file before running cmake from a separate build directory.")
ENDIF()

IF ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  MESSAGE(FATAL_ERROR  "In-source builds in ${CMAKE_BINARY_DIR} are not support. "
  "Please run cmake from a separate build directory."
  "\n"
  "Note that CMake created a file called CMakeCache.txt and a folder called "
  "CMakeFiles in the source directory that you have to remove before you can "
  "begin a build in a different directory.")
ENDIF()

##
# Is the deal.ii library installed?
##
FIND_PACKAGE(deal.II 9.3.0 REQUIRED QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()


##
# Are all dependencies fulfilled?
##

# p4est
IF(NOT DEAL_II_WITH_P4EST)
  MESSAGE(FATAL_ERROR "\n"
  "BoundaryTopographySolver requires a deal.II installation built with support for p4est but it appears to be missing!"
  "\n"
    )
ENDIF()

# Trilinos or PETSc
IF((NOT DEAL_II_WITH_TRILINOS) AND (NOT DEAL_II_WITH_PETSC))

   MESSAGE(FATAL_ERROR "\n"
   "This library requires a deal.II installation built with support for Trilinos "
   "or PETSc but both of these appear to be missing!"
   "This library requires a deal.II library that was configured with the following "
   "options:\n"
   "     DEAL_II_WITH_P4EST = ON\n"
   "     DEAL_II_WITH_PETSC = ON or DEAL_II_WITH_TRILINOS = ON\n"
   "However, the deal.II library found at ${DEAL_II_PATH} was configured with "
   "these options\n"
   "     DEAL_II_WITH_P4EST = ${DEAL_II_WITH_P4EST}\n"
   "     DEAL_II_WITH_TRILINOS = ${DEAL_II_WITH_TRILINOS}\n"
   "     DEAL_II_WITH_PETSC = ${DEAL_II_WITH_PETSC}\n"
   "which conflict with the requirements.\n"
   "One or both of the aforementioned combinations of prerequisites are not met "
   "by your installation, but at least one is required for\n"
   "this library.")

ENDIF()

IF(NOT DEAL_II_WITH_UMFPACK) # keep in one line
   MESSAGE(FATAL_ERROR "\n"
   "This library requires a deal.II installation built with support for UMFPACK"
   "but this appears to be missing!\n"
    )
ENDIF()

# DEBUG vs. RELEASE
ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
)

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)

# Set the include directory and the name of the project
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(boundary_topography)
ENABLE_TESTING()
INCLUDE_DIRECTORIES(include/
                    ${CMAKE_SOURCE_DIR}/include/
                    SYSTEM)
ADD_SUBDIRECTORY(applications)
ADD_SUBDIRECTORY(source)
ADD_SUBDIRECTORY(tests/applications/)
ADD_SUBDIRECTORY(tests/basic_tests/)

MESSAGE("========================================================")