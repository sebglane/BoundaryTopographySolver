PROJECT(applications)
ENABLE_TESTING()
INCLUDE_DIRECTORIES(
    include
    ${CMAKE_SOURCE_DIR}/include
    )

SET(SOURCE_FILES
    advection_problem.cc
    cavity_problem.cc
    viscous_topography_problem.cc
    viscous_buoyant_topography_problem.cc
    )

FOREACH(sourcefile ${SOURCE_FILES})
    # string replace: cut off .cc from files
    STRING(REPLACE ".cc" "" executablename ${sourcefile} )
    PROJECT(${executablename})
    ADD_EXECUTABLE(${executablename} ${sourcefile})
    DEAL_II_SETUP_TARGET(${executablename})
    TARGET_LINK_LIBRARIES(${executablename} boundary_topography)
ENDFOREACH(sourcefile ${APP_SOURCES})
