SET(SOURCE_FILES
    advection_problem.cc
#    buoyant_topography_problem.cc
#    buoyant_topography_problem_perturbed.cc
#    buoyant_rotating_topography_problem.cc
#    buoyant_rotating_topography_problem_perturbed.cc
    cavity_problem.cc
    rotating_cavity_problem.cc
    rotating_topography_problem.cc
    topography_problem.cc
    topography_problem_perturbed.cc
    topography_mesh.cc
    )

FOREACH(sourcefile ${SOURCE_FILES})
    # string replace: cut off .cc from files
    STRING(REPLACE ".cc" "" executablename ${sourcefile} )
    PROJECT(${executablename})
    ADD_EXECUTABLE(${executablename} ${sourcefile})
    DEAL_II_SETUP_TARGET(${executablename})
    TARGET_LINK_LIBRARIES(${executablename} boundary_topography)
ENDFOREACH(sourcefile ${SOURCE_FILES})