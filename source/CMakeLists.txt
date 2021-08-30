# Set the source files to be compiled
SET(TARGET_SRC
    assemble_system.cc
    assembly_data.cc
    entropy_viscosity.cc
    equation_data.cc
    grid_factory.cc
    linear_solver.cc
    local_assemble.cc
    make_grid.cc
    newton_iteration.cc
    parameters.cc
    postprocessor.cc
    postprocessing.cc
    setup.cc
    solver.cc
)
ADD_LIBRARY(boundary_topography SHARED ${TARGET_SRC})
DEAL_II_SETUP_TARGET(boundary_topography)

