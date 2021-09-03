# Set the source files to be compiled
SET(TARGET_SRC
    boundary_conditions.cc
    hydrodynamic_postprocessor.cc
    hydrodynamic_problem.cc
    parameters.cc
    # solver base
    solver_base/apply_constraints.cc
    solver_base/refine_mesh.cc
    solver_base/setup.cc
    solver_base/solve_linear_system.cc
    solver_base/solver_base.cc
    # hydrodynamic solver
    hydrodynamic_solver/apply_boundary_conditions.cc
    hydrodynamic_solver/assemble_system.cc
    hydrodynamic_solver/assemble_rhs.cc
    hydrodynamic_solver/hydrodynamic_solver.cc
    hydrodynamic_solver/setup.cc
    # assemble_system.cc
    # assembly_data.cc
    # entropy_viscosity.cc
    # equation_data.cc
    grid_factory.cc
    # linear_solver.cc
    # local_assemble.cc
    # make_grid.cc
    # newton_iteration.cc
    # postprocessor.cc
    # postprocessing.cc
    # setup.cc
    # solver.cc
)
ADD_LIBRARY(boundary_topography SHARED ${TARGET_SRC})
DEAL_II_SETUP_TARGET(boundary_topography)

