# Set the source files to be compiled
SET(TARGET "boundary_topography")

SET(TARGET_SRC
    angular_velocity.cc
    assembly_functions.cc
    block_schur_preconditioner.cc
    boundary_conditions.cc
    evaluation_boundary_traction.cc
    evaluation_stabilization.cc
    grid_factory.cc
    parameters.cc
    postprocessor_base.cc
    # solver base
    solver_base/apply_constraints.cc
    solver_base/copy_local_to_global.cc
    solver_base/refine_mesh.cc
    solver_base/setup.cc
    solver_base/solve_linear_system.cc
    solver_base/solver_base.cc
    # hydrodynamic solver
    hydrodynamic_assembly_data.cc
    hydrodynamic_options.cc
    hydrodynamic_postprocessor.cc
    hydrodynamic_problem.cc
    hydrodynamic_solver/apply_boundary_conditions.cc
    hydrodynamic_solver/assemble_local_boundary.cc
    hydrodynamic_solver/assemble_rhs.cc
    hydrodynamic_solver/assemble_rhs_local.cc
    hydrodynamic_solver/assemble_system.cc
    hydrodynamic_solver/assemble_system_local.cc
    hydrodynamic_solver/hydrodynamic_solver.cc
    hydrodynamic_solver/setup.cc
    # buoyant hydrodynamic solver
    buoyant_hydrodynamic_assembly_data.cc
    buoyant_hydrodynamic_options.cc
    buoyant_hydrodynamic_problem.cc
    buoyant_hydrodynamic_solver/apply_boundary_conditions.cc
    buoyant_hydrodynamic_solver/assemble_system.cc
    buoyant_hydrodynamic_solver/assemble_system_local.cc
    buoyant_hydrodynamic_solver/assemble_rhs.cc
    buoyant_hydrodynamic_solver/assemble_rhs_local.cc
    buoyant_hydrodynamic_solver/buoyant_hydrodynamic_solver.cc
    buoyant_hydrodynamic_solver/setup.cc
    # advection solver
    advection_assembly_data.cc
    advection_options.cc
    advection_problem.cc
    advection_solver/apply_boundary_conditions.cc
    advection_solver/assemble_system.cc
    advection_solver/assemble_system_local.cc
    advection_solver/assemble_rhs.cc
    advection_solver/assemble_rhs_local.cc
    advection_solver/assemble_local_boundary.cc
    advection_solver/advection_solver.cc
    advection_solver/setup.cc
    # magnetic induction solver
    magnetic_induction_assembly_data.cc
    magnetic_induction_options.cc
    magnetic_postprocessor.cc
)
ADD_LIBRARY(${TARGET} SHARED ${TARGET_SRC})
DEAL_II_SETUP_TARGET(boundary_topography)
