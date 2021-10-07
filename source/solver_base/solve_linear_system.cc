/*
 * solver_linear_system.cc
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>

#include <solver_base.h>

namespace SolverBase {

template <>
void Solver<2>::solve_linear_system(const bool use_homogeneous_constraints)
{
  if (verbose)
    std::cout << "    Solving linear system..." << std::endl;

  TimerOutput::Scope timer_section(computing_timer, "Solve linear system");

  SparseDirectUMFPACK     direct_solver;
  direct_solver.solve(container.system_matrix, container.system_rhs);

  container.solution_update = container.system_rhs;

//  if (dim==2)
//  {
//    SparseDirectUMFPACK     direct_solver;
//    direct_solver.solve(system_matrix, system_rhs);
//
//    solution_update = system_rhs;
//  }
//  else
//  {
//    SolverControl solver_control;
//
//    SparseILU<double>  preconditioner;
//    preconditioner.initialize(system_matrix,
//                              SparseILU<double>::AdditionalData());
//
//    SolverGMRES<BlockVector<double>>  solver(solver_control);
//    try
//    {
//      solver.solve(system_matrix, solution_update, system_rhs, preconditioner);
//    }
//    catch (std::exception &exc)
//    {
//      std::cerr << std::endl << std::endl
//              << "----------------------------------------------------"
//              << std::endl;
//      std::cerr << "Exception on iterative solution of the linear system: " << std::endl
//              << exc.what() << std::endl
//              << "Aborting!" << std::endl
//              << "----------------------------------------------------"
//              << std::endl;
//      std::abort();
//    }
//    catch (...)
//    {
//      std::cerr << std::endl << std::endl
//              << "----------------------------------------------------"
//              << std::endl;
//      std::cerr << "Unknown exception!" << std::endl
//              << "Aborting!" << std::endl
//              << "----------------------------------------------------"
//              << std::endl;
//      std::abort();
//    }
//  }

  const AffineConstraints<double> &constraints_used =
      (use_homogeneous_constraints ? zero_constraints: nonzero_constraints);
  constraints_used.distribute(container.solution_update);
}



template <>
void Solver<3>::solve_linear_system(const bool use_homogeneous_constraints)
{
  if (verbose)
    std::cout << "    Solving linear system..." << std::endl;

  TimerOutput::Scope timer_section(computing_timer, "Solve linear system");

  SparseDirectUMFPACK     direct_solver;
  direct_solver.solve(container.system_matrix, container.system_rhs);

  container.solution_update = container.system_rhs;

  const AffineConstraints<double> &constraints_used =
      (use_homogeneous_constraints ? zero_constraints: nonzero_constraints);
  constraints_used.distribute(container.solution_update);
}

// explicit instantiations
//template void Solver<2>::solve_linear_system(const bool);
//template void Solver<3>::solve_linear_system(const bool);

}  // namespace SolverBase

