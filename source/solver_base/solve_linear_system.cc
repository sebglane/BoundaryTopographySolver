/*
 * solver_linear_system.cc
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */
#include <deal.II/lac/sparse_direct.h>

#include <solver_base.h>

namespace SolverBase {

template<int dim>
void Solver<dim>::solve_linear_system(const bool use_homogeneous_constraints)
{
  if (verbose)
    std::cout << "    Solving linear system..." << std::endl;

  TimerOutput::Scope timer_section(computing_timer, "Solve linear system");

  SparseDirectUMFPACK     direct_solver;
  direct_solver.solve(system_matrix, system_rhs);

  solution_update = system_rhs;
  const AffineConstraints<double> &constraints_used =
      (use_homogeneous_constraints ? zero_constraints: nonzero_constraints);
  constraints_used.distribute(solution_update);
}

// explicit instantiations
template void Solver<2>::solve_linear_system(const bool);
template void Solver<3>::solve_linear_system(const bool);

}  // namespace SolverBase

