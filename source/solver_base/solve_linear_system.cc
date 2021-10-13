/*
 * solver_linear_system.cc
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <solver_base.h>

namespace SolverBase {

namespace internal
{

template<typename VectorType, typename MatrixType>
void solve_trilinos(const MatrixType &system_matrix,
                    const VectorType &system_rhs,
                    VectorType        solution)
{
  VectorType distributed_solution(system_rhs);

  try
  {
    SolverControl solver_control;
    TrilinosWrappers::SolverGMRES solver(solver_control);

    TrilinosWrappers::PreconditionILU preconditioner;
    preconditioner.initialize(system_matrix,
                              TrilinosWrappers::PreconditionILU::AdditionalData());

    solver.solve(system_matrix,
                 distributed_solution,
                 system_rhs,
                 preconditioner);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
    std::cerr << "Exception on iterative solution of the linear system: " << std::endl
            << exc.what() << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
    std::abort();
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
    std::cerr << "Unknown exception!" << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
    std::abort();
  }

  solution = distributed_solution;
}

}  // namespace internal

//template <>
//void Solver<2, Triangulation<2>, LinearAlgebraContainer<TrilinosWrappers::MPI::BlockVector,
//                                                        TrilinosWrappers::SparseMatrix,
//                                                        TrilinosWrappers::SparsityPattern>>::
//solve_linear_system(const bool use_homogeneous_constraints)
//{
//
//  TrilinosWrappers::MPI::Vector system_rhs;
//  system_rhs.reinit(container.get_locally_owned_dofs(),
//                    MPI_COMM_WORLD);
//  system_rhs = container.system_rhs;
//
//  TrilinosWrappers::MPI::Vector solution(system_rhs);
//
//
//  internal::solve_trilinos(container.system_matrix,
//                           system_rhs,
//                           solution);
//
//  const AffineConstraints<double> &constraints_used =
//      (use_homogeneous_constraints ? zero_constraints: nonzero_constraints);
//
//  constraints_used.distribute(container.solution_update);
//}
//
//
//
//template <>
//void Solver<3, Triangulation<3>, LinearAlgebraContainer<TrilinosWrappers::MPI::BlockVector,
//                                                        TrilinosWrappers::SparseMatrix,
//                                                        TrilinosWrappers::SparsityPattern>>::
//solve_linear_system(const bool use_homogeneous_constraints)
//{
//
//  internal::solve_trilinos(container.system_matrix,
//                           container.system_rhs,
//                           container.solution_update);
//
//  const AffineConstraints<double> &constraints_used =
//      (use_homogeneous_constraints ? zero_constraints: nonzero_constraints);
//
//  constraints_used.distribute(container.solution_update);
//}
//
//
//
//template <>
//void Solver<2, Triangulation<2>, LinearAlgebraContainer<TrilinosWrappers::MPI::Vector,
//                                                        TrilinosWrappers::SparseMatrix,
//                                                        TrilinosWrappers::SparsityPattern>>::
//solve_linear_system(const bool use_homogeneous_constraints)
//{
//
//  internal::solve_trilinos(container.system_matrix,
//                           container.system_rhs,
//                           container.solution_update);
//
//  const AffineConstraints<double> &constraints_used =
//      (use_homogeneous_constraints ? zero_constraints: nonzero_constraints);
//
//  constraints_used.distribute(container.solution_update);
//}
//
//
//
//template <>
//void Solver<3, Triangulation<3>, LinearAlgebraContainer<TrilinosWrappers::MPI::Vector,
//                                                        TrilinosWrappers::SparseMatrix,
//                                                        TrilinosWrappers::SparsityPattern>>::
//solve_linear_system(const bool use_homogeneous_constraints)
//{
//
//  internal::solve_trilinos(container.system_matrix,
//                           container.system_rhs,
//                           container.solution_update);
//
//  const AffineConstraints<double> &constraints_used =
//      (use_homogeneous_constraints ? zero_constraints: nonzero_constraints);
//
//  constraints_used.distribute(container.solution_update);
//}



template <>
void Solver<2>::solve_linear_system(const bool use_homogeneous_constraints)
{
  if (verbose)
    std::cout << "    Solving linear system..." << std::endl;

  TimerOutput::Scope timer_section(computing_timer, "Solve linear system");

  SparseDirectUMFPACK     direct_solver;
  direct_solver.solve(container.system_matrix, container.system_rhs);

  container.set_solution_update(container.system_rhs);

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

  container.set_solution_update(container.system_rhs);

  const AffineConstraints<double> &constraints_used =
      (use_homogeneous_constraints ? zero_constraints: nonzero_constraints);
  constraints_used.distribute(container.solution_update);
}

}  // namespace SolverBase

