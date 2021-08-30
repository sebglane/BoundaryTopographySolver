/*
 * solver_base.cc
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#include <deal.II/dofs/dof_renumbering.h>

#include <solver_base.h>

#include <iostream>

namespace TopographyProblem {

template <int dim>
SolverBase<dim>::SolverBase
(const unsigned int  n_refinements,
 const double        newton_tolerance,
 const unsigned int  n_maximum_iterations)
:
fe_system(nullptr),
dof_handler(triangulation),
computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
n_refinements(n_refinements),
newton_tolerance(newton_tolerance),
n_maximum_iterations(n_maximum_iterations)
{}



template <int dim>
void SolverBase<dim>::setup_dofs()
{
  TimerOutput::Scope timer_section(computing_timer, "Setup DoFs");

  std::cout << "   Setup dofs..." << std::endl;

  // distribute and renumber block-wise
  dof_handler.distribute_dofs(*fe_system);

  DoFRenumbering::block_wise(dof_handler);

  // IO
  std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler);

  std::cout << "      Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "      Number of total degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;
}



template<int dim>
void SolverBase<dim>::setup_vectors
(const std::vector<types::global_dof_index> &dofs_per_block)
{
  evaluation_point.reinit(dofs_per_block);
  present_solution.reinit(dofs_per_block);
  solution_update.reinit(dofs_per_block);
  system_rhs.reinit(dofs_per_block);
}



template<int dim>
void SolverBase<dim>::setup_system_matrix
(const std::vector<types::global_dof_index> &dofs_per_block,
 const Table<2, DoFTools::Coupling> &coupling_table)
{
  system_matrix.clear();

  BlockDynamicSparsityPattern dsp(dofs_per_block,
                                  dofs_per_block);

  DoFTools::make_sparsity_pattern(dof_handler,
                                  coupling_table,
                                  dsp,
                                  zero_constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
}



template <int dim>
void SolverBase<dim>::run()
{
  bool initial_step = true;

  this->make_grid();

  this->setup_fe_system();

  this->setup_dofs();

  for (unsigned int cycle = 0; cycle < n_refinements; ++cycle)
  {
    std::cout << "Cycle " << cycle << ':' << std::endl;

    newton_iteration(initial_step);

    this->postprocess_solution(cycle);

    this->output_results(cycle);

    if (cycle == 0)
        initial_step = false;
  }
}


template <int dim>
void SolverBase<dim>::newton_iteration(const bool is_initial_step)
{
  double current_residual = 1.0;
  double last_residual = 1.0;
  bool first_step = is_initial_step;

  unsigned int iteration = 0;

  while ((first_step || (current_residual > newton_tolerance)) &&
         iteration < n_maximum_iterations)
  {
    if (first_step)
    {
      // solve problem
      evaluation_point = present_solution;
      assemble_system(first_step);
      solve(first_step);
      present_solution = solution_update;
      nonzero_constraints.distribute(present_solution);
      first_step = false;
      // compute residual
      evaluation_point = present_solution;
      assemble_rhs(first_step);
      current_residual = system_rhs.l2_norm();
    }
    else
    {
      // solve problem
      evaluation_point = present_solution;
      assemble_system(first_step);
      solve(first_step);
      // line search
      std::cout << "   Line search: " << std::endl;
      for (double alpha = 1.0; alpha > 1e-4; alpha *= 0.5)
      {
        evaluation_point = present_solution;
        evaluation_point.add(alpha, solution_update);
        nonzero_constraints.distribute(evaluation_point);
        assemble_rhs(first_step);
        current_residual = system_rhs.l2_norm();
        std::cout << "      alpha = " << std::setw(6)
                  << std::scientific << alpha
                  << " residual = " << current_residual
                  << std::endl
                  << std::defaultfloat;
        if (current_residual < last_residual)
          break;
      }
      present_solution = evaluation_point;
    }

    // output residual
    std::cout << "Iteration: " << std::setw(4) << iteration
              << ", Current residual: "
              << std::scientific << current_residual
              << std::endl
              << std::defaultfloat;

    // update residual
    last_residual = current_residual;
    ++iteration;
  }
}



template SolverBase<2>::SolverBase(const unsigned int, const double, const unsigned int);
template SolverBase<3>::SolverBase(const unsigned int, const double, const unsigned int);

template void SolverBase<2>::setup_dofs();
template void SolverBase<3>::setup_dofs();

template void SolverBase<2>::setup_system_matrix
(const std::vector<types::global_dof_index> &,
 const Table<2, DoFTools::Coupling> &);
template void SolverBase<3>::setup_system_matrix
(const std::vector<types::global_dof_index> &,
 const Table<2, DoFTools::Coupling> &);

template void SolverBase<2>::setup_vectors
(const std::vector<types::global_dof_index> &);
template void SolverBase<3>::setup_vectors
(const std::vector<types::global_dof_index> &);

template void SolverBase<2>::newton_iteration(const bool);
template void SolverBase<3>::newton_iteration(const bool);

template void SolverBase<2>::run();
template void SolverBase<3>::run();

}  // namespace TopographyProblem



