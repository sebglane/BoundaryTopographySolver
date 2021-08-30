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
SolverBase<dim>::SolverBase()
:
fe_system(nullptr),
dof_handler(triangulation),
computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
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

template SolverBase<2>::SolverBase();
template SolverBase<3>::SolverBase();

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

}  // namespace TopographyProblem



