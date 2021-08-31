/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/fe/fe_q.h>

#include <hydrodynamic_solver.h>

namespace TopographyProblem {

template <int dim>
void HydrodynamicSolver<dim>::setup_fe_system()
{
  this->fe_system = new FESystem<dim>(FESystem<dim>(FE_Q<dim>(velocity_fe_degree), dim), 1,
                                      FE_Q<dim>(velocity_fe_degree - 1), 1);

}



template <int dim>
void HydrodynamicSolver<dim>::setup_dofs()
{
  TimerOutput::Scope timer_section(this->computing_timer, "Setup dofs");

  std::cout << "   Setup dofs..." << std::endl;

  SolverBase<dim>::setup_dofs();

  std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(this->dof_handler);

  Table<2, DoFTools::Coupling>  coupling_table;
  coupling_table.reinit(this->fe_system->n_components(),
                        this->fe_system->n_components());

  // velocity-pressure coupling
  for (unsigned int c=0; c<dim+1; ++c)
    for (unsigned int d=0; d<dim+1; ++d)
      if (c<dim || d<dim)
        coupling_table[c][d] = DoFTools::always;
      else if ((c==dim && d<dim) || (c<dim && d==dim))
        coupling_table[c][d] = DoFTools::always;
      else
        coupling_table[c][d] = DoFTools::none;

  this->setup_system_matrix(dofs_per_block, coupling_table);
  this->setup_vectors(dofs_per_block);

}

// explicit instantiation
template void HydrodynamicSolver<2>::setup_fe_system();
template void HydrodynamicSolver<3>::setup_fe_system();

template void HydrodynamicSolver<2>::setup_dofs();
template void HydrodynamicSolver<3>::setup_dofs();

}  // namespace TopographyProblem
