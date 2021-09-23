/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/fe/fe_q.h>

#include <buoyant_hydrodynamic_solver.h>

namespace BuoyantHydrodynamic {

template <int dim>
void Solver<dim>::setup_fe_system()
{
  if (this->verbose)
    std::cout << "    Setup FE system..." << std::endl;

  this->fe_system = new FESystem<dim>(FESystem<dim>(FE_Q<dim>(this->velocity_fe_degree), dim), 1,
                                      FE_Q<dim>(this->velocity_fe_degree - 1), 1,
                                      FE_Q<dim>(density_fe_degree), 1);

}



template <int dim>
void Solver<dim>::setup_dofs()
{
  TimerOutput::Scope timer_section(this->computing_timer, "Setup dofs");

  if (this->verbose)
    std::cout << "    Setup dofs..." << std::endl;

  SolverBase:: Solver<dim>::setup_dofs();

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
  if (this->stabilization & Hydrodynamic::apply_pspg)
  {
    coupling_table[dim][dim] = DoFTools::always;
    coupling_table[dim][dim+1] = DoFTools::always;
  }

  // density-velocity coupling
  for (unsigned int c=0; c<dim; ++c)
  {
    coupling_table[dim+1][c] = DoFTools::always;
    coupling_table[c][dim+1] = DoFTools::always;
  }
  // density-density coupling
  coupling_table[dim+1][dim+1] = DoFTools::always;

  this->setup_system_matrix(dofs_per_block, coupling_table);
  this->setup_vectors(dofs_per_block);

}

// explicit instantiation
template void Solver<2>::setup_fe_system();
template void Solver<3>::setup_fe_system();

template void Solver<2>::setup_dofs();
template void Solver<3>::setup_dofs();

}  // namespace BuoyantHydrodynamic
