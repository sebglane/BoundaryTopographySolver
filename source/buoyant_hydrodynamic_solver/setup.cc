/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/fe/fe_q.h>

#include <buoyant_hydrodynamic_solver.h>

namespace BuoyantHydrodynamic {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::setup_fe_system()
{
  if (this->verbose)
    this->pcout << "    Setup FE system..." << std::endl;

  this->fe_system = std::make_shared<FESystem<dim>>(FESystem<dim>(FE_Q<dim>(this->velocity_fe_degree), dim), 1,
                                                    FE_Q<dim>(this->velocity_fe_degree - 1), 1,
                                                    FE_Q<dim>(density_fe_degree), 1);
  this->velocity_fe_index = 0;
  this->pressure_fe_index = dim;
}



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::setup_dofs()
{
  TimerOutput::Scope timer_section(this->computing_timer, "Setup dofs");

  if (this->verbose)
    this->pcout << "    Setup dofs..." << std::endl;

  Base::Solver<dim, TriangulationType>::setup_dofs();

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
  if (this->stabilization & apply_pspg)
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

  this->setup_system_matrix(coupling_table);
  this->setup_vectors();
}

// explicit instantiation
template void Solver<2>::setup_fe_system();
template void Solver<3>::setup_fe_system();

template void Solver<2>::setup_dofs();
template void Solver<3>::setup_dofs();

}  // namespace BuoyantHydrodynamic
