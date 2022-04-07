/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/fe/fe_q.h>

#include <advection_solver.h>

namespace Advection {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::setup_fe_system()
{
  if (this->verbose)
    this->pcout << "    Setup FE system..." << std::endl;

  this->fe_system = std::make_shared<FESystem<dim>>(FE_Q<dim>(fe_degree), 1);

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
  for (unsigned int r=0; r<coupling_table.n_rows(); ++r)
    for (unsigned int c=0; c<coupling_table.n_cols(); ++c)
      coupling_table[r][c] = DoFTools::always;

  this->setup_system_matrix(coupling_table);
  this->setup_vectors();
}

// explicit instantiation
template void Solver<2>::setup_fe_system();
template void Solver<3>::setup_fe_system();

template void Solver<2>::setup_dofs();
template void Solver<3>::setup_dofs();

}  // namespace Advection
