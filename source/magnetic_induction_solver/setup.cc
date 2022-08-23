/*
 * setup.cc
 *
 *  Created on: Aug 22, 2022
 *      Author: sg
 */

#include <deal.II/fe/fe_q.h>

#include <magnetic_induction_solver.h>

namespace MagneticInduction {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::setup_fe_system()
{
  if (this->verbose)
    this->pcout << "    Setup FE system..." << std::endl;

  magnetic_field_fe_index = 0;
  magnetic_field_block_index = 0;

  magnetic_pressure_block_index = 1;
  magnetic_pressure_fe_index = dim;

  this->fe_system = std::make_shared<FESystem<dim>>(FE_Q<dim>(magnetic_fe_degree), dim,
                                                    FE_Q<dim>(magnetic_fe_degree - 1), 1);
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

  // magnetic field - pseudo-pressure coupling
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

}  // namespace MagneticInduction
