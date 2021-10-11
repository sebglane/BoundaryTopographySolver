/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/fe/fe_q.h>

#include <hydrodynamic_solver.h>

namespace Hydrodynamic {

template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void Solver<dim, TriangulationType, LinearAlgebraContainer>::setup_fe_system()
{
  if (this->verbose)
    std::cout << "    Setup FE system..." << std::endl;

  this->fe_system = new FESystem<dim>(FESystem<dim>(FE_Q<dim>(velocity_fe_degree), dim), 1,
                                      FE_Q<dim>(velocity_fe_degree - 1), 1);

}



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void Solver<dim, TriangulationType, LinearAlgebraContainer>::setup_dofs()
{
  TimerOutput::Scope timer_section(this->computing_timer, "Setup dofs");

  if (this->verbose)
    std::cout << "    Setup dofs..." << std::endl;

  SolverBase::Solver<dim, TriangulationType, LinearAlgebraContainer>::setup_dofs();

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

  if (stabilization & apply_pspg)
    coupling_table[dim][dim] = DoFTools::always;

  this->container.setup(this->dof_handler,
                        this->zero_constraints,
                        coupling_table,
                        this->fe_system->n_blocks());
}

// explicit instantiation
template void Solver<2>::setup_fe_system();
template void Solver<3>::setup_fe_system();

template void Solver<2>::setup_dofs();
template void Solver<3>::setup_dofs();

}  // namespace Hydrodynamic
