/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */
#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <hydrodynamic_solver.h>

namespace Hydrodynamic {

using TrilinosContainer = typename
                          SolverBase::
                          LinearAlgebraContainer<TrilinosWrappers::MPI::Vector,
                                                 TrilinosWrappers::SparseMatrix,
                                                 TrilinosWrappers::SparsityPattern>;



template <int dim>
using ParallelTriangulation =  parallel::distributed::Triangulation<dim>;



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void Solver<dim, TriangulationType, LinearAlgebraContainer>::setup_fe_system()
{
  if (this->verbose)
    this->pcout << "    Setup FE system..." << std::endl;

  this->fe_system = std::make_shared<FESystem<dim>>(FESystem<dim>(FE_Q<dim>(velocity_fe_degree), dim), 1,
                                                    FE_Q<dim>(velocity_fe_degree - 1), 1);
}



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void Solver<dim, TriangulationType, LinearAlgebraContainer>::setup_dofs()
{
  TimerOutput::Scope timer_section(this->computing_timer, "Setup dofs");

  if (this->verbose)
    this->pcout << "    Setup dofs..." << std::endl;

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
  this->container.setup_vector(this->present_solution);
  this->container.setup_vector(this->evaluation_point);
  this->container.setup_vector(this->solution_update);
}

// explicit instantiation
template void Solver<2>::setup_fe_system();
template void Solver<3>::setup_fe_system();

template
void
Solver<2, ParallelTriangulation<2>, TrilinosContainer>::
setup_fe_system();
template
void
Solver<3, ParallelTriangulation<3>, TrilinosContainer>::
setup_fe_system();


template void Solver<2>::setup_dofs();
template void Solver<3>::setup_dofs();

template
void
Solver<2, ParallelTriangulation<2>, TrilinosContainer>::
setup_dofs();
template
void
Solver<3, ParallelTriangulation<3>, TrilinosContainer>::
setup_dofs();

}  // namespace Hydrodynamic
