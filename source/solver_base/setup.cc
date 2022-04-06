/*
 * setup.cc
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <solver_base.h>

namespace SolverBase {

using TrilinosContainer = LinearAlgebraContainer<TrilinosWrappers::MPI::Vector,
                                                 TrilinosWrappers::SparseMatrix,
                                                 TrilinosWrappers::SparsityPattern>;



template <int dim>
using ParallelTriangulation =  parallel::distributed::Triangulation<dim>;



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void Solver<dim, TriangulationType, LinearAlgebraContainer>::setup_dofs()
{
  // distribute and renumber block-wise
  dof_handler.distribute_dofs(*fe_system);

  DoFRenumbering::block_wise(dof_handler);

  pcout << "    Number of active cells: "
        << triangulation.n_global_active_cells()
        << std::endl
        << "    Number of total degrees of freedom: "
        << dof_handler.n_dofs()
        << std::endl;

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  nonzero_constraints.clear();
  nonzero_constraints.reinit(locally_relevant_dofs);

  zero_constraints.clear();
  zero_constraints.reinit(locally_relevant_dofs);

  component_mean_values.clear();

  // possibly initialize the mapping
  MappingQCache<dim> *mapping_q_cache_ptr = dynamic_cast<MappingQCache<dim>*>(&mapping);
  if (mapping_q_cache_ptr != nullptr)
  {
    if (verbose)
      pcout << "Initialize mapping..." << std::endl;
    mapping_q_cache_ptr->initialize(triangulation,
                                    MappingQGeneric<dim>(mapping_q_cache_ptr->get_degree()));
  }

  this->apply_hanging_node_constraints();

  this->apply_boundary_conditions();

  nonzero_constraints.close();
  zero_constraints.close();
}


// explicit instantiations
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

}  // namespace SolverBase
