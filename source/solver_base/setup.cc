/*
 * setup.cc
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */
#include <base.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/mapping_q_cache.h>

namespace Base {


template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
setup_dofs()
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




template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
setup_system_matrix(const Table<2, DoFTools::Coupling>  &coupling_table)
{
  system_matrix.clear();

  const std::vector<types::global_dof_index>
  dofs_per_block{DoFTools::count_dofs_per_fe_block(dof_handler)};

  BlockDynamicSparsityPattern dsp(dofs_per_block,
                                  dofs_per_block);

  DoFTools::make_sparsity_pattern(dof_handler,
                                  coupling_table,
                                  dsp,
                                  nonzero_constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
}



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
setup_vectors()
{
  const std::vector<types::global_dof_index>
  dofs_per_block{DoFTools::count_dofs_per_fe_block(dof_handler)};

  system_rhs.reinit(dofs_per_block);
  present_solution.reinit(dofs_per_block);
  evaluation_point.reinit(dofs_per_block);
  solution_update.reinit(dofs_per_block);
}



// explicit instantiations
template void Solver<2>::setup_dofs();
template void Solver<3>::setup_dofs();

template void Solver<2>::setup_system_matrix(const Table<2, DoFTools::Coupling> &);
template void Solver<3>::setup_system_matrix(const Table<2, DoFTools::Coupling> &);

template void Solver<2>::setup_vectors();
template void Solver<3>::setup_vectors();


}  // namespace Base
