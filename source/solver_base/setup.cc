/*
 * setup.cc
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/mapping_q_cache.h>

#include <solver_base.h>

namespace SolverBase {

template <int dim>
void Solver<dim>::setup_dofs()
{
  // distribute and renumber block-wise
  dof_handler.distribute_dofs(*fe_system);

  DoFRenumbering::block_wise(dof_handler);

  std::cout << "    Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "    Number of total degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  nonzero_constraints.clear();
  zero_constraints.clear();

  // possibly initialize the mapping
  MappingQCache<dim> *mapping_q_cache_ptr = dynamic_cast<MappingQCache<dim>*>(&mapping);
  if (mapping_q_cache_ptr != nullptr)
  {
    if (verbose)
      std::cout << "Initialize mapping..." << std::endl;
    mapping_q_cache_ptr->initialize(triangulation,
                                    MappingQGeneric<dim>(mapping_q_cache_ptr->get_degree()));
  }

  this->apply_hanging_node_constraints();

  this->apply_boundary_conditions();

  nonzero_constraints.close();
  zero_constraints.close();
}



template<int dim>
void Solver<dim>::setup_system_matrix
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



template<int dim>
void Solver<dim>::setup_vectors
(const std::vector<types::global_dof_index> &dofs_per_block)
{
  evaluation_point.reinit(dofs_per_block);
  present_solution.reinit(dofs_per_block);
  solution_update.reinit(dofs_per_block);
  system_rhs.reinit(dofs_per_block);
}

// explicit instantiations
template void Solver<2>::setup_dofs();
template void Solver<3>::setup_dofs();

template void Solver<2>::setup_system_matrix
(const std::vector<types::global_dof_index> &,
 const Table<2, DoFTools::Coupling> &);
template void Solver<3>::setup_system_matrix
(const std::vector<types::global_dof_index> &,
 const Table<2, DoFTools::Coupling> &);

template void Solver<2>::setup_vectors
(const std::vector<types::global_dof_index> &);
template void Solver<3>::setup_vectors
(const std::vector<types::global_dof_index> &);

template class Solver<2>;
template class Solver<3>;

}  // namespace SolverBase
