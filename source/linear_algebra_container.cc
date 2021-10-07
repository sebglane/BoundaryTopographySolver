/*
 * linear_algebra_container.cc
 *
 *  Created on: Oct 7, 2021
 *      Author: sg
 */

#include <linear_algebra_container.h>

namespace SolverBase
{


template <typename VectorType, typename MatrixType, typename SparsityPatternType>
LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::LinearAlgebraContainer()
{}



template <>
template <>
void LinearAlgebraContainer<>::setup_system_matrix<2, double>
(const DoFHandler<2>              &dof_handler,
 const AffineConstraints<double> &constraints,
 const std::vector<types::global_dof_index> &dofs_per_block,
 const Table<2, DoFTools::Coupling> &coupling_table)
{
  system_matrix.clear();

  BlockDynamicSparsityPattern dsp(dofs_per_block,
                                  dofs_per_block);

  DoFTools::make_sparsity_pattern(dof_handler,
                                  coupling_table,
                                  dsp,
                                  constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
}



template <>
template <>
void LinearAlgebraContainer<>::setup_system_matrix<3, double>
(const DoFHandler<3>              &dof_handler,
 const AffineConstraints<double> &constraints,
 const std::vector<types::global_dof_index> &dofs_per_block,
 const Table<2, DoFTools::Coupling> &coupling_table)
{
  system_matrix.clear();

  BlockDynamicSparsityPattern dsp(dofs_per_block,
                                  dofs_per_block);

  DoFTools::make_sparsity_pattern(dof_handler,
                                  coupling_table,
                                  dsp,
                                  constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
}



template <>
void LinearAlgebraContainer<>::setup_vectors
(const std::vector<types::global_dof_index> &dofs_per_block)
{
  evaluation_point.reinit(dofs_per_block);
  present_solution.reinit(dofs_per_block);
  solution_update.reinit(dofs_per_block);
  system_rhs.reinit(dofs_per_block);
}

// explicit instantiations
template struct LinearAlgebraContainer<>;

}  // namespace SolverBase


