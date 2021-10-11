/*
 * linear_algebra_container.cc
 *
 *  Created on: Oct 7, 2021
 *      Author: sg
 */

#include <linear_algebra_container.h>

#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>

#include <iterator>
#include <numeric>

namespace SolverBase
{


template <typename VectorType, typename MatrixType, typename SparsityPatternType>
LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
LinearAlgebraContainer(const MPI_Comm &mpi_comm)
:
mpi_communicator(mpi_comm)
{}


template <typename VectorType, typename MatrixType, typename SparsityPatternType>
template <int dim, typename ValueType>
void LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::setup
(const DoFHandler<dim>              &dof_handler,
 const AffineConstraints<ValueType> &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table,
 const unsigned int                  n_blocks)
{
  dofs_per_block.resize(n_blocks);
  dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler);

  std::vector<types::global_dof_index> accumulated_dofs_per_block(dofs_per_block.size() + 1, 0);
  {
    auto it = accumulated_dofs_per_block.begin();
    std::advance(it, 1);
    std::partial_sum(dofs_per_block.begin(),
                     dofs_per_block.end(),
                     it);
  }


  IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  for (unsigned int i=0; i<dofs_per_block.size(); ++i)
    locally_owned_dofs_per_block.push_back(locally_owned_dofs.get_view(accumulated_dofs_per_block[i],
                                                                       accumulated_dofs_per_block[i+1]));

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  for (unsigned int i=0; i<dofs_per_block.size(); ++i)
    locally_relevant_dofs_per_block.push_back(locally_relevant_dofs.get_view(accumulated_dofs_per_block[i],
                                                                             accumulated_dofs_per_block[i+1]));

  setup_system_matrix(dof_handler, constraints, coupling_table);
  setup_vectors();
}



template <>
template <>
void LinearAlgebraContainer<>::setup_system_matrix<2, double>
(const DoFHandler<2>                &dof_handler,
 const AffineConstraints<double>    &constraints,
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
(const DoFHandler<3>                &dof_handler,
 const AffineConstraints<double>    &constraints,
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
void LinearAlgebraContainer<TrilinosWrappers::MPI::BlockVector,
                            TrilinosWrappers::BlockSparseMatrix,
                            TrilinosWrappers::BlockSparsityPattern>::
setup_system_matrix<2, double>
(const DoFHandler<2>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table)
{
  system_matrix.clear();

  sparsity_pattern.reinit(locally_owned_dofs_per_block,
                          locally_owned_dofs_per_block,
                          locally_relevant_dofs_per_block,
                          mpi_communicator);

  DoFTools::make_sparsity_pattern(dof_handler,
                                  coupling_table,
                                  sparsity_pattern,
                                  constraints,
                                  false,
                                  Utilities::MPI::this_mpi_process(mpi_communicator));
  sparsity_pattern.compress();

  system_matrix.reinit(sparsity_pattern);
}



template <>
template <>
void LinearAlgebraContainer<TrilinosWrappers::MPI::BlockVector,
                            TrilinosWrappers::BlockSparseMatrix,
                            TrilinosWrappers::BlockSparsityPattern>::
setup_system_matrix<3, double>
(const DoFHandler<3>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table)
{
  system_matrix.clear();

  sparsity_pattern.reinit(locally_owned_dofs_per_block,
                          locally_owned_dofs_per_block,
                          locally_relevant_dofs_per_block,
                          mpi_communicator);

  DoFTools::make_sparsity_pattern(dof_handler,
                                  coupling_table,
                                  sparsity_pattern,
                                  constraints,
                                  false,
                                  Utilities::MPI::this_mpi_process(mpi_communicator));
  sparsity_pattern.compress();

  system_matrix.reinit(sparsity_pattern);
}



template <>
void LinearAlgebraContainer<>::setup_vectors()
{
  evaluation_point.reinit(dofs_per_block);
  present_solution.reinit(dofs_per_block);
  solution_update.reinit(dofs_per_block);
  system_rhs.reinit(dofs_per_block);
}



template <>
void LinearAlgebraContainer<TrilinosWrappers::MPI::BlockVector,
                            TrilinosWrappers::BlockSparseMatrix,
                            TrilinosWrappers::BlockSparsityPattern>::setup_vectors()
{
  evaluation_point.reinit(locally_relevant_dofs_per_block, mpi_communicator);
  present_solution.reinit(evaluation_point);
  solution_update.reinit(evaluation_point);
  system_rhs.reinit(locally_owned_dofs_per_block,
                    locally_relevant_dofs_per_block,
                    mpi_communicator,
                    true);
}

// explicit instantiations
template void LinearAlgebraContainer<>::setup<2, double>
(const DoFHandler<2>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table,
 const unsigned int                  n_blocks);
template void LinearAlgebraContainer<>::setup<3, double>
(const DoFHandler<3>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table,
 const unsigned int                  n_blocks);

template void LinearAlgebraContainer<TrilinosWrappers::MPI::BlockVector,
                                     TrilinosWrappers::BlockSparseMatrix,
                                     TrilinosWrappers::BlockSparsityPattern>::
setup<2, double>
(const DoFHandler<2>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table,
 const unsigned int                  n_blocks);
template void LinearAlgebraContainer<TrilinosWrappers::MPI::BlockVector,
                                     TrilinosWrappers::BlockSparseMatrix,
                                     TrilinosWrappers::BlockSparsityPattern>::
setup<3, double>
(const DoFHandler<3>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table,
 const unsigned int                  n_blocks);

template struct LinearAlgebraContainer<>;
template struct LinearAlgebraContainer<TrilinosWrappers::MPI::BlockVector,
                                       TrilinosWrappers::BlockSparseMatrix,
                                       TrilinosWrappers::BlockSparsityPattern>;

}  // namespace SolverBase


