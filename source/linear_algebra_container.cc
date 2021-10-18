/*
 * linear_algebra_container.cc
 *
 *  Created on: Oct 7, 2021
 *      Author: sg
 */

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>

#include <linear_algebra_container.h>

#include <iterator>
#include <numeric>

namespace SolverBase
{


template <typename VectorType, typename MatrixType, typename SparsityPatternType>
LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
LinearAlgebraContainer(const MPI_Comm &mpi_comm)
:
mpi_communicator(mpi_comm),
distributed_vector_ptr()
{}


template <typename VectorType, typename MatrixType, typename SparsityPatternType>
template <int dim, typename ValueType>
void LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::setup
(const DoFHandler<dim>              &dof_handler,
 const AffineConstraints<ValueType> &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table,
 const unsigned int                  n_blocks)
{
  std::cout << "    Setup linear algebra container..." << std::endl;

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

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  for (unsigned int i=0; i<dofs_per_block.size(); ++i)
    locally_owned_dofs_per_block.push_back(locally_owned_dofs.get_view(accumulated_dofs_per_block[i],
                                                                       accumulated_dofs_per_block[i+1]));

  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  for (unsigned int i=0; i<dofs_per_block.size(); ++i)
    locally_relevant_dofs_per_block.push_back(locally_relevant_dofs.
                                              get_view(accumulated_dofs_per_block[i],
                                                       accumulated_dofs_per_block[i+1]));

  setup_system_matrix(dof_handler, constraints, coupling_table);
  setup_vectors();
}



template <typename VectorType, typename MatrixType, typename SparsityPatternType>
template <int dim, typename ValueType>
void LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
setup_system_matrix
(const DoFHandler<dim>               &dof_handler,
 const AffineConstraints<ValueType>  &constraints,
 const Table<2, DoFTools::Coupling>  &coupling_table)
{
  system_matrix.clear();

  DynamicSparsityPattern dsp(locally_relevant_dofs);

  DoFTools::make_sparsity_pattern(dof_handler,
                                  coupling_table,
                                  dsp,
                                  constraints,
                                  false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);
  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       mpi_communicator);
}




template <>
template <>
void LinearAlgebraContainer<BlockVector<double>,
                            BlockSparseMatrix<double>,
                            BlockSparsityPattern>::setup_system_matrix<2, double>
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
void LinearAlgebraContainer<BlockVector<double>,
                            BlockSparseMatrix<double>,
                            BlockSparsityPattern>::setup_system_matrix<3, double>
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
void LinearAlgebraContainer<Vector<double>,
                            SparseMatrix<double>,
                            SparsityPattern>::setup_system_matrix<2, double>
(const DoFHandler<2>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table)
{
  system_matrix.clear();

  DynamicSparsityPattern dsp(locally_owned_dofs);

  DoFTools::make_sparsity_pattern(dof_handler,
                                  coupling_table,
                                  dsp,
                                  constraints,
                                  false);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
}




template <>
template <>
void LinearAlgebraContainer<Vector<double>,
                            SparseMatrix<double>,
                            SparsityPattern>::setup_system_matrix<3, double>
(const DoFHandler<3>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table)
{
  system_matrix.clear();

  DynamicSparsityPattern dsp(locally_owned_dofs);

  DoFTools::make_sparsity_pattern(dof_handler,
                                  coupling_table,
                                  dsp,
                                  constraints,
                                  false);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
}



template <typename VectorType, typename MatrixType, typename SparsityPatternType>
void LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
setup_vectors()
{
  evaluation_point.reinit(locally_relevant_dofs,
                          mpi_communicator);
  present_solution.reinit(evaluation_point);
  solution_update.reinit(evaluation_point);

  system_rhs.reinit(locally_owned_dofs,
                    locally_relevant_dofs,
                    mpi_communicator,
                    true);
}



template <>
void LinearAlgebraContainer<BlockVector<double>,
                            BlockSparseMatrix<double>,
                            BlockSparsityPattern>::setup_vectors()
{
  evaluation_point.reinit(dofs_per_block);
  present_solution.reinit(dofs_per_block);
  solution_update.reinit(dofs_per_block);

  system_rhs.reinit(dofs_per_block);
}



template <>
void LinearAlgebraContainer<Vector<double>,
                            SparseMatrix<double>,
                            SparsityPattern>::setup_vectors()
{
  evaluation_point.reinit(*dofs_per_block.rbegin());
  present_solution.reinit(*dofs_per_block.rbegin());
  solution_update.reinit(*dofs_per_block.rbegin());

  system_rhs.reinit(*dofs_per_block.rbegin());
}



// explicit instantiations
template void LinearAlgebraContainer<TrilinosWrappers::MPI::Vector,
                                     TrilinosWrappers::SparseMatrix,
                                     TrilinosWrappers::SparsityPattern>::
setup_system_matrix<2, double>
(const DoFHandler<2>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table);

template void LinearAlgebraContainer<TrilinosWrappers::MPI::Vector,
                                     TrilinosWrappers::SparseMatrix,
                                     TrilinosWrappers::SparsityPattern>::
setup_system_matrix<3, double>
(const DoFHandler<3>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table);

template void LinearAlgebraContainer<Vector<double>,
                                     SparseMatrix<double>,
                                     SparsityPattern>::setup<2, double>
(const DoFHandler<2>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table,
 const unsigned int                  n_blocks);

template void LinearAlgebraContainer<Vector<double>,
                                     SparseMatrix<double>,
                                     SparsityPattern>::setup<3, double>
(const DoFHandler<3>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table,
 const unsigned int                  n_blocks);

template void LinearAlgebraContainer<BlockVector<double>,
                                     BlockSparseMatrix<double>,
                                     BlockSparsityPattern>::setup<2, double>
(const DoFHandler<2>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table,
 const unsigned int                  n_blocks);

template void LinearAlgebraContainer<BlockVector<double>,
                                     BlockSparseMatrix<double>,
                                     BlockSparsityPattern>::setup<3, double>
(const DoFHandler<3>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table,
 const unsigned int                  n_blocks);

template void LinearAlgebraContainer<TrilinosWrappers::MPI::Vector,
                                     TrilinosWrappers::SparseMatrix,
                                     TrilinosWrappers::SparsityPattern>::
setup<2, double>
(const DoFHandler<2>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table,
 const unsigned int                  n_blocks);

template void LinearAlgebraContainer<TrilinosWrappers::MPI::Vector,
                                     TrilinosWrappers::SparseMatrix,
                                     TrilinosWrappers::SparsityPattern>::
setup<3, double>
(const DoFHandler<3>                &dof_handler,
 const AffineConstraints<double>    &constraints,
 const Table<2, DoFTools::Coupling> &coupling_table,
 const unsigned int                  n_blocks);

template struct LinearAlgebraContainer<Vector<double>,
                                       SparseMatrix<double>,
                                       SparsityPattern>;
template struct LinearAlgebraContainer<BlockVector<double>,
                                       BlockSparseMatrix<double>,
                                       BlockSparsityPattern>;
template struct LinearAlgebraContainer<TrilinosWrappers::MPI::Vector,
                                       TrilinosWrappers::SparseMatrix,
                                       TrilinosWrappers::SparsityPattern>;

}  // namespace SolverBase


