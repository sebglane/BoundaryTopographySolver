/*
 * linear_algebra_container.h
 *
 *  Created on: Oct 7, 2021
 *      Author: sg
 */

#ifndef INCLUDE_LINEAR_ALGEBRA_CONTAINER_H_
#define INCLUDE_LINEAR_ALGEBRA_CONTAINER_H_

#include <deal.II/base/table.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <vector>

namespace SolverBase
{

using namespace dealii;

template<typename VectorType = Vector<double>,
         typename MatrixType = SparseMatrix<double>,
         typename SparsityPatternType = SparsityPattern>
struct LinearAlgebraContainer : public Subscriptor
{
  LinearAlgebraContainer(const MPI_Comm &mpi_comm=MPI_COMM_SELF);

  using vector_type = VectorType;

  template <int dim, typename ValueType>
  void setup
  (const DoFHandler<dim>              &dof_handler,
   const AffineConstraints<ValueType> &constraints,
   const Table<2, DoFTools::Coupling> &coupling_table,
   const unsigned int                  n_blocks);

  void add_to_evaluation_point(const VectorType &other,
                               const double s = 1.0);

  void add_to_present_solution(const VectorType &other,
                               const double s = 1.0);

  const IndexSet& get_locally_owned_dofs() const;

  std::vector<double> get_residual_components() const;

  void set_evaluation_point(const VectorType &other);

  void set_present_solution(const VectorType &other);

  void set_solution_update(const VectorType &other);

  void set_block(VectorType          &vector,
                 const unsigned int   block_number,
                 const double         value = 0.0);

  void distribute_constraints(VectorType &vector,
                              const AffineConstraints<double> &constraints);

private:
  const MPI_Comm  mpi_communicator;

  std::shared_ptr<VectorType> distributed_vector_ptr;

  template <int dim, typename ValueType>
  void setup_system_matrix
  (const DoFHandler<dim>              &dof_handler,
   const AffineConstraints<ValueType> &constraints,
   const Table<2, DoFTools::Coupling> &coupling_table);

  void setup_vectors();

  SparsityPatternType sparsity_pattern;

  /*!
   * @brief The set of the degrees of freedom owned by the processor.
   */
  IndexSet              locally_owned_dofs;

  /*!
   * @brief The set of the degrees of freedom that are relevant for by
   * the processor.
   */
  IndexSet              locally_relevant_dofs;

  /*!
   * @brief The set of the degrees of freedom per block owned by the processor.
   */
  std::vector<IndexSet> locally_owned_dofs_per_block;

  /*!
   * @brief The set of the degrees of freedom per block that are relevant for
   * the processor.
   */
  std::vector<IndexSet> locally_relevant_dofs_per_block;

  /*!
   * @brief
   */
  std::vector<types::global_dof_index>  dofs_per_block;

public:
  MatrixType          system_matrix;

  VectorType          evaluation_point;
  VectorType          present_solution;
  VectorType          solution_update;
  VectorType          system_rhs;
};



template <typename VectorType, typename MatrixType, typename SparsityPatternType>
inline const
IndexSet&
LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
get_locally_owned_dofs() const
{
  return (locally_owned_dofs);
}



template <typename VectorType, typename MatrixType, typename SparsityPatternType>
inline void
LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
add_to_present_solution(const VectorType &other, const double s)
{
  if (!distributed_vector_ptr)
    distributed_vector_ptr = std::make_shared<VectorType>(locally_owned_dofs,
                                                          mpi_communicator);

  VectorType &distributed_vector(*distributed_vector_ptr);
  VectorType other_distributed_vector(distributed_vector);
  distributed_vector = present_solution;
  other_distributed_vector = other;
  distributed_vector.add(s, other_distributed_vector);

  present_solution = distributed_vector;
}



template <typename VectorType, typename MatrixType, typename SparsityPatternType>
inline void
LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
add_to_evaluation_point(const VectorType &other, const double s)
{
  if (!distributed_vector_ptr)
    distributed_vector_ptr = std::make_shared<VectorType>(locally_owned_dofs,
                                                          mpi_communicator);

  VectorType &distributed_vector(*distributed_vector_ptr);
  VectorType other_distributed_vector(distributed_vector);
  distributed_vector = evaluation_point;
  other_distributed_vector = other;
  distributed_vector.add(s, other_distributed_vector);

  evaluation_point = distributed_vector;
}



template <typename VectorType, typename MatrixType, typename SparsityPatternType>
inline void
LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
set_evaluation_point(const VectorType &other)
{
  if (!distributed_vector_ptr)
    distributed_vector_ptr = std::make_shared<VectorType>(locally_owned_dofs,
                                                          mpi_communicator);

  VectorType &distributed_vector(*distributed_vector_ptr);
  distributed_vector = other;

  evaluation_point = distributed_vector;
}



template <typename VectorType, typename MatrixType, typename SparsityPatternType>
inline void
LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
set_present_solution(const VectorType &other)
{
  if (!distributed_vector_ptr)
    distributed_vector_ptr = std::make_shared<VectorType>(locally_owned_dofs,
                                                          mpi_communicator);

  VectorType &distributed_vector(*distributed_vector_ptr);
  distributed_vector = other;

  present_solution = distributed_vector;
}



template <typename VectorType, typename MatrixType, typename SparsityPatternType>
inline void LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
set_solution_update(const VectorType &other)
{
  if (!distributed_vector_ptr)
    distributed_vector_ptr = std::make_shared<VectorType>(locally_owned_dofs,
                                                          mpi_communicator);

  VectorType &distributed_vector(*distributed_vector_ptr);
  distributed_vector = other;

  solution_update = distributed_vector;
}


template <>
inline void
LinearAlgebraContainer<Vector<double>, SparseMatrix<double>, SparsityPattern>::
add_to_evaluation_point
(const Vector<double> &other, const double s)
{
  evaluation_point.add(s, other);
}



template <>
inline void
LinearAlgebraContainer<Vector<double>, SparseMatrix<double>, SparsityPattern>::
add_to_present_solution
(const Vector<double> &other, const double s)
{
  present_solution.add(s, other);
}



template <>
inline void
LinearAlgebraContainer<Vector<double>, SparseMatrix<double>, SparsityPattern>::
set_evaluation_point
(const Vector<double> &other)
{
  evaluation_point = other;
}



template <>
inline void
LinearAlgebraContainer<Vector<double>, SparseMatrix<double>, SparsityPattern>::
set_present_solution
(const Vector<double> &other)
{
  present_solution = other;
}



template <>
inline void
LinearAlgebraContainer<Vector<double>, SparseMatrix<double>, SparsityPattern>::
set_solution_update
(const Vector<double> &other)
{
  solution_update = other;
}


template <>
inline void
LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>::
add_to_evaluation_point
(const BlockVector<double> &other, const double s)
{
  evaluation_point.add(s, other);
}



template <>
inline void
LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>::
add_to_present_solution
(const BlockVector<double> &other, const double s)
{
  present_solution.add(s, other);
}



template <>
inline void
LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>::
set_evaluation_point
(const BlockVector<double> &other)
{
  evaluation_point = other;
}



template <>
inline void
LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>::
set_present_solution
(const BlockVector<double> &other)
{
  present_solution = other;
}



template <>
inline void
LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>::
set_solution_update
(const BlockVector<double> &other)
{
  solution_update = other;
}



template <>
inline std::vector<double>
LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>::
get_residual_components() const
{
  const std::size_t n_blocks{dofs_per_block.size()};

  std::vector<double> l2_norms(n_blocks, std::numeric_limits<double>::min());

  for (std::size_t i=0; i<n_blocks; ++i)
    l2_norms[i] = system_rhs.block(i).l2_norm();

  return (l2_norms);
}



template <>
inline void
LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>::
set_block
(BlockVector<double> &vector,
 const unsigned int   block_number,
 const double         value)
{
  vector.block(block_number) = value;
}




template <typename VectorType, typename MatrixType, typename SparsityPatternType>
void
LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
distribute_constraints
(VectorType &vector,
 const AffineConstraints<double> &constraints)
{
  if (!distributed_vector_ptr)
    distributed_vector_ptr = std::make_shared<VectorType>(locally_owned_dofs,
                                                          mpi_communicator);

  VectorType &distributed_vector(*distributed_vector_ptr);
  distributed_vector = vector;

  constraints.distribute(distributed_vector);

  vector = distributed_vector;
}




template <>
inline void
LinearAlgebraContainer<Vector<double>, SparseMatrix<double>, SparsityPattern>::
distribute_constraints
(Vector<double> &vector,
 const AffineConstraints<double> &constraints)
{
  constraints.distribute(vector);
}



template <>
inline void
LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>::
distribute_constraints
(BlockVector<double> &vector,
 const AffineConstraints<double> &constraints)
{
  constraints.distribute(vector);
}




}  // namespace SolverBase



#endif /* INCLUDE_LINEAR_ALGEBRA_CONTAINER_H_ */
