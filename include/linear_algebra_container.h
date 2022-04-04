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

  void add(const VectorType &src,
           VectorType       &dst);

  void add(const double a,
           const VectorType &src,
           VectorType       &dst);

  void sadd(const double s,
            const VectorType &src,
            VectorType       &dst);

  void sadd(const double s,
            const double a,
            const VectorType &src,
            VectorType       &dst);

  const IndexSet& get_locally_owned_dofs() const;

  std::vector<double> get_residual_components() const;

  void setup_vector(VectorType &dst);

  void set_vector(const VectorType &src,
                  VectorType       &dst);

  void set_block(const unsigned int   block_number,
                 const double         value,
                 VectorType          &vector);

  void distribute_constraints(const AffineConstraints<double> &constraints,
                              VectorType &vector);

private:
  const MPI_Comm  mpi_communicator;

  std::shared_ptr<VectorType> distributed_vector_ptr;

  template <int dim, typename ValueType>
  void setup_system_matrix
  (const DoFHandler<dim>              &dof_handler,
   const AffineConstraints<ValueType> &constraints,
   const Table<2, DoFTools::Coupling> &coupling_table);

  void setup_system_rhs();

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
add(const VectorType &src, VectorType &dst)
{
  if (!distributed_vector_ptr)
    distributed_vector_ptr = std::make_shared<VectorType>(locally_owned_dofs,
                                                          mpi_communicator);

  VectorType &distributed_dst_vector(*distributed_vector_ptr);
  VectorType distributed_src_vector(distributed_dst_vector);
  distributed_dst_vector = dst;
  distributed_src_vector = src;
  distributed_dst_vector += distributed_src_vector;

  dst = distributed_dst_vector;
}



template <typename VectorType, typename MatrixType, typename SparsityPatternType>
inline void
LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
add(const double a, const VectorType &src, VectorType &dst)
{
  if (!distributed_vector_ptr)
    distributed_vector_ptr = std::make_shared<VectorType>(locally_owned_dofs,
                                                          mpi_communicator);

  VectorType &distributed_dst_vector(*distributed_vector_ptr);
  VectorType distributed_src_vector(distributed_dst_vector);
  distributed_dst_vector = dst;
  distributed_src_vector = src;
  distributed_dst_vector.add(a, distributed_src_vector);

  dst = distributed_dst_vector;
}



template <typename VectorType, typename MatrixType, typename SparsityPatternType>
inline void
LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
sadd(const double s, const VectorType &src, VectorType &dst)
{
  if (!distributed_vector_ptr)
    distributed_vector_ptr = std::make_shared<VectorType>(locally_owned_dofs,
                                                          mpi_communicator);

  VectorType &distributed_dst_vector(*distributed_vector_ptr);
  VectorType distributed_src_vector(distributed_dst_vector);
  distributed_dst_vector = dst;
  distributed_src_vector = src;

  distributed_dst_vector.sadd(s, distributed_src_vector);

  dst = distributed_src_vector;
}



template <typename VectorType, typename MatrixType, typename SparsityPatternType>
inline void
LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
sadd(const double a, const double b, const VectorType &src, VectorType &dst)
{
  if (!distributed_vector_ptr)
    distributed_vector_ptr = std::make_shared<VectorType>(locally_owned_dofs,
                                                          mpi_communicator);

  VectorType &distributed_dst_vector(*distributed_vector_ptr);
  VectorType distributed_src_vector(distributed_dst_vector);
  distributed_dst_vector = dst;
  distributed_src_vector = src;

  distributed_dst_vector.sadd(a, b, distributed_src_vector);

  dst = distributed_src_vector;
}



template <typename VectorType, typename MatrixType, typename SparsityPatternType>
inline void
LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
set_vector(const VectorType &src, VectorType &dst)
{
  if (!distributed_vector_ptr)
    distributed_vector_ptr = std::make_shared<VectorType>(locally_owned_dofs,
                                                          mpi_communicator);

  VectorType &distributed_vector(*distributed_vector_ptr);
  distributed_vector = src;

  dst = distributed_vector;
}



template <typename VectorType, typename MatrixType, typename SparsityPatternType>
void
LinearAlgebraContainer<VectorType, MatrixType, SparsityPatternType>::
distribute_constraints
(const AffineConstraints<double> &constraints,
 VectorType &vector)
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
add
(const Vector<double> &src, Vector<double> &dst)
{
  dst += src;
}



template <>
inline void
LinearAlgebraContainer<Vector<double>, SparseMatrix<double>, SparsityPattern>::
add
(const double s, const Vector<double> &src, Vector<double> &dst)
{
  dst.add(s, src);
}



template <>
inline void
LinearAlgebraContainer<Vector<double>, SparseMatrix<double>, SparsityPattern>::
sadd
(const double s, const Vector<double> &src, Vector<double> &dst)
{
  dst.sadd(s, src);
}



template <>
inline void
LinearAlgebraContainer<Vector<double>, SparseMatrix<double>, SparsityPattern>::
sadd
(const double a, const double b, const Vector<double> &src, Vector<double> &dst)
{
  dst.sadd(a, b, src);
}



template <>
inline void
LinearAlgebraContainer<Vector<double>, SparseMatrix<double>, SparsityPattern>::
set_vector
(const Vector<double>  &src,
 Vector<double>        &dst)
{
  dst = src;
}



template <>
inline void
LinearAlgebraContainer<Vector<double>, SparseMatrix<double>, SparsityPattern>::
distribute_constraints
(const AffineConstraints<double> &constraints,
 Vector<double> &vector)
{
  constraints.distribute(vector);
}



template <>
inline void
LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>::
add
(const BlockVector<double> &src, BlockVector<double> &dst)
{
  dst += src;
}



template <>
inline void
LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>::
add
(const double s, const BlockVector<double> &src, BlockVector<double> &dst)
{
  dst.add(s, src);
}



template <>
inline void
LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>::
sadd
(const double s, const BlockVector<double> &src, BlockVector<double> &dst)
{
  dst.sadd(s, src);
}



template <>
inline void
LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>::
sadd
(const double a, const double b, const BlockVector<double> &src, BlockVector<double> &dst)
{
  dst.sadd(a, b, src);
}



template <>
inline void
LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>::
set_vector
(const BlockVector<double>  &src,
 BlockVector<double>        &dst)
{
  dst = src;
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
(const unsigned int   block_number,
 const double         value,
 BlockVector<double> &vector)
{
  vector.block(block_number) = value;
}



template <>
inline void
LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>::
distribute_constraints
(const AffineConstraints<double> &constraints,
 BlockVector<double> &vector)
{
  constraints.distribute(vector);
}

}  // namespace SolverBase



#endif /* INCLUDE_LINEAR_ALGEBRA_CONTAINER_H_ */
