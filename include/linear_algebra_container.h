/*
 * linear_algebra_container.h
 *
 *  Created on: Oct 7, 2021
 *      Author: sg
 */

#ifndef INCLUDE_LINEAR_ALGEBRA_CONTAINER_H_
#define INCLUDE_LINEAR_ALGEBRA_CONTAINER_H_

#include <deal.II/base/table.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>

#include <vector>

namespace SolverBase
{

using namespace dealii;

template<typename VectorType = BlockVector<double>, typename MatrixType = BlockSparseMatrix<double>, typename SparsityPatternType = BlockSparsityPattern>
struct LinearAlgebraContainer
{
  LinearAlgebraContainer(const MPI_Comm &mpi_comm=MPI_COMM_SELF);

  using vector_type = VectorType;

  template<int dim, typename ValueType>
  void setup
  (const DoFHandler<dim>              &dof_handler,
   const AffineConstraints<ValueType> &constraints,
   const Table<2, DoFTools::Coupling> &coupling_table,
   const unsigned int                  n_blocks);

  inline void add_to_evaluation_point(const VectorType &other,
                                      const double s = 1.0);

  inline void add_to_present_solution(const VectorType &other,
                                      const double s = 1.0);

  inline void set_evaluation_point(const VectorType &other);

  inline void set_present_solution(const VectorType &other);

  inline void set_solution_update(const VectorType &other);

  MatrixType          system_matrix;

  VectorType          evaluation_point;
  VectorType          present_solution;
  VectorType          solution_update;
  VectorType          system_rhs;

private:
  const MPI_Comm  &mpi_communicator;

  template<int dim, typename ValueType>
  void setup_system_matrix
  (const DoFHandler<dim>              &dof_handler,
   const AffineConstraints<ValueType> &constraints,
   const Table<2, DoFTools::Coupling> &coupling_table);

  void setup_vectors();

  SparsityPatternType sparsity_pattern;

  /*!
   * @brief The set of the degrees of freedom owned by the processor.
   */
  std::vector<IndexSet> locally_owned_dofs_per_block;

  /*!
   * @brief The set of the degrees of freedom that are relevant for
   * the processor.
   */
  std::vector<IndexSet> locally_relevant_dofs_per_block;

  /*!
   * @brief
   */
  std::vector<types::global_dof_index>  dofs_per_block;
};



template<>
inline void LinearAlgebraContainer<>::add_to_evaluation_point
(const BlockVector<double> &other, const double s)
{
  evaluation_point.add(s, other);
}



template<>
inline void LinearAlgebraContainer<>::add_to_present_solution
(const BlockVector<double> &other, const double s)
{
  present_solution.add(s, other);
}



template<>
inline void LinearAlgebraContainer<>::set_evaluation_point
(const BlockVector<double> &other)
{
  evaluation_point = other;
}



template<>
inline void LinearAlgebraContainer<>::set_present_solution
(const BlockVector<double> &other)
{
  present_solution = other;
}



template<>
inline void LinearAlgebraContainer<>::set_solution_update
(const BlockVector<double> &other)
{
  solution_update = other;
}

}  // namespace SolverBase



#endif /* INCLUDE_LINEAR_ALGEBRA_CONTAINER_H_ */
