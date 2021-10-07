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
  LinearAlgebraContainer();

  template<int dim, typename ValueType>
  void setup_system_matrix
  (const DoFHandler<dim>                      &dof_handler,
   const AffineConstraints<ValueType>         &constraints,
   const std::vector<types::global_dof_index> &dofs_per_block,
   const Table<2, DoFTools::Coupling>         &coupling_table);

  void setup_vectors(const std::vector<types::global_dof_index> &dofs_per_block);

  SparsityPatternType sparsity_pattern;

  MatrixType          system_matrix;

  VectorType          evaluation_point;
  VectorType          present_solution;
  VectorType          solution_update;
  VectorType          system_rhs;
};

}  // namespace SolverBase



#endif /* INCLUDE_LINEAR_ALGEBRA_CONTAINER_H_ */