/*
 * block_schur_preconditioner.h
 *
 *  Created on: Apr 7, 2022
 *      Author: sg
 */

#ifndef INCLUDE_BLOCK_SCHUR_PRECONDITIONER_H_
#define INCLUDE_BLOCK_SCHUR_PRECONDITIONER_H_

#include <deal.II/lac/generic_linear_algebra.h>

namespace LA {

using namespace dealii::LinearAlgebraDealII;

}  // namespace LA


namespace Preconditioning {

using namespace dealii;

/**
 * @todo Add documentation.
 */
template <class PreconditionerTypeA, class PreconditionerTypeS>
class BlockSchurPreconditioner : public Subscriptor
{
public:
  BlockSchurPreconditioner(const LA::BlockSparseMatrix &system_matrix,
                           const LA::SparseMatrix      &schur_complement_matrix,
                           const PreconditionerTypeA   &preconditioner_A,
                           const PreconditionerTypeS   &preconditioner_S,
                           const bool                   do_solve_A,
                           const bool                   A_symmetric,
                           const bool                   S_symmetric);

  virtual void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;

  mutable unsigned int n_iterations_A;

  mutable unsigned int n_iterations_S;

private:
  const LA::BlockSparseMatrix &system_matrix;

  const LA::SparseMatrix      &schur_complement_matrix;

  const PreconditionerTypeA   &preconditioner_A;

  const PreconditionerTypeS   &preconditioner_S;

  const bool do_solve_A;

  const bool A_symmetric;

  const bool S_symmetric;
};

}  // namespace Preconditioning


#endif /* INCLUDE_BLOCK_SCHUR_PRECONDITIONER_H_ */
