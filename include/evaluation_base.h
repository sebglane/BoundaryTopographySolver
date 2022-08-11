/*
 * evaluation_base.h
 *
 *  Created on: Sep 1, 2021
 *      Author: sg
 */

#ifndef INCLUDE_EVALUATION_BASE_H_
#define INCLUDE_EVALUATION_BASE_H_

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe.h>

#include <deal.II/lac/block_vector.h>

namespace Base {

using namespace dealii;

/**
 * @todo Add documentation.
 */
template <int dim, typename VectorType = BlockVector<double> >
class EvaluationBase
{
public:
  virtual ~EvaluationBase() = default;

  void set_cycle(const unsigned int cycle);

  virtual void operator()(const Mapping<dim>        &mapping,
                          const FiniteElement<dim>  &fe,
                          const DoFHandler<dim>     &dof_handler,
                          const VectorType          &solution) = 0;

protected:
  unsigned int cycle;
};

// inline functions
template <int dim, typename VectorType>
void EvaluationBase<dim, VectorType>::set_cycle(const unsigned int current_cycle)
{
  cycle = current_cycle;
}

}  // namespace SolverBase

#endif /* INCLUDE_EVALUATION_BASE_H_ */
