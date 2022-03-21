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

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/trilinos_vector.h>

namespace SolverBase {

using namespace dealii;

template <int dim>
class EvaluationBase
{
public:
  virtual ~EvaluationBase() = default;

  void set_cycle(const unsigned int cycle);

  virtual void operator()(const Mapping<dim>        &mapping,
                          const FiniteElement<dim>  &fe,
                          const DoFHandler<dim>     &dof_handler,
                          const Vector<double>      &solution) = 0;

  virtual void operator()(const Mapping<dim>        &mapping,
                          const FiniteElement<dim>  &fe,
                          const DoFHandler<dim>     &dof_handler,
                          const BlockVector<double> &solution) = 0;

  virtual void operator()(const Mapping<dim>        &mapping,
                          const FiniteElement<dim>  &fe,
                          const DoFHandler<dim>     &dof_handler,
                          const TrilinosWrappers::MPI::Vector &solution) = 0;

protected:
  unsigned int cycle;
};

// inline functions
template <int dim>
void EvaluationBase<dim>::set_cycle(const unsigned int current_cycle)
{
  cycle = current_cycle;
}

}  // namespace SolverBase

#endif /* INCLUDE_EVALUATION_BASE_H_ */
