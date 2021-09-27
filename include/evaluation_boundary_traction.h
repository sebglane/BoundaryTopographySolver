/*
 * evaluation_boundary_traction.h
 *
 *  Created on: Sep 27, 2021
 *      Author: sg
 */

#ifndef INCLUDE_EVALUATION_BOUNDARY_TRACTION_H_
#define INCLUDE_EVALUATION_BOUNDARY_TRACTION_H_

#include <deal.II/base/table_handler.h>

#include <evaluation_base.h>

namespace Hydrodynamic {

using namespace dealii;

template <int dim>
class EvaluationBoundaryTraction : public SolverBase::EvaluationBase<dim>
{
public:
  EvaluationBoundaryTraction(const unsigned int velocity_start_index,
                             const unsigned int pressure_index,
                             const double reynolds_number);
  ~EvaluationBoundaryTraction();

  void set_boundary_id(const types::boundary_id boundary_id);

  virtual void operator()(const Mapping<dim>        &mapping,
                          const FiniteElement<dim>  &fe,
                          const DoFHandler<dim>     &dof_handler,
                          const Vector<double>      &solution) override;

  virtual void operator()(const Mapping<dim>        &mapping,
                          const FiniteElement<dim>  &fe,
                          const DoFHandler<dim>     &dof_handler,
                          const BlockVector<double> &solution) override;

private:
  TableHandler  traction_table;
  TableHandler  pressure_table;
  TableHandler  viscous_table;

  types::boundary_id  boundary_id;

  const unsigned int velocity_start_index;

  const unsigned int pressure_index;

  const double       reynolds_number;
};

// inline functions
template <int dim>
inline void EvaluationBoundaryTraction<dim>::set_boundary_id(const types::boundary_id bndry_id)
{
  boundary_id = bndry_id;
}

}  // namespace Hydrodynamic

#endif /* INCLUDE_EVALUATION_BOUNDARY_TRACTION_H_ */
