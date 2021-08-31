/*
 * hydrodynamic_solver.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_HYDRODYNAMIC_SOLVER_H_
#define INCLUDE_HYDRODYNAMIC_SOLVER_H_

#include <solver_base.h>

namespace TopographyProblem {

template <int dim>
class HydrodynamicSolver: public SolverBase<dim>
{

public:
  HydrodynamicSolver(const Triangulation<dim>  &tria,
                     const double               reynolds_number,
                     const unsigned int         velocity_fe_degree = 2,
                     const unsigned int         n_refinements = 3,
                     const double               newton_tolerance = 1e-9,
                     const unsigned int         n_maximum_iterations = 10);

private:
  virtual void setup_fe_system();

  virtual void setup_dofs();

  virtual void assemble_system(const bool initial_step);

  virtual void assemble_rhs(const bool initial_step);

  virtual void output_results(const unsigned int cycle = 0) const;

  const unsigned int  velocity_fe_degree;

  const double        reynolds_number;

};

}  // namespace TopographyProblem

#endif /* INCLUDE_HYDRODYNAMIC_SOLVER_H_ */
