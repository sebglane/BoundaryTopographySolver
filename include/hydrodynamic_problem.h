/*
 * hydrodynamic_problem.h
 *
 *  Created on: Sep 1, 2021
 *      Author: sg
 */

#ifndef INCLUDE_HYDRODYNAMIC_PROBLEM_H_
#define INCLUDE_HYDRODYNAMIC_PROBLEM_H_

#include <deal.II/fe/mapping_q_cache.h>

#include <hydrodynamic_solver.h>

namespace TopographyProblem {

template <int dim>
class HydrodynamicProblem
{
public:
  HydrodynamicProblem();

  void run();

protected:
  virtual void make_grid() = 0;

  void initialize_mapping();

  virtual void set_boundary_conditions() = 0;

  virtual void set_body_force_term();

  Triangulation<dim>       triangulation;

  MappingQCache<dim>       mapping;

  HydrodynamicSolver<dim>  solver;

};

// inline functions
template <int dim>
void HydrodynamicProblem<dim>::set_body_force_term()
{
  return;
}

}  // namespace TopographyProblem



#endif /* INCLUDE_HYDRODYNAMIC_PROBLEM_H_ */
