/*
 * buoyant_hydrodynamic_solver.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_BUOYANT_HYDRODYNAMIC_SOLVER_H_
#define INCLUDE_BUOYANT_HYDRODYNAMIC_SOLVER_H_


#include <hydrodynamic_solver.h>

namespace BuoyantHydrodynamic {

using namespace BoundaryConditions;

template <int dim>
class Solver: public Hydrodynamic::Solver<dim>
{

public:
  Solver(Triangulation<dim>  &tria,
         Mapping<dim>        &mapping,
         const Hydrodynamic::SolverParameters &parameters,
         const double         reynolds_number = 1.0,
         const double         froude_number = 0.0,
         const double         stratification_number = 1.0);

  void set_reference_density(const Function<dim> &reference_density);

  ScalarBoundaryConditions<dim>&  get_density_bcs();

  const ScalarBoundaryConditions<dim>&  get_density_bcs() const;

  double get_stratification_number() const;

private:
  virtual void setup_fe_system();

  virtual void setup_dofs();

  virtual void apply_boundary_conditions();

  virtual void assemble_system(const bool initial_step);

  virtual void assemble_rhs(const bool initial_step);

  virtual void output_results(const unsigned int cycle = 0) const;

  ScalarBoundaryConditions<dim> density_boundary_conditions;

  const Function<dim>          *reference_density_ptr;

  const TensorFunction<1, dim> *gravity_field_ptr;

  const double        stratification_number;

  const unsigned int  density_fe_degree;
};

// inline functions
template <int dim>
inline const ScalarBoundaryConditions<dim> &
Solver<dim>::get_density_bcs() const
{
  return density_boundary_conditions;
}



template <int dim>
inline ScalarBoundaryConditions<dim> &
Solver<dim>::get_density_bcs()
{
  return density_boundary_conditions;
}

}  // namespace TopographyProblem



#endif /* INCLUDE_BUOYANT_HYDRODYNAMIC_SOLVER_H_ */
