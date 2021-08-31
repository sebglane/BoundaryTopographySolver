/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <hydrodynamic_solver.h>

namespace TopographyProblem {

template <int dim>
HydrodynamicSolver<dim>::HydrodynamicSolver
(const double       reynolds_number,
 const unsigned int velocity_fe_degree,
 const unsigned int n_refinements,
 const double       newton_tolerance,
 const unsigned int n_maximum_iterations)
:
SolverBase<dim>(n_refinements, newton_tolerance, n_maximum_iterations),
velocity_fe_degree(velocity_fe_degree),
reynolds_number(reynolds_number)
{}

// explicit instantiation
template HydrodynamicSolver<2>::HydrodynamicSolver
(const double       ,
 const unsigned int ,
 const unsigned int ,
 const double       ,
 const unsigned int );
template HydrodynamicSolver<3>::HydrodynamicSolver
(const double       ,
 const unsigned int ,
 const unsigned int ,
 const double       ,
 const unsigned int );

template class HydrodynamicSolver<2>;
template class HydrodynamicSolver<3>;

}  // namespace TopographyProblem
