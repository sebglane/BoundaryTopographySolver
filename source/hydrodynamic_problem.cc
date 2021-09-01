/*
 * hydrodynamic_problem.cc
 *
 *  Created on: Sep 1, 2021
 *      Author: sg
 */

#include <hydrodynamic_problem.h>

namespace TopographyProblem {

template <int dim>
HydrodynamicProblem<dim>::HydrodynamicProblem()
:
mapping(2),
solver(triangulation, mapping, 1.0)
{}



template <int dim>
void HydrodynamicProblem<dim>::initialize_mapping()
{
  std::cout << "    Initialize mapping..." << std::endl;

  mapping.initialize(triangulation, MappingQGeneric<dim>(2));
}



template <int dim>
void HydrodynamicProblem<dim>::run()
{
  this->make_grid();

  initialize_mapping();

  this->set_boundary_conditions();

  this->set_body_force_term();

  solver.solve();
}


template HydrodynamicProblem<2>::HydrodynamicProblem();
template HydrodynamicProblem<3>::HydrodynamicProblem();

template void HydrodynamicProblem<2>::initialize_mapping();
template void HydrodynamicProblem<3>::initialize_mapping();

template void HydrodynamicProblem<2>::run();
template void HydrodynamicProblem<3>::run();

template class HydrodynamicProblem<2>;
template class HydrodynamicProblem<3>;


}  // namespace TopographyProblem
