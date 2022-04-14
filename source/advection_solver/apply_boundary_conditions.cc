/*
 * apply_boundary_conditions.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/exceptions.h>

#include <advection_solver.h>

namespace Advection {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::apply_boundary_conditions()
{
  if (this->verbose)
    this->pcout << "    Apply boundary conditions..." << std::endl;

  AssertThrow(scalar_boundary_conditions.closed(),
              ExcMessage("The velocity boundary conditions have not been closed."));

  if (!scalar_boundary_conditions.periodic_bcs.empty())
    this->apply_periodicity_constraints(scalar_boundary_conditions.periodic_bcs,
                                        this->fe_system->component_mask(scalar_field));

  FEValuesExtractors::Scalar  scalar_field(scalar_fe_index);

  if (scalar_boundary_conditions.dirichlet_bcs.empty())
    this->apply_dirichlet_constraints(scalar_boundary_conditions.dirichlet_bcs,
                                      this->fe_system->component_mask(scalar_field));
}

// explicit instantiation
template void Solver<2>::apply_boundary_conditions();
template void Solver<3>::apply_boundary_conditions();

}  // namespace Advection
