/*
 * apply_boundary_conditions.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/exceptions.h>

#include <advection_solver.h>

namespace Advection {

template <int dim, typename TriangulationType, typename VectorType, typename MatrixType >
void Solver<dim, TriangulationType, VectorType, MatrixType>::apply_boundary_conditions()
{
  if (this->verbose)
    std::cout << "    Apply boundary conditions..." << std::endl;

  AssertThrow(boundary_conditions.closed(),
              ExcMessage("The velocity boundary conditions have not been closed."));

  if (!boundary_conditions.periodic_bcs.empty())
    this->apply_periodicity_constraints(boundary_conditions.periodic_bcs);

  const FEValuesExtractors::Scalar  field(0);

  if (boundary_conditions.dirichlet_bcs.empty())
    this->apply_dirichlet_constraints(boundary_conditions.dirichlet_bcs,
                                      this->fe_system->component_mask(field));
}

// explicit instantiation
template void Solver<2>::apply_boundary_conditions();
template void Solver<3>::apply_boundary_conditions();

}  // namespace Advection
