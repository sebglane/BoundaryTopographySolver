/*
 * apply_boundary_conditions.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/exceptions.h>

#include <hydrodynamic_solver.h>

namespace TopographyProblem {

template <int dim>
void HydrodynamicSolver<dim>::apply_boundary_conditions()
{
  AssertThrow(velocity_boundary_conditions.closed(),
              ExcMessage("The velocity boundary conditions have not been closed."));
  AssertThrow(pressure_boundary_conditions.closed(),
              ExcMessage("The pressure boundary conditions have not been closed."));

  if (!velocity_boundary_conditions.periodic_bcs.empty() ||
      !pressure_boundary_conditions.periodic_bcs.empty())
  {
    if (!velocity_boundary_conditions.periodic_bcs.empty())
      AssertThrow(!pressure_boundary_conditions.periodic_bcs.empty(),
                  ExcMessage("No periodic boundary conditions were specified for "
                             "the pressure."));
    else if (!pressure_boundary_conditions.periodic_bcs.empty())
      AssertThrow(!velocity_boundary_conditions.periodic_bcs.empty(),
                  ExcMessage("No periodic boundary conditions were specified for "
                             "the velocity."));
    AssertDimension(velocity_boundary_conditions.periodic_bcs.size(),
                    pressure_boundary_conditions.periodic_bcs.size());

    for (std::size_t i=0; i<velocity_boundary_conditions.size(); ++i)
    {
      PeriodicBoundaryData<dim> &velocity_bc =
          pressure_boundary_conditions.periodic_bcs[i];

      bool matching_bc_found = false;

      for (std::size_t j=0; j<pressure_boundary_conditions.size(); ++j)
      {
        PeriodicBoundaryData<dim> &pressure_bc =
            pressure_boundary_conditions.periodic_bcs[j];

        if (velocity_bc.direction != pressure_bc.direction)
          continue;
        if (velocity_bc.boundary_pair.first != velocity_bc.boundary_pair.first)
          continue;
        if (velocity_bc.boundary_pair.second != velocity_bc.boundary_pair.second)
          continue;

        matching_bc_found = true;
        break;
      }

      AssertThrow(matching_bc_found == true,
                  ExcMessage("A matching periodic boundary could not be found."));
    }

    this->apply_periodicity_constraints(velocity_boundary_conditions.periodic_bcs);
  }
  {
    const FEValuesExtractors::Vector  velocity(0);

    if (!velocity_boundary_conditions.dirichlet_bcs.empty())
      this->apply_dirichlet_constraints(velocity_boundary_conditions.dirichlet_bcs,
                                        this->fe_system->component_mask(velocity));

    if (!velocity_boundary_conditions.normal_flux_bcs.empty())
      this->apply_normal_flux_constraints(velocity_boundary_conditions.normal_flux_bcs,
                                          this->fe_system->component_mask(velocity));

  }
  {
    const FEValuesExtractors::Scalar  pressure(dim);

    if (!pressure_boundary_conditions.dirichlet_bcs.empty())
      this->apply_dirichlet_constraints(pressure_boundary_conditions.dirichlet_bcs,
                                        this->fe_system->component_mask(pressure));
  }
}

}  // namespace TopographyProblem



