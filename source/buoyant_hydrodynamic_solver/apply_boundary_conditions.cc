/*
 * apply_boundary_conditions.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/exceptions.h>

#include <buoyant_hydrodynamic_solver.h>

namespace BuoyantHydrodynamic {

template <int dim>
void Solver<dim>::apply_boundary_conditions()
{
  if (this->verbose)
    std::cout << "    Apply boundary conditions..." << std::endl;

  AssertThrow(this->velocity_boundary_conditions.closed(),
              ExcMessage("The velocity boundary conditions have not been closed."));
  AssertThrow(this->pressure_boundary_conditions.closed(),
              ExcMessage("The pressure boundary conditions have not been closed."));

  if (!this->velocity_boundary_conditions.periodic_bcs.empty() ||
      !this->pressure_boundary_conditions.periodic_bcs.empty() ||
      !density_boundary_conditions.periodic_bcs.empty())
  {
    AssertThrow(this->velocity_boundary_conditions.periodic_bcs.empty(),
                ExcMessage("No periodic boundary conditions were specified for "
                           "the velocity."));

    AssertThrow(this->pressure_boundary_conditions.periodic_bcs.empty(),
                ExcMessage("No periodic boundary conditions were specified for "
                           "the pressure."));

    AssertThrow(density_boundary_conditions.periodic_bcs.empty(),
                ExcMessage("No periodic boundary conditions were specified for "
                           "the density."));

    AssertDimension(this->velocity_boundary_conditions.periodic_bcs.size(),
                    this->pressure_boundary_conditions.periodic_bcs.size());

    AssertDimension(this->velocity_boundary_conditions.periodic_bcs.size(),
                    density_boundary_conditions.periodic_bcs.size());

    for (std::size_t i=0; i<this->velocity_boundary_conditions.periodic_bcs.size(); ++i)
    {
      const PeriodicBoundaryData<dim> &velocity_bc =
          this->velocity_boundary_conditions.periodic_bcs[i];

      bool matching_bc_found = false;

      for (std::size_t j=0; j<this->pressure_boundary_conditions.periodic_bcs.size(); ++j)
      {
        const PeriodicBoundaryData<dim> &pressure_bc =
            this->pressure_boundary_conditions.periodic_bcs[j];

        if (velocity_bc.direction != pressure_bc.direction)
          continue;
        if (velocity_bc.boundary_pair.first != pressure_bc.boundary_pair.first)
          continue;
        if (velocity_bc.boundary_pair.second != pressure_bc.boundary_pair.second)
          continue;

        matching_bc_found = true;
        break;
      }

      AssertThrow(matching_bc_found == true,
                  ExcMessage("A matching periodic boundary could not be found."));

      matching_bc_found = false;
      for (std::size_t j=0; j<density_boundary_conditions.periodic_bcs.size(); ++j)
      {
        const PeriodicBoundaryData<dim> &density_bc =
            density_boundary_conditions.periodic_bcs[j];

        if (velocity_bc.direction != density_bc.direction)
          continue;
        if (velocity_bc.boundary_pair.first != density_bc.boundary_pair.first)
          continue;
        if (velocity_bc.boundary_pair.second != density_bc.boundary_pair.second)
          continue;

        matching_bc_found = true;
        break;
      }
      AssertThrow(matching_bc_found == true,
                  ExcMessage("A matching periodic boundary could not be found."));

    }

    this->apply_periodicity_constraints(this->velocity_boundary_conditions.periodic_bcs);
  }
  {
    const FEValuesExtractors::Vector  velocity(0);

    if (!this->velocity_boundary_conditions.dirichlet_bcs.empty())
      this->apply_dirichlet_constraints(this->velocity_boundary_conditions.dirichlet_bcs,
                                        this->fe_system->component_mask(velocity));

    if (!this->velocity_boundary_conditions.normal_flux_bcs.empty())
      this->apply_normal_flux_constraints(this->velocity_boundary_conditions.normal_flux_bcs,
                                          this->fe_system->component_mask(velocity));
  }
  {
    const FEValuesExtractors::Scalar  pressure(dim);

    if (!this->pressure_boundary_conditions.dirichlet_bcs.empty())
      this->apply_dirichlet_constraints(this->pressure_boundary_conditions.dirichlet_bcs,
                                        this->fe_system->component_mask(pressure));
  }
}

// explicit instantiation
template void Solver<2>::apply_boundary_conditions();
template void Solver<3>::apply_boundary_conditions();

}  // namespace BuoyantHydrodynamic