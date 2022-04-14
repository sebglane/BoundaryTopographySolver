/*
 * apply_boundary_conditions.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/exceptions.h>

#include <buoyant_hydrodynamic_solver.h>

namespace BuoyantHydrodynamic {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::apply_boundary_conditions()
{
  if (this->verbose)
    this->pcout << "    Apply boundary conditions..." << std::endl;

  // periodic boundary conditions
  if (!this->velocity_boundary_conditions.periodic_bcs.empty() ||
      !this->pressure_boundary_conditions.periodic_bcs.empty() ||
      !this->scalar_boundary_conditions.periodic_bcs.empty())
  {
    AssertThrow(!this->velocity_boundary_conditions.periodic_bcs.empty(),
                ExcMessage("No periodic boundary conditions were specified for "
                           "the velocity."));

    AssertThrow(!this->pressure_boundary_conditions.periodic_bcs.empty(),
                ExcMessage("No periodic boundary conditions were specified for "
                           "the pressure."));

    AssertThrow(!this->scalar_boundary_conditions.periodic_bcs.empty(),
                ExcMessage("No periodic boundary conditions were specified for "
                           "the density."));

    AssertDimension(this->velocity_boundary_conditions.periodic_bcs.size(),
                    this->pressure_boundary_conditions.periodic_bcs.size());

    AssertDimension(this->velocity_boundary_conditions.periodic_bcs.size(),
                    this->scalar_boundary_conditions.periodic_bcs.size());

    // check match of periodic boundary conditions
    for (std::size_t i=0; i<this->velocity_boundary_conditions.periodic_bcs.size(); ++i)
    {
      const PeriodicBoundaryData<dim> &velocity_bc =
          this->velocity_boundary_conditions.periodic_bcs[i];

      // check match of velocity and pressure periodic boundary conditions
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

      // check match of velocity and density periodic boundary conditions
      matching_bc_found = false;

      for (std::size_t j=0; j<this->scalar_boundary_conditions.periodic_bcs.size(); ++j)
      {
        const PeriodicBoundaryData<dim> &density_bc =
            this->scalar_boundary_conditions.periodic_bcs[j];

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
  }

  Hydrodynamic::Solver<dim, TriangulationType>::apply_boundary_conditions();
  Advection::Solver<dim, TriangulationType>::apply_boundary_conditions();

}

// explicit instantiation
template void Solver<2>::apply_boundary_conditions();
template void Solver<3>::apply_boundary_conditions();

}  // namespace BuoyantHydrodynamic
