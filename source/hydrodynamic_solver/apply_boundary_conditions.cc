/*
 * apply_boundary_conditions.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/exceptions.h>

#include <hydrodynamic_solver.h>

#include <set>

namespace Hydrodynamic {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::apply_boundary_conditions()
{
  if (this->verbose)
    this->pcout << "    Apply boundary conditions..." << std::endl;

  AssertThrow(velocity_boundary_conditions.closed(),
              ExcMessage("The velocity boundary conditions have not been closed."));
  AssertThrow(pressure_boundary_conditions.closed(),
              ExcMessage("The pressure boundary conditions have not been closed."));

  const FEValuesExtractors::Vector  velocity(velocity_fe_index);
  const FEValuesExtractors::Scalar  pressure(pressure_fe_index);

  // periodic boundary conditions
  if (!velocity_boundary_conditions.periodic_bcs.empty() ||
      !pressure_boundary_conditions.periodic_bcs.empty())
  {
    AssertThrow(!pressure_boundary_conditions.periodic_bcs.empty(),
                ExcMessage("No periodic boundary conditions were specified for "
                           "the pressure."));

    AssertThrow(!velocity_boundary_conditions.periodic_bcs.empty(),
                ExcMessage("No periodic boundary conditions were specified for "
                           "the velocity."));

    AssertDimension(velocity_boundary_conditions.periodic_bcs.size(),
                    pressure_boundary_conditions.periodic_bcs.size());

    // task: check that periodic boundary conditions are applied equally to both
    //       the velocity and the pressure
    // iterate over periodic bcs of the velocity
    for (std::size_t i=0; i<velocity_boundary_conditions.periodic_bcs.size(); ++i)
    {
      const PeriodicBoundaryData<dim> &velocity_bc =
          velocity_boundary_conditions.periodic_bcs[i];

      bool matching_bc_found = false;

      // iterate over periodic bcs of the pressure
      for (std::size_t j=0; j<pressure_boundary_conditions.periodic_bcs.size(); ++j)
      {
        const PeriodicBoundaryData<dim> &pressure_bc =
            pressure_boundary_conditions.periodic_bcs[j];

        // check whether the bcs are equivalent
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
    }
    // done

    this->apply_periodicity_constraints(velocity_boundary_conditions.periodic_bcs,
                                        this->fe_system->component_mask(velocity));
    this->apply_periodicity_constraints(pressure_boundary_conditions.periodic_bcs,
                                        this->fe_system->component_mask(pressure));
  }

  // Dirichlet velocity boundary conditions
  {
    if (!velocity_boundary_conditions.dirichlet_bcs.empty())
      this->apply_dirichlet_constraints(velocity_boundary_conditions.dirichlet_bcs,
                                        this->fe_system->component_mask(velocity));

    if (!velocity_boundary_conditions.normal_flux_bcs.empty())
      this->apply_normal_flux_constraints(velocity_boundary_conditions.normal_flux_bcs,
                                          this->fe_system->component_mask(velocity));
  }
  // Dirichlet pressure boundary conditions
  {
    if (!include_boundary_stress_terms)
      AssertThrow(pressure_boundary_conditions.regularity_guaranteed(),
                  ExcMessage("No boundary conditions were set for the pressure"));

    if (!pressure_boundary_conditions.dirichlet_bcs.empty())
      this->apply_dirichlet_constraints(pressure_boundary_conditions.dirichlet_bcs,
                                        this->fe_system->component_mask(pressure));

    // apply mean value constraint if boundary traction are ignored
    if (pressure_boundary_conditions.datum_at_boundary() && !include_boundary_stress_terms)
      this->apply_mean_value_constraint(this->fe_system->component_mask(pressure));
  }

  // collect unconstrained boundaries requiring to include the traction on the
  // boundary
  if (include_boundary_stress_terms)
  {
    std::set<types::boundary_id>  fully_constrained_boundary_ids;

    for (const auto &[key, value]: velocity_boundary_conditions.dirichlet_bcs)
      fully_constrained_boundary_ids.insert(key);
    for (const auto &[key, value]: velocity_boundary_conditions.neumann_bcs)
      fully_constrained_boundary_ids.insert(key);
    for (const auto &[key, value]: pressure_boundary_conditions.dirichlet_bcs)
      fully_constrained_boundary_ids.insert(key);

    for (const auto boundary_id: this->triangulation.get_boundary_ids())
      if (fully_constrained_boundary_ids.find(boundary_id) != fully_constrained_boundary_ids.end())
        boundary_stress_ids.push_back(boundary_id);
  }

}

// explicit instantiation
template void Solver<2>::apply_boundary_conditions();
template void Solver<3>::apply_boundary_conditions();


}  // namespace Hydrodynamic
