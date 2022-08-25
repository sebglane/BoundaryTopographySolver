/*
 * apply_boundary_conditions.cc
 *
 *  Created on: Aug 22, 2022
 *      Author: sg
 */

#include <deal.II/base/exceptions.h>

#include <magnetic_induction_solver.h>

#include <set>

namespace MagneticInduction {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::apply_boundary_conditions()
{
  if (this->verbose)
    this->pcout << "    Apply boundary conditions..." << std::endl;

  AssertThrow(magnetic_field_boundary_conditions.closed(),
              ExcMessage("The magnetic field boundary conditions have not been closed."));
  AssertThrow(magnetic_pressure_boundary_conditions.closed(),
              ExcMessage("The magnetic pseudo pressure boundary conditions have not been closed."));

  const FEValuesExtractors::Vector  magnetic_field(magnetic_field_fe_index);
  const FEValuesExtractors::Scalar  magnetic_pressure(magnetic_pressure_fe_index);

  // periodic boundary conditions
  if (!magnetic_field_boundary_conditions.periodic_bcs.empty() ||
      !magnetic_pressure_boundary_conditions.periodic_bcs.empty())
  {
    AssertThrow(!magnetic_field_boundary_conditions.periodic_bcs.empty(),
                ExcMessage("No periodic boundary conditions were specified for "
                           "the magnetic field."));

    AssertThrow(!magnetic_pressure_boundary_conditions.periodic_bcs.empty(),
                ExcMessage("No periodic boundary conditions were specified for "
                           "the magnetic pseudo pressure."));

    AssertDimension(magnetic_field_boundary_conditions.periodic_bcs.size(),
        magnetic_pressure_boundary_conditions.periodic_bcs.size());

    // task: check that periodic boundary conditions are applied equally to both
    //       the magnetic field and the magnetic pressure
    // iterate over periodic bcs of the magnetic field
    for (std::size_t i=0; i<magnetic_field_boundary_conditions.periodic_bcs.size(); ++i)
    {
      const PeriodicBoundaryData<dim> &magnetic_field_bc =
          magnetic_field_boundary_conditions.periodic_bcs[i];

      bool matching_bc_found = false;

      // iterate over periodic bcs of the magnetic pressure
      for (std::size_t j=0; j<magnetic_pressure_boundary_conditions.periodic_bcs.size(); ++j)
      {
        const PeriodicBoundaryData<dim> &magnetic_pressure_bc =
            magnetic_pressure_boundary_conditions.periodic_bcs[j];

        // check whether the bcs are equivalent
        if (magnetic_field_bc.direction != magnetic_pressure_bc.direction)
          continue;
        if (magnetic_field_bc.boundary_pair.first != magnetic_pressure_bc.boundary_pair.first)
          continue;
        if (magnetic_field_bc.boundary_pair.second != magnetic_pressure_bc.boundary_pair.second)
          continue;

        matching_bc_found = true;
        break;
      }

      AssertThrow(matching_bc_found == true,
                  ExcMessage("A matching periodic boundary could not be found."));
    }
    // done

    this->apply_periodicity_constraints(magnetic_field_boundary_conditions.periodic_bcs,
                                        this->fe_system->component_mask(magnetic_field));
    this->apply_periodicity_constraints(magnetic_pressure_boundary_conditions.periodic_bcs,
                                        this->fe_system->component_mask(magnetic_pressure));
  }

  // Dirichlet magnetic field boundary conditions
  AssertThrow(magnetic_field_boundary_conditions.dirichlet_bcs.empty(),
              ExcMessage("Only tangential boundary conditions can be applied for the magnetic field."));
  AssertThrow(magnetic_field_boundary_conditions.neumann_bcs.empty(),
              ExcMessage("Only tangential boundary conditions can be applied for the magnetic field."));
  AssertThrow(magnetic_field_boundary_conditions.normal_flux_bcs.empty(),
              ExcMessage("Only tangential boundary conditions can be applied for the magnetic field."));

  if (!magnetic_field_boundary_conditions.tangential_flux_bcs.empty())
    this->apply_tangential_flux_constraints(magnetic_field_boundary_conditions.tangential_flux_bcs,
                                            this->fe_system->component_mask(magnetic_field));

  // Dirichlet magnetic pressure boundary conditions
  if (!magnetic_pressure_boundary_conditions.dirichlet_bcs.empty())
    this->apply_dirichlet_constraints(magnetic_pressure_boundary_conditions.dirichlet_bcs,
                                      this->fe_system->component_mask(magnetic_pressure));

}

// explicit instantiation
template void Solver<2>::apply_boundary_conditions();
template void Solver<3>::apply_boundary_conditions();


}  // namespace MagneticInduction
