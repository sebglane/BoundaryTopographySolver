/*
 * assemble_rhs_local.cc
 *
 *  Created on: Apr 11, 2022
 *      Author: sg
 */

#include <assembly_functions.h>
#include <hydrodynamic_solver.h>

#include <optional>

namespace Hydrodynamic {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
assemble_rhs_local_cell
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 AssemblyData::RightHandSide::ScratchData<dim>         &scratch,
 MeshWorker::CopyData<0,1,1>                           &data) const
{
  data.vectors[0] = 0;
  cell->get_dof_indices(data.local_dof_indices[0]);

  const auto &fe_values = scratch.reinit(cell);
  scratch.extract_local_dof_values("evaluation_point",
                                   this->evaluation_point);
  const auto &JxW = scratch.get_JxW_values();

  const FEValuesExtractors::Vector  velocity(velocity_fe_index);
  const FEValuesExtractors::Scalar  pressure(pressure_fe_index);

  // viscosity
  const double nu{1.0 / reynolds_number};

  // stabilization parameter
  const double delta{c * std::pow(cell->diameter(), 2)};

  // solution values
  scratch.present_velocity_values = scratch.get_values("evaluation_point",
                                                       velocity);
  scratch.present_velocity_gradients = scratch.get_gradients("evaluation_point",
                                                             velocity);
  scratch.present_pressure_values = scratch.get_values("evaluation_point",
                                                       pressure);

  // assign vector options
  scratch.assign_vector_options("evaluation_point",
                                velocity,
                                pressure,
                                angular_velocity_ptr,
                                body_force_ptr,
                                background_velocity_ptr,
                                rossby_number,
                                froude_number);
  scratch.adjust_velocity_field_local_cell();

  // stabilization
  compute_strong_residual(scratch, nu);

  for (const auto q: fe_values.quadrature_point_indices())
  {
    for (const auto i: fe_values.dof_indices())
    {
      scratch.phi_velocity[i] = fe_values[velocity].value(i, q);
      scratch.grad_phi_velocity[i] = fe_values[velocity].gradient(i, q);
      scratch.div_phi_velocity[i] = fe_values[velocity].divergence(i, q);
      scratch.phi_pressure[i] = fe_values[pressure].value(i, q);
    }

    // assign optional shape functions
    scratch.assign_optional_shape_functions_rhs_local(velocity, pressure, q);

    for (const auto i: fe_values.dof_indices())
    {
      const double rhs{compute_rhs(scratch,
                                   i,
                                   q,
                                   nu,
                                   mu,
                                   delta)};

      data.vectors[0](i) += rhs * JxW[q];

    }
  } // end loop over cell quadrature points
}



// explicit instantiation
template
void
Solver<2>::
assemble_rhs_local_cell
(const typename DoFHandler<2>::active_cell_iterator  &cell,
 AssemblyData::RightHandSide::ScratchData<2>         &scratch,
 MeshWorker::CopyData<0,1,1>                         &data) const;
template
void
Solver<3>::
assemble_rhs_local_cell
(const typename DoFHandler<3>::active_cell_iterator  &cell,
 AssemblyData::RightHandSide::ScratchData<3>         &scratch,
 MeshWorker::CopyData<0,1,1>                         &data) const;

}  // namespace Hydrodynamic


