/*
 * assemble_system_local.cc
 *
 *  Created on: Apr 11, 2022
 *      Author: sg
 */

#include <assembly_functions.h>
#include <hydrodynamic_solver.h>

#include <functional>
#include <optional>


namespace Hydrodynamic {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
assemble_system_local_cell
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 AssemblyData::Matrix::ScratchData<dim>                &scratch,
 MeshWorker::CopyData<1,1,1>                           &data,
 const bool                                             use_newton_linearization) const
{
  data.matrices[0] = 0;
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
    scratch.assign_optional_shape_functions_system_local(velocity, pressure, q);

    for (const auto i: fe_values.dof_indices())
    {
      for (const auto j: fe_values.dof_indices())
      {
        const double matrix{compute_matrix(scratch,
                                           i,
                                           j,
                                           q,
                                           nu,
                                           delta,
                                           mu,
                                           use_newton_linearization)};

        data.matrices[0](i, j) +=  matrix * JxW[q];
      }

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
assemble_system_local_cell
(const typename DoFHandler<2>::active_cell_iterator &,
 AssemblyData::Matrix::ScratchData<2>               &,
 MeshWorker::CopyData<1,1,1>                        &,
 const bool                                           ) const;
template
void
Solver<3>::
assemble_system_local_cell
(const typename DoFHandler<3>::active_cell_iterator &,
 AssemblyData::Matrix::ScratchData<3>               &,
 MeshWorker::CopyData<1,1,1>                        &,
 const bool                                           ) const;


}  // namespace Hydrodynamic
