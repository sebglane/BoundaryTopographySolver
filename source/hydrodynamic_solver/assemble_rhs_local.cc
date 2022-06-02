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
 MeshWorker::CopyData<0,1,1>                           &data,
 const bool                                             /* use_stress_form */) const
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
  auto &present_velocity_values = scratch.present_velocity_values;
  auto &present_velocity_gradients = scratch.present_velocity_gradients;
  present_velocity_values = scratch.get_values("evaluation_point",
                                               velocity);
  present_velocity_gradients = scratch.get_gradients("evaluation_point",
                                                     velocity);
  const auto &present_pressure_values = scratch.get_values("evaluation_point",
                                                           pressure);

  // assign vector options
  scratch.assign_vector_options_local_cell("evaluation_point",
                                           velocity,
                                           pressure,
                                           angular_velocity_ptr,
                                           body_force_ptr,
                                           background_velocity_ptr,
                                           rossby_number,
                                           froude_number);
  scratch.adjust_velocity_field_local_cell();

  // stabilization
  if (stabilization & (apply_supg|apply_pspg))
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
    scratch.assign_optional_shape_functions_local_cell(velocity, pressure, q);

    // assign scalar options
    scratch.assign_scalar_options_local_cell(q);

    for (const auto i: fe_values.dof_indices())
    {
      // stress form
      if (scratch.scalar_options.use_stress_form)
        scratch.scalar_options.velocity_test_function_symmetric_gradient =
            scratch.sym_grad_phi_velocity[i];

      const double rhs{compute_rhs(stabilization,
                                   scratch,
                                   present_pressure_values[q],
                                   i,
                                   q,
                                   nu,
                                   mu,
                                   delta)};

      data.vectors[0](i) += rhs * JxW[q];

    }
  } // end loop over cell quadrature points
}



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
assemble_rhs_local_boundary
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 const unsigned int                                     face_number,
 AssemblyData::RightHandSide::ScratchData<dim>         &scratch,
 MeshWorker::CopyData<0,1,1>                           &data,
 const bool                                             /* use_stress_form */) const
{
  const FEValuesExtractors::Vector  velocity(velocity_fe_index);
  const FEValuesExtractors::Scalar  pressure(pressure_fe_index);

  const types::boundary_id  boundary_id{cell->face(face_number)->boundary_id()};

  const typename VectorBoundaryConditions<dim>::NeumannBCMapping
  &neumann_bcs = velocity_boundary_conditions.neumann_bcs;

  // Neumann boundary condition
  if (!neumann_bcs.empty())
    if (neumann_bcs.find(boundary_id) != neumann_bcs.end())
    {
      const auto &fe_face_values = scratch.reinit(cell, face_number);
      const auto &JxW = scratch.get_JxW_values();

      // assign vector options
      scratch.assign_vector_options_local_boundary("",
                                                   velocity,
                                                   pressure,
                                                   0.0,
                                                   neumann_bcs.at(boundary_id),
                                                   background_velocity_ptr);

      // boundary traction
      const auto &boundary_tractions{scratch.vector_options.boundary_traction_values};

      // loop over face quadrature points
      for (const auto q: fe_face_values.quadrature_point_indices())
      {
        // extract the test function's values at the face quadrature points
        for (const auto i: fe_face_values.dof_indices())
          scratch.phi_velocity[i] = fe_face_values[velocity].value(i,q);

        // loop over the degrees of freedom
        for (const auto i: fe_face_values.dof_indices())
          data.vectors[0](i) += scratch.phi_velocity[i] *
                                boundary_tractions[q] *
                                JxW[q];
      } // loop over face quadrature points
    }

  // unconstrained boundary condition
  if (include_boundary_stress_terms)
    if (std::find(boundary_stress_ids.begin(),
                  boundary_stress_ids.end(),
                  boundary_id) != boundary_stress_ids.end())
    {
      const double nu{1.0 / reynolds_number};

      const auto &fe_face_values = scratch.reinit(cell, face_number);
      const auto &JxW = scratch.get_JxW_values();

      // evaluate solution
      scratch.extract_local_dof_values("evaluation_point",
                                       this->evaluation_point);

      // assign vector options
      scratch.assign_vector_options_local_boundary("evaluation_point",
                                                   velocity,
                                                   pressure,
                                                   nu,
                                                   nullptr,
                                                   background_velocity_ptr);
      scratch.adjust_velocity_field_local_boundary();

      // boundary traction
      const auto &boundary_tractions{scratch.vector_options.boundary_traction_values};

      // loop over face quadrature points
      for (const auto q: fe_face_values.quadrature_point_indices())
      {
        // extract the test function's values at the face quadrature points
        for (const auto i: fe_face_values.dof_indices())
          scratch.phi_velocity[i] = fe_face_values[velocity].value(i, q);

        // loop over the degrees of freedom
        for (const auto i: fe_face_values.dof_indices())
          data.vectors[0](i) += scratch.phi_velocity[i] *
                                boundary_tractions[q] *
                                JxW[q];
      } // Loop over face quadrature points
    }
}


// explicit instantiation
template
void
Solver<2>::
assemble_rhs_local_cell
(const typename DoFHandler<2>::active_cell_iterator  &cell,
 AssemblyData::RightHandSide::ScratchData<2>         &scratch,
 MeshWorker::CopyData<0,1,1>                         &data,
 const bool                                           use_stress_form) const;
template
void
Solver<3>::
assemble_rhs_local_cell
(const typename DoFHandler<3>::active_cell_iterator  &cell,
 AssemblyData::RightHandSide::ScratchData<3>         &scratch,
 MeshWorker::CopyData<0,1,1>                         &data,
 const bool                                           use_stress_form) const;

template
void
Solver<2>::
assemble_rhs_local_boundary
(const typename DoFHandler<2>::active_cell_iterator  &cell,
 const unsigned int                                   face_number,
 AssemblyData::RightHandSide::ScratchData<2>         &scratch,
 MeshWorker::CopyData<0,1,1>                         &data,
 const bool                                           use_stress_form) const;
template
void
Solver<3>::
assemble_rhs_local_boundary
(const typename DoFHandler<3>::active_cell_iterator  &cell,
 const unsigned int                                   face_number,
 AssemblyData::RightHandSide::ScratchData<3>         &scratch,
 MeshWorker::CopyData<0,1,1>                         &data,
 const bool                                           use_stress_form) const;

}  // namespace Hydrodynamic


