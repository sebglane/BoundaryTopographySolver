/*
 * assemble_rhs_local.cc
 *
 *  Created on: Apr 14, 2022
 *      Author: sg
 */
#include <assembly_functions.h>
#include <buoyant_hydrodynamic_solver.h>

#include <optional>

namespace BuoyantHydrodynamic {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
assemble_rhs_local_cell
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 AssemblyData::RightHandSide::ScratchData<dim>         &scratch,
 MeshWorker::CopyData<0,1,1>                           &data,
 const bool                                             use_stress_form) const
{
  data.vectors[0] = 0;
  cell->get_dof_indices(data.local_dof_indices[0]);

  const auto &fe_values = scratch.reinit(cell);
  scratch.extract_local_dof_values("evaluation_point",
                                   this->evaluation_point);
  const auto &JxW = scratch.get_JxW_values();

  const FEValuesExtractors::Vector  velocity(this->velocity_fe_index);
  const FEValuesExtractors::Scalar  pressure(this->pressure_fe_index);
  const FEValuesExtractors::Scalar  density(this->scalar_fe_index);

  const double nu{1.0 / this->reynolds_number};
  const double delta{this->Hydrodynamic::Solver<dim, TriangulationType>::c *
                     std::pow(cell->diameter(), 2)};
  const double delta_density{this->Advection::Solver<dim, TriangulationType>::c *
                             std::pow(cell->diameter(), 2)};

  Hydrodynamic::AssemblyData::RightHandSide::
  ScratchData<dim> &hydrodynamic_scratch
    = static_cast<Hydrodynamic::AssemblyData::RightHandSide::ScratchData<dim> &>(scratch);
  Advection::AssemblyData::RightHandSide::
  ScratchData<dim> &advection_scratch
    = static_cast<Advection::AssemblyData::RightHandSide::ScratchData<dim> &>(scratch);

  // solution values
  auto &present_velocity_values = scratch.present_velocity_values;
  auto &present_velocity_gradients = scratch.present_velocity_gradients;
  present_velocity_values = scratch.get_values("evaluation_point",
                                               velocity);
  present_velocity_gradients = scratch.get_gradients("evaluation_point",
                                                     velocity);
  const auto &present_pressure_values = scratch.get_values("evaluation_point",
                                                           pressure);
  const auto &present_density_values = scratch.get_values("evaluation_point",
                                                          density);
  const auto &present_density_gradients = scratch.get_gradients("evaluation_point",
                                                                density);

  // assign vector options
  hydrodynamic_scratch.assign_vector_options_local_cell("evaluation_point",
                                                        velocity,
                                                        pressure,
                                                        this->angular_velocity_ptr,
                                                        this->body_force_ptr,
                                                        this->background_velocity_ptr,
                                                        this->rossby_number,
                                                        this->froude_number);
  hydrodynamic_scratch.adjust_velocity_field_local_cell();
  advection_scratch.advection_field_values = present_velocity_values;

  // reference density
  if (this->reference_field_ptr != nullptr)
  {
    this->reference_field_ptr->gradient_list(scratch.get_quadrature_points(),
                                             *advection_scratch.vector_options.reference_gradients);

    advection_scratch.vector_options.gradient_scaling = this->gradient_scaling_number;
    advection_scratch.scalar_options.gradient_scaling = this->gradient_scaling_number;
  }

  // gravity field
  if (gravity_field_ptr != nullptr)
  {
    gravity_field_ptr->value_list(scratch.get_quadrature_points(),
                                  *scratch.vector_options.gravity_field_values);

    hydrodynamic_scratch.vector_options.froude_number = this->froude_number;
    hydrodynamic_scratch.scalar_options.froude_number = this->froude_number;
  }

  // stabilization
  compute_strong_residuals(scratch,
                           present_density_gradients,
                           present_density_values,
                           nu);

  for (const auto q: fe_values.quadrature_point_indices())
  {
    for (const auto i: fe_values.dof_indices())
    {
      hydrodynamic_scratch.phi_velocity[i] = fe_values[velocity].value(i, q);
      hydrodynamic_scratch.grad_phi_velocity[i] = fe_values[velocity].gradient(i, q);
      hydrodynamic_scratch.div_phi_velocity[i] = fe_values[velocity].divergence(i, q);
      hydrodynamic_scratch.phi_pressure[i] = fe_values[pressure].value(i, q);
      advection_scratch.phi[i] = fe_values[density].value(i, q);
      advection_scratch.grad_phi[i] = fe_values[density].gradient(i, q);
    }

    // assign optional shape functions
    hydrodynamic_scratch.assign_optional_shape_functions_local_cell(velocity, pressure, q);

    // assign scalar options
    hydrodynamic_scratch.assign_scalar_options_local_cell(q);

    // reference density
    if (advection_scratch.vector_options.reference_gradients)
      advection_scratch.scalar_options.reference_gradient =
          advection_scratch.vector_options.reference_gradients->at(q);

    // gravity field
    if (scratch.vector_options.gravity_field_values)
      scratch.scalar_options.gravity_field_value =
          scratch.vector_options.gravity_field_values->at(q);

    for (const auto i: fe_values.dof_indices())
    {
      // stress form
      if (hydrodynamic_scratch.scalar_options.use_stress_form)
        hydrodynamic_scratch.scalar_options.velocity_test_function_symmetric_gradient =
            hydrodynamic_scratch.sym_grad_phi_velocity[i];

      const double rhs{compute_rhs(this->stabilization,
                                   scratch,
                                   present_density_gradients[q],
                                   present_density_values[q],
                                   present_pressure_values[q],
                                   i,
                                   q,
                                   nu,
                                   this->mu,
                                   delta,
                                   delta_density)};

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
 const bool                                             use_stress_form) const
{
  const FEValuesExtractors::Vector  velocity(this->velocity_fe_index);
  const FEValuesExtractors::Scalar  pressure(this->pressure_fe_index);
  const FEValuesExtractors::Scalar  density(this->scalar_fe_index);

  const types::boundary_id  boundary_id{cell->face(face_number)->boundary_id()};

  Hydrodynamic::AssemblyData::RightHandSide::
  ScratchData<dim> &hydrodynamic_scratch
    = static_cast<Hydrodynamic::AssemblyData::RightHandSide::ScratchData<dim> &>(scratch);
  Advection::AssemblyData::RightHandSide::
  ScratchData<dim> &advection_scratch
    = static_cast<Advection::AssemblyData::RightHandSide::ScratchData<dim> &>(scratch);

  // Dirichlet boundary conditions
  const typename ScalarBoundaryConditions<dim>::BCMapping
  &dirichlet_bcs = this->scalar_boundary_conditions.dirichlet_bcs;

  if (dirichlet_bcs.find(boundary_id) != dirichlet_bcs.end())
  {
    const auto &fe_face_values = scratch.reinit(cell, face_number);
    const auto &JxW = scratch.get_JxW_values();

    // evaluate solution
    scratch.extract_local_dof_values("evaluation_point",
                                     this->evaluation_point);
    const auto &present_values  = scratch.get_values("evaluation_point",
                                                     density);
    const auto &present_velocity_values = scratch.get_values("evaluation_point",
                                                             velocity);

    // boundary values
    dirichlet_bcs.at(boundary_id)->value_list(fe_face_values.get_quadrature_points(),
                                              advection_scratch.vector_options.boundary_values);
    const auto &boundary_values{advection_scratch.vector_options.boundary_values};

    // normal vectors
    const auto &normal_vectors = fe_face_values.get_normal_vectors();

    // loop over face quadrature points
    for (const auto q: fe_face_values.quadrature_point_indices())
      if (normal_vectors[q] * present_velocity_values[q] < 0.)
      {
        // extract the test function's values at the face quadrature points
        for (const auto i: fe_face_values.dof_indices())
          advection_scratch.phi[i] = fe_face_values[density].value(i,q);

        // loop over the degrees of freedom
        for (const auto i: fe_face_values.dof_indices())
          data.vectors[0](i) += present_velocity_values[q] *
                                normal_vectors[q] *
                                advection_scratch.phi[i] *
                                (present_values[q] - boundary_values[q]) *
                                JxW[q];
      } // loop over face quadrature points
  }


  // Neumann boundary conditions
  const typename VectorBoundaryConditions<dim>::NeumannBCMapping
  &neumann_bcs = this->velocity_boundary_conditions.neumann_bcs;

  if (neumann_bcs.find(boundary_id) != neumann_bcs.end())
  {
    const auto &fe_face_values = hydrodynamic_scratch.reinit(cell, face_number);
    const auto &JxW = hydrodynamic_scratch.get_JxW_values();

    // boundary values
    AssertDimension(fe_face_values.n_quadrature_points,
                    hydrodynamic_scratch.vector_options.boundary_traction_values.size());
    neumann_bcs.at(boundary_id)->value_list(scratch.get_quadrature_points(),
                                            hydrodynamic_scratch.vector_options.boundary_traction_values);
    const std::vector<Tensor<1,dim>>  &boundary_tractions = hydrodynamic_scratch.vector_options.boundary_traction_values;

    // loop over face quadrature points
    for (const auto q: fe_face_values.quadrature_point_indices())
    {
      // extract the test function's values at the face quadrature points
      for (const auto i: fe_face_values.dof_indices())
        hydrodynamic_scratch.phi_velocity[i] = fe_face_values[velocity].value(i, q);

      // loop over the degrees of freedom
      for (const auto i: fe_face_values.dof_indices())
        data.vectors[0](i) += hydrodynamic_scratch.phi_velocity[i] *
                              boundary_tractions[q] *
                              JxW[q];
    } // loop over face quadrature points
  }

  if (this->include_boundary_stress_terms)
    if (std::find(this->boundary_stress_ids.begin(),
                  this->boundary_stress_ids.end(),
                  boundary_id) != this->boundary_stress_ids.end())
    {
      const double nu{1.0 / this->reynolds_number};

      const auto &fe_face_values = scratch.reinit(cell, face_number);
      const auto &JxW = scratch.get_JxW_values();

      scratch.extract_local_dof_values("evaluation_point",
                                       this->evaluation_point);
      const auto &present_pressure_values = scratch.get_values("evaluation_point",
                                                               pressure);

      // normal vectors
      const auto &face_normal_vectors = scratch.get_normal_vectors();

      // compute present boundary traction
      AssertDimension(fe_face_values.n_quadrature_points,
                      hydrodynamic_scratch.vector_options.boundary_traction_values.size());
      std::vector<Tensor<1,dim>>  &boundary_tractions = hydrodynamic_scratch.vector_options.boundary_traction_values;
      if (use_stress_form)
      {
        const auto &present_velocity_sym_gradients
          = scratch.get_symmetric_gradients("evaluation_point",
                                            velocity);
        for (const auto q: fe_face_values.quadrature_point_indices())
          boundary_tractions[q] =
              - present_pressure_values[q] * face_normal_vectors[q]
              + 2.0 * nu * present_velocity_sym_gradients[q] * face_normal_vectors[q];
      }
      else
      {
        const auto &present_velocity_gradients = scratch.get_gradients("evaluation_point",
                                                                       velocity);
        for (const auto q: fe_face_values.quadrature_point_indices())
          boundary_tractions[q] =
              - present_pressure_values[q] * face_normal_vectors[q]
              + nu * present_velocity_gradients[q] * face_normal_vectors[q];
      }

      // loop over face quadrature points
      for (const auto q: fe_face_values.quadrature_point_indices())
      {
        // extract the test function's values at the face quadrature points
        for (const auto i: fe_face_values.dof_indices())
          hydrodynamic_scratch.phi_velocity[i] = fe_face_values[velocity].value(i, q);

        // loop over the degrees of freedom
        for (const auto i: fe_face_values.dof_indices())
          data.vectors[0](i) += hydrodynamic_scratch.phi_velocity[i] *
                                boundary_tractions[q] *
                                JxW[q];
      } // loop over face quadrature points
    } // loop over face quadrature points
}



// explicit instantiation
template
void
Solver<2>::
assemble_rhs_local_cell
(const typename DoFHandler<2>::active_cell_iterator &,
 AssemblyData::RightHandSide::ScratchData<2>        &,
 MeshWorker::CopyData<0,1,1>                        &,
 const bool                                           ) const;
template
void Solver<3>::
assemble_rhs_local_cell
(const typename DoFHandler<3>::active_cell_iterator &,
 AssemblyData::RightHandSide::ScratchData<3>        &,
 MeshWorker::CopyData<0,1,1>                        &,
 const bool                                           ) const;

template
void Solver<2>::
assemble_rhs_local_boundary
(const typename DoFHandler<2>::active_cell_iterator &,
 const unsigned int                                  ,
 AssemblyData::RightHandSide::ScratchData<2>        &,
 MeshWorker::CopyData<0,1,1>                        &,
 const bool                                           ) const;
template
void Solver<3>::
assemble_rhs_local_boundary
(const typename DoFHandler<3>::active_cell_iterator &,
 const unsigned int                                  ,
 AssemblyData::RightHandSide::ScratchData<3>        &,
 MeshWorker::CopyData<0,1,1>                        &,
 const bool                                           ) const;

}  // namespace BuoyantHydrodynamic

