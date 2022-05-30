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
  const auto &present_velocity_values = scratch.get_values("evaluation_point",
                                                            velocity);
  const auto &present_velocity_gradients = scratch.get_gradients("evaluation_point",
                                                                 velocity);
  const auto &present_pressure_values = scratch.get_values("evaluation_point",
                                                           pressure);
  const auto &present_density_values = scratch.get_values("evaluation_point",
                                                          density);
  const auto &present_density_gradients = scratch.get_gradients("evaluation_point",
                                                                density);

  // stress form
  std::optional<std::vector<SymmetricTensor<2, dim>>> present_sym_velocity_gradients;
  if (use_stress_form)
    present_sym_velocity_gradients = scratch.get_symmetric_gradients("evaluation_point",
                                                                     velocity);
  // stabilization related solution values
  std::vector<Tensor<1, dim>> present_velocity_laplaceans(fe_values.n_quadrature_points);
  std::vector<Tensor<1, dim>> present_pressure_gradients(fe_values.n_quadrature_points);
  if (this->stabilization & (apply_supg|apply_pspg))
  {
    present_velocity_laplaceans = scratch.get_laplacians("evaluation_point",
                                                         velocity);
    present_pressure_gradients = scratch.get_gradients("evaluation_point",
                                                       pressure);
    if (use_stress_form)
    {
      const auto &present_hessians = scratch.get_hessians("evaluation_point",
                                                          velocity);

      std::vector<Tensor<1, dim>> &present_velocity_grad_divergences =
          hydrodynamic_scratch.vector_options.present_velocity_grad_divergences.value();
      for (std::size_t q=0; q<present_hessians.size(); ++q)
      {
        present_velocity_grad_divergences[q] = 0;
        for (unsigned int d=0; d<dim; ++d)
          present_velocity_grad_divergences[q] += present_hessians[q][d][d];
      }
    }
  }

  // body force
  if (this->body_force_ptr != nullptr)
  {
    this->body_force_ptr->value_list(scratch.get_quadrature_points(),
                                     *hydrodynamic_scratch.vector_options.body_force_values);
    hydrodynamic_scratch.vector_options.froude_number = this->froude_number;
    hydrodynamic_scratch.scalar_options.froude_number = this->froude_number;
  }

  // background field
  if (this->background_velocity_ptr != nullptr)
  {
    this->background_velocity_ptr->value_list(scratch.get_quadrature_points(),
                                              *hydrodynamic_scratch.vector_options.background_velocity_values);
    this->background_velocity_ptr->gradient_list(scratch.get_quadrature_points(),
                                                 *hydrodynamic_scratch.vector_options.background_velocity_gradients);
  }

  // Coriolis term
  if (this->angular_velocity_ptr != nullptr)
  {
    hydrodynamic_scratch.vector_options.angular_velocity = this->angular_velocity_ptr->value();
    hydrodynamic_scratch.vector_options.rossby_number = this->rossby_number;

    hydrodynamic_scratch.scalar_options.angular_velocity = this->angular_velocity_ptr->value();
    hydrodynamic_scratch.scalar_options.rossby_number = this->rossby_number;
  }

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
  if (this->stabilization & (apply_supg|apply_pspg))
    LegacyBuoyantHydrodynamic::
    compute_strong_hydrodynamic_residual(present_velocity_values,
                                         present_velocity_gradients,
                                         present_velocity_laplaceans,
                                         present_pressure_gradients,
                                         present_density_values,
                                         hydrodynamic_scratch.present_strong_residuals,
                                         nu,
                                         hydrodynamic_scratch.vector_options,
                                         scratch.vector_options);

  std::vector<double> present_strong_density_residuals(fe_values.n_quadrature_points);
  compute_strong_density_residual(present_density_gradients,
                                  present_velocity_values,
                                  present_strong_density_residuals,
                                  hydrodynamic_scratch.vector_options,
                                  advection_scratch.vector_options);



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

    // stress form
    if (use_stress_form)
      for (const auto i: fe_values.dof_indices())
        hydrodynamic_scratch.sym_grad_phi_velocity[i] = fe_values[velocity].symmetric_gradient(i, q);

    // stabilization related shape functions
    if (this->stabilization & (apply_supg|apply_pspg))
      for (const auto i: fe_values.dof_indices())
        hydrodynamic_scratch.grad_phi_pressure[i] = fe_values[pressure].gradient(i, q);

    // stress form
    if (use_stress_form)
      hydrodynamic_scratch.scalar_options.present_symmetric_velocity_gradient =
          present_sym_velocity_gradients->at(q);

    // background field
    std::optional<Tensor<1,dim>>  background_velocity_value;
    if (hydrodynamic_scratch.vector_options.background_velocity_values)
      background_velocity_value = hydrodynamic_scratch.vector_options.background_velocity_values->at(q);
    std::optional<Tensor<2,dim>>  background_velocity_gradient;
    if (hydrodynamic_scratch.vector_options.background_velocity_gradients)
      background_velocity_gradient = hydrodynamic_scratch.vector_options.background_velocity_gradients->at(q);

    // body force
    if (hydrodynamic_scratch.vector_options.body_force_values)
      hydrodynamic_scratch.scalar_options.body_force_value =
          hydrodynamic_scratch.vector_options.body_force_values->at(q);

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
      if (use_stress_form)
        hydrodynamic_scratch.scalar_options.velocity_test_function_symmetric_gradient =
            hydrodynamic_scratch.sym_grad_phi_velocity[i];

      // rhs step 1: hydrodynamic part
      double rhs = BuoyantHydrodynamic::
                   compute_hydrodynamic_rhs(hydrodynamic_scratch.phi_velocity[i],
                                            hydrodynamic_scratch.grad_phi_velocity[i],
                                            present_velocity_values[q],
                                            present_velocity_gradients[q],
                                            present_pressure_values[q],
                                            present_density_values[q],
                                            hydrodynamic_scratch.phi_pressure[i],
                                            nu,
                                            hydrodynamic_scratch.scalar_options,
                                            scratch.scalar_options,
                                            background_velocity_value,
                                            background_velocity_gradient);

      if (this->stabilization & (apply_supg|apply_pspg))
      {
        Tensor<1, dim> stabilization_test_function;

        if (this->stabilization & apply_supg)
        {
          stabilization_test_function += hydrodynamic_scratch.grad_phi_velocity[i] *
                                         present_velocity_values[q];
          if (background_velocity_value)
            stabilization_test_function += hydrodynamic_scratch.grad_phi_velocity[i] *
                                           *background_velocity_value;
        }
        if (this->stabilization & apply_pspg)
          stabilization_test_function += hydrodynamic_scratch.grad_phi_pressure[i];

        rhs -= delta * hydrodynamic_scratch.present_strong_residuals[q] * stabilization_test_function;
      }

      if (this->stabilization & apply_grad_div)
        rhs += this->mu * Hydrodynamic::
               compute_grad_div_rhs(present_velocity_gradients[q],
                                    hydrodynamic_scratch.grad_phi_velocity[i]);

      // rhs step 2: density part
      rhs += compute_density_rhs(present_density_gradients[q],
                                 present_velocity_values[q],
                                 advection_scratch.phi[i],
                                 advection_scratch.scalar_options,
                                 background_velocity_value);

      // standard stabilization terms
      {
        double stabilization_test_function{present_velocity_values[q] *
                                           advection_scratch.grad_phi[i]};

        if (background_velocity_value)
          stabilization_test_function += *background_velocity_value *
                                         advection_scratch.grad_phi[i];

        rhs -= delta_density * present_strong_density_residuals[q] *
               stabilization_test_function;
      }

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

