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
 const bool                                             use_newton_linearization,
 const bool                                             use_stress_form) const
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

  const double nu{1.0 / reynolds_number};
  const double delta{c * std::pow(cell->diameter(), 2)};

  OptionalScalarArguments<dim> &scalar_options = scratch.scalar_options;
  OptionalVectorArguments<dim> &vector_options = scratch.vector_options;
//  scalar_options.use_stress_form = use_stress_form;
//  vector_options.use_stress_form = use_stress_form;

  // solution values
  const auto &present_velocity_values = scratch.get_values("evaluation_point",
                                                            velocity);
  const auto &present_velocity_gradients = scratch.get_gradients("evaluation_point",
                                                                 velocity);
  const auto &present_pressure_values = scratch.get_values("evaluation_point",
                                                           pressure);
  // stress form
  std::optional<std::vector<SymmetricTensor<2, dim>>> present_sym_velocity_gradients;
  if (use_stress_form)
    present_sym_velocity_gradients = scratch.get_symmetric_gradients("evaluation_point",
                                                                     velocity);
  // stabilization related solution values
  std::optional<std::vector<Tensor<1, dim>>> present_velocity_laplaceans;
  std::optional<std::vector<Tensor<1, dim>>> present_pressure_gradients;
  if (stabilization & (apply_supg|apply_pspg))
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
          vector_options.present_velocity_grad_divergences.value();
      for (std::size_t q=0; q<present_hessians.size(); ++q)
      {
        present_velocity_grad_divergences[q] = 0;
        for (unsigned int d=0; d<dim; ++d)
          present_velocity_grad_divergences[q] += present_hessians[q][d][d];
      }
    }
  }

  // body force
  if (body_force_ptr != nullptr)
  {
    body_force_ptr->value_list(scratch.get_quadrature_points(),
                               *vector_options.body_force_values);
    vector_options.froude_number = froude_number;
    scalar_options.froude_number = froude_number;
  }

  // background field
  if (background_velocity_ptr != nullptr)
  {
    background_velocity_ptr->value_list(scratch.get_quadrature_points(),
                                        *vector_options.background_velocity_values);
    background_velocity_ptr->gradient_list(scratch.get_quadrature_points(),
                                           *vector_options.background_velocity_gradients);
  }

  // Coriolis term
  if (angular_velocity_ptr != nullptr)
  {
    vector_options.angular_velocity = angular_velocity_ptr->value();
    vector_options.rossby_number = rossby_number;

    scalar_options.angular_velocity = angular_velocity_ptr->value();
    scalar_options.rossby_number = rossby_number;
  }

  if (stabilization & (apply_supg|apply_pspg))
    compute_strong_residual(present_velocity_values,
                            present_velocity_gradients,
                            present_velocity_laplaceans.value(),
                            present_pressure_gradients.value(),
                            scratch.present_strong_residuals,
                            nu,
                            vector_options);

  for (const auto q: fe_values.quadrature_point_indices())
  {
    for (const auto i: fe_values.dof_indices())
    {
      scratch.phi_velocity[i] = fe_values[velocity].value(i, q);
      scratch.grad_phi_velocity[i] = fe_values[velocity].gradient(i, q);
      scratch.div_phi_velocity[i] = fe_values[velocity].divergence(i, q);
      scratch.phi_pressure[i] = fe_values[pressure].value(i, q);

      // stress form
      if (use_stress_form)
        scratch.sym_grad_phi_velocity[i] = fe_values[velocity].symmetric_gradient(i, q);

      // stabilization related shape functions
      if (stabilization & (apply_supg|apply_pspg))
      {
        scratch.grad_phi_pressure[i] = fe_values[pressure].gradient(i, q);

        const Tensor<3, dim> shape_hessian(fe_values[velocity].hessian(i, q));
        for (unsigned int d=0; d<dim; ++d)
          scratch.laplace_phi_velocity[i][d] = trace(shape_hessian[d]);

        // stress form
        if (use_stress_form)
        {
          scratch.grad_div_phi_velocity[i] = 0;
          for (unsigned int d=0; d<dim; ++d)
            scratch.grad_div_phi_velocity[i] += shape_hessian[d][d];
        }
      }
    }

    // stress form
    if (use_stress_form)
      scalar_options.present_symmetric_velocity_gradient =
          present_sym_velocity_gradients->at(q);

    // background field
    if (vector_options.background_velocity_values)
      scalar_options.background_velocity_value =
          vector_options.background_velocity_values->at(q);
    if (vector_options.background_velocity_gradients)
      scalar_options.background_velocity_gradient =
          vector_options.background_velocity_gradients->at(q);

    // body force
    if (vector_options.body_force_values)
      scalar_options.body_force_value =
          vector_options.body_force_values->at(q);

    for (const auto i: fe_values.dof_indices())
    {
      const Tensor<1, dim> &velocity_test_function = scratch.phi_velocity[i];
      const Tensor<2, dim> &velocity_test_function_gradient = scratch.grad_phi_velocity[i];

      const double          pressure_test_function = scratch.phi_pressure[i];

      // stress form
      if (use_stress_form)
        scalar_options.velocity_test_function_symmetric_gradient =
            scratch.sym_grad_phi_velocity[i];

      // stabilization
      std::optional<Tensor<2,dim>> optional_velocity_test_function_gradient;
      if (stabilization & apply_supg)
        optional_velocity_test_function_gradient =
            velocity_test_function_gradient;
      std::optional<Tensor<1,dim>> pressure_test_function_gradient;
      if (stabilization & apply_pspg)
        pressure_test_function_gradient = scratch.grad_phi_pressure[i];

      for (const auto j: fe_values.dof_indices())
      {
        // stress form
        if (use_stress_form)
          scalar_options.velocity_trial_function_symmetric_gradient =
                scratch.sym_grad_phi_velocity[j];

        double matrix = compute_matrix(scratch.phi_velocity[j],
                                       scratch.grad_phi_velocity[j],
                                       velocity_test_function,
                                       velocity_test_function_gradient,
                                       present_velocity_values[q],
                                       present_velocity_gradients[q],
                                       scratch.phi_pressure[j],
                                       pressure_test_function,
                                       nu,
                                       scalar_options,
                                       use_newton_linearization);

        if (stabilization & (apply_supg|apply_pspg))
        {
          // stress form
          if (use_stress_form)
            scalar_options.velocity_trial_function_grad_divergence =
                  scratch.grad_div_phi_velocity[j];

          matrix += delta *
                    compute_residual_linearization_matrix(scratch.phi_velocity[j],
                                                          scratch.grad_phi_velocity[j],
                                                          scratch.laplace_phi_velocity[j],
                                                          scratch.grad_phi_pressure[j],
                                                          present_velocity_values[q],
                                                          present_velocity_gradients[q],
                                                          optional_velocity_test_function_gradient,
                                                          pressure_test_function_gradient,
                                                          nu,
                                                          scalar_options,
                                                          use_newton_linearization);
          if (stabilization & apply_supg)
            matrix += delta * scratch.present_strong_residuals[q] *
                      (velocity_test_function_gradient * scratch.phi_velocity[j]);
        }

        if (stabilization & apply_grad_div)
          matrix += mu * compute_grad_div_matrix(scratch.grad_phi_velocity[j],
                                                 velocity_test_function_gradient);

        data.matrices[0](i, j) +=  matrix * JxW[q];
      }

      double rhs = compute_rhs(velocity_test_function,
                               velocity_test_function_gradient,
                               present_velocity_values[q],
                               present_velocity_gradients[q],
                               present_pressure_values[q],
                               pressure_test_function,
                               nu,
                               scalar_options);

      if (stabilization & (apply_supg|apply_pspg))
      {
        Tensor<1, dim> stabilization_test_function;

        if (stabilization & apply_supg)
        {
          stabilization_test_function += velocity_test_function_gradient *
                                         present_velocity_values[q];
          if (scalar_options.background_velocity_value)
            stabilization_test_function += velocity_test_function_gradient *
                                           *scalar_options.background_velocity_value;
        }
        if (stabilization & apply_pspg)
          stabilization_test_function += scratch.grad_phi_pressure[i];

        rhs -= delta * scratch.present_strong_residuals[q] * stabilization_test_function;
      }

      if (stabilization & apply_grad_div)
        rhs += mu * compute_grad_div_rhs(present_velocity_gradients[q],
                                         velocity_test_function_gradient);
      data.vectors[0](i) += rhs * JxW[q];
    }
  } // end loop over cell quadrature points

}



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
assemble_system_local_boundary
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 const unsigned int                                     face_number,
 AssemblyData::Matrix::ScratchData<dim>                &scratch,
 MeshWorker::CopyData<1,1,1>                           &data,
 const bool                                             use_stress_form) const
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

      AssertDimension(fe_face_values.n_quadrature_points,
                      scratch.vector_options.boundary_traction_values.size());
      neumann_bcs.at(boundary_id)->value_list(scratch.get_quadrature_points(),
                                              scratch.vector_options.boundary_traction_values);

      // loop over face quadrature points
      for (const auto q: fe_face_values.quadrature_point_indices())
      {
        // extract the test function's values at the face quadrature points
        for (const auto i: fe_face_values.dof_indices())
          scratch.phi_velocity[i] = fe_face_values[velocity].value(i, q);

        // loop over the degrees of freedom
        for (const auto i: fe_face_values.dof_indices())
          data.vectors[0](i) += scratch.phi_velocity[i] *
                                scratch.vector_options.boundary_traction_values[q] *
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

      scratch.extract_local_dof_values("evaluation_point",
                                       this->evaluation_point);
      const auto &present_pressure_values = scratch.get_values("evaluation_point",
                                                               pressure);

      // normal vectors
      const auto &face_normal_vectors = scratch.get_normal_vectors();

      // compute present boundary traction
      AssertDimension(fe_face_values.n_quadrature_points,
                      scratch.vector_options.boundary_traction_values.size());
      if (use_stress_form)
      {
        const auto &present_velocity_sym_gradients
          = scratch.get_symmetric_gradients("evaluation_point",
                                            velocity);
        for (const auto q: fe_face_values.quadrature_point_indices())
          scratch.vector_options.boundary_traction_values[q] =
              - present_pressure_values[q] * face_normal_vectors[q]
              + 2.0 * nu * present_velocity_sym_gradients[q] * face_normal_vectors[q];
      }
      else
      {
        const auto &present_velocity_gradients = scratch.get_gradients("evaluation_point",
                                                                       velocity);
        for (const auto q: fe_face_values.quadrature_point_indices())
          scratch.vector_options.boundary_traction_values[q] =
              - present_pressure_values[q] * face_normal_vectors[q]
              + nu * present_velocity_gradients[q] * face_normal_vectors[q];
      }

      // loop over face quadrature points
      for (const auto q: fe_face_values.quadrature_point_indices())
      {
        // extract the test function's values at the face quadrature points
        for (const auto i: fe_face_values.dof_indices())
        {
          scratch.phi_velocity[i] = fe_face_values[velocity].value(i, q);
          scratch.phi_pressure[i] = fe_face_values[pressure].value(i, q);

          if (use_stress_form)
            scratch.sym_grad_phi_velocity[i] = fe_face_values[velocity].symmetric_gradient(i, q);
          else
            scratch.grad_phi_velocity[i] = fe_face_values[velocity].gradient(i, q);

        }

        // loop over the degrees of freedom
        for (const auto i: fe_face_values.dof_indices())
        {
          if (use_stress_form)
            for (const auto j: fe_face_values.dof_indices())
              data.matrices[0](i, j) -=
                  (-scratch.phi_pressure[j] * face_normal_vectors[q] +
                   2.0 * nu * scratch.sym_grad_phi_velocity[j] * face_normal_vectors[q]) *
                   scratch.phi_velocity[i] * JxW[q];
          else
            for (const auto j: fe_face_values.dof_indices())
              data.matrices[0](i, j) -=
                  (-scratch.phi_pressure[j] * face_normal_vectors[q] +
                   nu * scratch.grad_phi_velocity[j] * face_normal_vectors[q]) *
                   scratch.phi_velocity[i] * JxW[q];

          data.vectors[0](i) += scratch.phi_velocity[i] *
                                scratch.vector_options.boundary_traction_values[q] *
                                JxW[q];
        }

      } // loop over face quadrature points
    }
}



// explicit instantiation
template
void
Solver<2>::
assemble_system_local_cell
(const typename DoFHandler<2>::active_cell_iterator &,
 AssemblyData::Matrix::ScratchData<2>               &,
 MeshWorker::CopyData<1,1,1>                        &,
 const bool                                          ,
 const bool                                           ) const;
template
void
Solver<3>::
assemble_system_local_cell
(const typename DoFHandler<3>::active_cell_iterator &,
 AssemblyData::Matrix::ScratchData<3>               &,
 MeshWorker::CopyData<1,1,1>                        &,
 const bool                                          ,
 const bool                                           ) const;

template
void
Solver<2>::
assemble_system_local_boundary
(const typename DoFHandler<2>::active_cell_iterator &,
 const unsigned int                                  ,
 AssemblyData::Matrix::ScratchData<2>               &,
 MeshWorker::CopyData<1,1,1>                        &,
 const bool                                           ) const;
template
void
Solver<3>::
assemble_system_local_boundary
(const typename DoFHandler<3>::active_cell_iterator &,
 const unsigned int                                  ,
 AssemblyData::Matrix::ScratchData<3>               &,
 MeshWorker::CopyData<1,1,1>                        &,
 const bool                                           ) const;


}  // namespace Hydrodynamic
