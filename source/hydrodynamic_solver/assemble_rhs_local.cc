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
 const bool                                             use_stress_form) const
{
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
  Assert(delta > 0.0, ExcLowerRangeType<double>(0.0, delta));

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

  // stabilization
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
      if (stabilization & apply_pspg)
        scratch.grad_phi_pressure[i] = fe_values[pressure].gradient(i, q);
    }

    // stress form
    if (use_stress_form)
      scalar_options.present_symmetric_velocity_gradient =
          present_sym_velocity_gradients->at(q);

    // background field
    std::optional<Tensor<1,dim>>  background_velocity_value;
    if (vector_options.background_velocity_values)
      background_velocity_value = vector_options.background_velocity_values->at(q);
    std::optional<Tensor<2,dim>>  background_velocity_gradient;
    if (vector_options.background_velocity_gradients)
      background_velocity_gradient = vector_options.background_velocity_gradients->at(q);

    // body force
    if (vector_options.body_force_values)
      scalar_options.body_force_value =
          vector_options.body_force_values->at(q);

    for (const auto i: fe_values.dof_indices())
    {
      // stress form
      if (use_stress_form)
        scalar_options.velocity_test_function_symmetric_gradient =
            scratch.sym_grad_phi_velocity[i];

      double rhs = compute_rhs(scratch.phi_velocity[i],
                               scratch.grad_phi_velocity[i],
                               present_velocity_values[q],
                               present_velocity_gradients[q],
                               present_pressure_values[q],
                               scratch.phi_pressure[i],
                               nu,
                               scalar_options,
                               background_velocity_value,
                               background_velocity_gradient);

      if (stabilization & (apply_supg|apply_pspg))
      {
        Tensor<1, dim> stabilization_test_function;

        if (stabilization & apply_supg)
        {
          stabilization_test_function += scratch.grad_phi_velocity[i] *
                                         present_velocity_values[q];
          if (background_velocity_ptr != nullptr)
          {
            Assert(background_velocity_value,
                   ExcMessage("Optional background velocity was not specified."));
            stabilization_test_function += scratch.grad_phi_velocity[i] *
                                           *background_velocity_value;
          }
        }
        if (stabilization & apply_pspg)
          stabilization_test_function += scratch.grad_phi_pressure[i];

        rhs -= delta * scratch.present_strong_residuals[q] * stabilization_test_function;
      }

      if (stabilization & apply_grad_div)
        rhs += mu * compute_grad_div_rhs(present_velocity_gradients[q],
                                         scratch.grad_phi_velocity[i]);

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
      neumann_bcs.at(boundary_id)->value_list(fe_face_values.get_quadrature_points(),
                                              scratch.vector_options.boundary_traction_values);

      // loop over face quadrature points
      for (const auto q: fe_face_values.quadrature_point_indices())
      {
        // extract the test function's values at the face quadrature points
        for (const auto i: fe_face_values.dof_indices())
          scratch.phi_velocity[i] = fe_face_values[velocity].value(i,q);

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
          scratch.phi_velocity[i] = fe_face_values[velocity].value(i, q);

        // loop over the degrees of freedom
        for (const auto i: fe_face_values.dof_indices())
          data.vectors[0](i) += scratch.phi_velocity[i] *
                                scratch.vector_options.boundary_traction_values[q] *
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


