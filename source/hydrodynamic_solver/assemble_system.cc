/*
 * assemble_system.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/fe/fe_values.h>

#include <assembly_functions.h>
#include <hydrodynamic_solver.h>

namespace Hydrodynamic {

template <int dim>
void Solver<dim>::assemble_system(const bool use_homogeneous_constraints)
{
  if (this->verbose)
    std::cout << "    Assemble linear system..." << std::endl;

  if (angular_velocity_ptr != nullptr)
    AssertThrow(rossby_number > 0.0,
                ExcMessage("Non-vanishing Rossby number is required if the angular "
                           "velocity vector is specified."));
  if (body_force_ptr != nullptr)
    AssertThrow(froude_number > 0.0,
                ExcMessage("Non-vanishing Froude number is required if the body "
                           "force is specified."));

  AssertThrow(reynolds_number != 0.0,
              ExcMessage("The Reynolds must not vanish (stabilization is not "
                         "implemented yet)."));

  TimerOutput::Scope timer_section(this->computing_timer, "Assemble system");

  this->system_matrix = 0;
  this->system_rhs = 0;

  // Initiate the quadrature formula
  const QGauss<dim>   quadrature_formula(velocity_fe_degree + 1);

  // Initiate the face quadrature formula
  const QGauss<dim-1>   face_quadrature_formula(velocity_fe_degree + 1);

  // Set up the lambda function for the local assembly operation
  using Scratch = AssemblyData::Matrix::Scratch<dim>;
  using Copy = AssemblyBaseData::Matrix::Copy;
  auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
         Scratch  &scratch,
         Copy     &data)
         {
            assemble_local_system(cell, scratch, data);
         };

  // Set up the lambda function for the copy local to global operation
  auto copier = [this, use_homogeneous_constraints](const Copy   &data)
      {
        copy_local_to_global_system(data, use_homogeneous_constraints);
      };

  // Assemble using the WorkStream approach
  UpdateFlags update_flags = update_values|
                             update_gradients|
                             update_JxW_values;
  if (body_force_ptr != nullptr || background_velocity_ptr != nullptr)
    update_flags |= update_quadrature_points;
  if (stabilization & (apply_supg|apply_pspg))
    update_flags |= update_hessians;

  const UpdateFlags face_update_flags = update_values|
                                        update_quadrature_points|
                                        update_JxW_values;


  WorkStream::run
  (this->dof_handler.begin_active(),
   this->dof_handler.end(),
   worker,
   copier,
   Scratch(this->mapping,
           quadrature_formula,
           *this->fe_system,
           update_flags,
           face_quadrature_formula,
           face_update_flags,
           stabilization,
           body_force_ptr != nullptr,
           !velocity_boundary_conditions.neumann_bcs.empty(),
           background_velocity_ptr != nullptr),
   Copy(this->fe_system->n_dofs_per_cell()));

}



template<int dim>
void Solver<dim>::assemble_local_system
(const typename DoFHandler<dim>::active_cell_iterator &cell,
 AssemblyData::Matrix::Scratch<dim> &scratch,
 AssemblyBaseData::Matrix::Copy     &data) const
{
  scratch.fe_values.reinit(cell);

  data.local_matrix = 0;
  data.local_rhs = 0;

  cell->get_dof_indices(data.local_dof_indices);

  const FEValuesExtractors::Vector  velocity(0);
  const FEValuesExtractors::Scalar  pressure(dim);

  const double nu{1.0 / reynolds_number};

  const double delta{c * std::pow(cell->diameter(), 2)};

  // solution values
  scratch.fe_values[velocity].get_function_values(this->evaluation_point,
                                                  scratch.present_velocity_values);
  scratch.fe_values[velocity].get_function_gradients(this->evaluation_point,
                                                     scratch.present_velocity_gradients);

  scratch.fe_values[pressure].get_function_values(this->evaluation_point,
                                                  scratch.present_pressure_values);

  // stabilization related solution values
  if (stabilization & (apply_supg|apply_pspg))
  {
    scratch.fe_values[velocity].get_function_laplacians(this->evaluation_point,
                                                        *scratch.present_velocity_laplaceans);

    scratch.fe_values[pressure].get_function_gradients(this->evaluation_point,
                                                       *scratch.present_pressure_gradients);
  }

  // body force
  if (body_force_ptr != nullptr)
  {
    body_force_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                               *scratch.body_force_values);

  }

  // background field
  if (background_velocity_ptr != nullptr)
  {
    background_velocity_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                        *scratch.background_velocity_values);
    background_velocity_ptr->gradient_list(scratch.fe_values.get_quadrature_points(),
                                           *scratch.background_velocity_gradients);
  }

  // Coriolis term
  if (angular_velocity_ptr != nullptr)
    scratch.angular_velocity_value = angular_velocity_ptr->value();

  if (stabilization & (apply_supg|apply_pspg))
    compute_strong_residual(scratch.present_velocity_values,
                            scratch.present_velocity_gradients,
                            scratch.present_velocity_laplaceans.value(),
                            scratch.present_pressure_gradients.value(),
                            scratch.present_strong_residuals.value(),
                            nu,
                            scratch.background_velocity_values,
                            scratch.background_velocity_gradients,
                            scratch.body_force_values,
                            froude_number,
                            scratch.angular_velocity_value,
                            rossby_number);

  std::optional<Tensor<1,dim>> background_velocity_value;
  std::optional<Tensor<2,dim>> background_velocity_gradient;
  std::optional<Tensor<1,dim>> body_force_value;

  std::optional<Tensor<1,dim>> pressure_test_function_gradient;
  std::optional<Tensor<2,dim>> optional_velocity_test_function_gradient;

  for (const auto q: scratch.fe_values.quadrature_point_indices())
  {
    for (const auto i: scratch.fe_values.dof_indices())
    {
      scratch.phi_velocity[i] = scratch.fe_values[velocity].value(i, q);
      scratch.grad_phi_velocity[i] = scratch.fe_values[velocity].gradient(i, q);
      scratch.div_phi_velocity[i] = scratch.fe_values[velocity].divergence(i, q);
      scratch.phi_pressure[i] = scratch.fe_values[pressure].value(i, q);

      // stabilization related shape functions
      if (stabilization & (apply_supg|apply_pspg))
      {
        scratch.grad_phi_pressure->at(i) = scratch.fe_values[pressure].gradient(i, q);

        const Tensor<3, dim> shape_hessian(scratch.fe_values[velocity].hessian(i, q));
        for (unsigned int d=0; d<dim; ++d)
          scratch.laplace_phi_velocity->at(i)[d] = trace(shape_hessian[d]);
      }
    }

    if (scratch.background_velocity_values)
      background_velocity_value = scratch.background_velocity_values->at(q);
    if (scratch.background_velocity_gradients)
      background_velocity_gradient = scratch.background_velocity_gradients->at(q);
    if (scratch.body_force_values)
      body_force_value = scratch.body_force_values->at(q);

    const double JxW{scratch.fe_values.JxW(q)};

    for (const auto i: scratch.fe_values.dof_indices())
    {
      const Tensor<1, dim> &velocity_test_function = scratch.phi_velocity[i];
      const Tensor<2, dim> &velocity_test_function_gradient = scratch.grad_phi_velocity[i];

      const double          pressure_test_function = scratch.phi_pressure[i];

      for (const auto j: scratch.fe_values.dof_indices())
      {
        double matrix = compute_matrix(scratch.phi_velocity[j],
                                       scratch.grad_phi_velocity[j],
                                       velocity_test_function,
                                       velocity_test_function_gradient,
                                       scratch.present_velocity_values[q],
                                       scratch.present_velocity_gradients[q],
                                       scratch.phi_pressure[j],
                                       pressure_test_function,
                                       nu,
                                       background_velocity_value,
                                       background_velocity_gradient,
                                       scratch.angular_velocity_value,
                                       rossby_number);

        if (stabilization & (apply_supg|apply_pspg))
        {
          if (stabilization & apply_supg)
            optional_velocity_test_function_gradient = velocity_test_function_gradient;

          if (stabilization & apply_pspg)
            pressure_test_function_gradient = scratch.grad_phi_pressure->at(i);

          matrix += delta *
                    compute_residual_linearization_matrix(scratch.phi_velocity[j],
                                                          scratch.grad_phi_velocity[j],
                                                          scratch.laplace_phi_velocity->at(j),
                                                          scratch.grad_phi_pressure->at(j),
                                                          scratch.present_velocity_values[q],
                                                          scratch.present_velocity_gradients[q],
                                                          nu,
                                                          optional_velocity_test_function_gradient,
                                                          pressure_test_function_gradient,
                                                          background_velocity_value,
                                                          background_velocity_gradient,
                                                          scratch.angular_velocity_value,
                                                          rossby_number);

          if (stabilization & apply_supg)
            matrix += delta * scratch.present_strong_residuals->at(q) *
                      (*optional_velocity_test_function_gradient * scratch.phi_velocity[j]);
        }

        if (stabilization & apply_grad_div)
          matrix += mu * compute_grad_div_matrix(scratch.grad_phi_velocity[j],
                                                 velocity_test_function_gradient);

        data.local_matrix(i, j) +=  matrix * JxW;
      }

      double rhs = compute_rhs(velocity_test_function,
                               velocity_test_function_gradient,
                               scratch.present_velocity_values[q],
                               scratch.present_velocity_gradients[q],
                               scratch.present_pressure_values[q],
                               pressure_test_function,
                               nu,
                               background_velocity_value,
                               background_velocity_gradient,
                               body_force_value,
                               froude_number,
                               scratch.angular_velocity_value,
                               rossby_number);

      if (stabilization & (apply_supg|apply_pspg))
      {
        Tensor<1, dim> stabilization_test_function;

        if (stabilization & apply_supg)
        {
          stabilization_test_function += velocity_test_function_gradient * scratch.present_velocity_values[q];
          if (background_velocity_value)
            stabilization_test_function += velocity_test_function_gradient * *background_velocity_value;
        }
        if (stabilization & apply_pspg)
          stabilization_test_function += scratch.grad_phi_pressure->at(i);

        rhs -= delta * scratch.present_strong_residuals->at(q) * stabilization_test_function;
      }

      if (stabilization & apply_grad_div)
        rhs += mu * compute_grad_div_rhs(scratch.present_velocity_gradients[q],
                                         velocity_test_function_gradient);
      data.local_rhs(i) += rhs * JxW;
    }
  } // end loop over cell quadrature points

  // Loop over the faces of the cell
  const typename VectorBoundaryConditions<dim>::NeumannBCMapping
  &neumann_bcs = velocity_boundary_conditions.neumann_bcs;
  if (!neumann_bcs.empty())
    if (cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() &&
            neumann_bcs.find(face->boundary_id()) != neumann_bcs.end())
        {
          // Neumann boundary condition
          scratch.fe_face_values.reinit(cell, face);

          const types::boundary_id  boundary_id{face->boundary_id()};
          neumann_bcs.at(boundary_id)->value_list(scratch.fe_face_values.get_quadrature_points(),
                                                  *scratch.boundary_traction_values);

          // Loop over face quadrature points
          for (const auto q: scratch.fe_face_values.quadrature_point_indices())
          {
            // Extract the test function's values at the face quadrature points
            for (const auto i: scratch.fe_face_values.dof_indices())
              scratch.phi_velocity[i] = scratch.fe_face_values[velocity].value(i,q);

            const double JxW_face{scratch.fe_face_values.JxW(q)};

            // Loop over the degrees of freedom
            for (const auto i: scratch.fe_face_values.dof_indices())
              data.local_rhs(i) += scratch.phi_velocity[i] *
                                   scratch.boundary_traction_values->at(q) *
                                   JxW_face;

          } // Loop over face quadrature points
        } // Loop over the faces of the cell
 }



template <int dim>
void Solver<dim>::copy_local_to_global_system
(const AssemblyBaseData::Matrix::Copy     &data,
 const bool use_homogeneous_constraints)
{
  const AffineConstraints<double> &constraints =
      (use_homogeneous_constraints ? this->zero_constraints: this->nonzero_constraints);

  constraints.distribute_local_to_global(data.local_matrix,
                                         data.local_rhs,
                                         data.local_dof_indices,
                                         this->system_matrix,
                                         this->system_rhs);
}

// explicit instantiation
template void Solver<2>::assemble_local_system
(const typename DoFHandler<2>::active_cell_iterator &cell,
 AssemblyData::Matrix::Scratch<2> &scratch,
 AssemblyBaseData::Matrix::Copy     &data) const;
template void Solver<3>::assemble_local_system
(const typename DoFHandler<3>::active_cell_iterator &cell,
 AssemblyData::Matrix::Scratch<3> &scratch,
 AssemblyBaseData::Matrix::Copy     &data) const;

template void Solver<2>::copy_local_to_global_system
(const AssemblyBaseData::Matrix::Copy &, const bool);
template void Solver<3>::copy_local_to_global_system
(const AssemblyBaseData::Matrix::Copy &, const bool);

template void Solver<2>::assemble_system(const bool);
template void Solver<3>::assemble_system(const bool);

}  // namespace Hydrodynamic

