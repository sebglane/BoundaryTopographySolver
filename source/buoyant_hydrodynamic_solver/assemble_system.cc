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
#include <buoyant_hydrodynamic_solver.h>

namespace BuoyantHydrodynamic {

template <int dim>
void Solver<dim>::assemble_system
(const bool use_homogeneous_constraints,
 const bool use_newton_linearization)
{
  if (this->verbose)
    std::cout << "    Assemble linear system..." << std::endl;

  if (this->angular_velocity_ptr != nullptr)
    AssertThrow(this->rossby_number > 0.0,
                ExcMessage("Non-vanishing Rossby number is required if the angular "
                           "velocity vector is specified."));
  AssertThrow(gravity_field_ptr != nullptr,
              ExcMessage("For a buoyant fluid, the gravity field must be specified."));

  AssertThrow(this->froude_number > 0.0,
              ExcMessage("For a buoyant fluid, the Froude number must be specified."));
  AssertThrow(this->reynolds_number > 0.0,
              ExcMessage("The Reynolds must not vanish (stabilization is not "
                         "implemented yet)."));

  TimerOutput::Scope timer_section(this->computing_timer, "Assemble system");

  this->system_matrix = 0;
  this->system_rhs = 0;

  // Initiate the quadrature formula
  const QGauss<dim>   quadrature_formula(this->velocity_fe_degree + 1);

  // Initiate the face quadrature formula
  const QGauss<dim-1>   face_quadrature_formula(this->velocity_fe_degree + 1);

  // Set up the lambda function for the local assembly operation
  using Scratch = AssemblyData::Matrix::Scratch<dim>;
  using Copy = AssemblyBaseData::Matrix::Copy;
  auto worker = [this, use_newton_linearization]
                 (const typename DoFHandler<dim>::active_cell_iterator &cell,
                  Scratch  &scratch,
                  Copy     &data)
      {
        assemble_local_system(cell, scratch, data, use_newton_linearization);
      };

  // Set up the lambda function for the copy local to global operation
  auto copier = [this, use_homogeneous_constraints](const Copy   &data)
      {
        this->copy_local_to_global_system(data, use_homogeneous_constraints);
      };

  // Assemble using the WorkStream approach
  UpdateFlags update_flags = update_values|
                             update_gradients|
                             update_quadrature_points|
                             update_JxW_values;
  if (this->body_force_ptr != nullptr)
    update_flags |= update_quadrature_points;
  if (this->stabilization & (apply_supg|apply_pspg))
    update_flags |= update_hessians;

  UpdateFlags face_update_flags = update_values|
                                  update_quadrature_points|
                                  update_JxW_values;
  if (!density_boundary_conditions.dirichlet_bcs.empty())
    face_update_flags |= update_normal_vectors;

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
           this->stabilization,
           this->body_force_ptr != nullptr,
           !this->velocity_boundary_conditions.neumann_bcs.empty(),
           this->background_velocity_ptr != nullptr,
           reference_density_ptr != nullptr,
           !density_boundary_conditions.dirichlet_bcs.empty()),
   Copy(this->fe_system->n_dofs_per_cell()));
}



template<int dim>
void Solver<dim>::assemble_local_system
(const typename DoFHandler<dim>::active_cell_iterator &cell,
 AssemblyData::Matrix::Scratch<dim> &scratch,
 AssemblyBaseData::Matrix::Copy     &data,
 const bool use_newton_linearization) const
{
  scratch.fe_values.reinit(cell);

  data.local_matrix = 0;
  data.local_rhs = 0;

  cell->get_dof_indices(data.local_dof_indices);

  const FEValuesExtractors::Vector  velocity(0);
  const FEValuesExtractors::Scalar  pressure(dim);
  const FEValuesExtractors::Scalar  density(dim+1);

  const double nu{1.0 / this->reynolds_number};

  const double delta{this->c * std::pow(cell->diameter(), 2)};
  const double delta_density{c_density * std::pow(cell->diameter(), 2)};

  // solution values
  scratch.fe_values[velocity].get_function_values(this->evaluation_point,
                                                  scratch.present_velocity_values);
  scratch.fe_values[velocity].get_function_gradients(this->evaluation_point,
                                                     scratch.present_velocity_gradients);

  scratch.fe_values[pressure].get_function_values(this->evaluation_point,
                                                  scratch.present_pressure_values);

  scratch.fe_values[density].get_function_values(this->evaluation_point,
                                                 scratch.present_density_values);
  scratch.fe_values[density].get_function_gradients(this->evaluation_point,
                                                    scratch.present_density_gradients);

  // stabilization related solution values
  if (this->stabilization & (apply_supg|apply_pspg))
  {
    scratch.fe_values[velocity].get_function_laplacians(this->evaluation_point,
                                                        *scratch.present_velocity_laplaceans);

    scratch.fe_values[pressure].get_function_gradients(this->evaluation_point,
                                                       *scratch.present_pressure_gradients);
  }

  // body force
  if (this->body_force_ptr != nullptr)
  {
    this->body_force_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                     *scratch.body_force_values);

  }

  // background field
  if (this->background_velocity_ptr != nullptr)
  {
    this->background_velocity_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                              *scratch.background_velocity_values);
    this->background_velocity_ptr->gradient_list(scratch.fe_values.get_quadrature_points(),
                                                 *scratch.background_velocity_gradients);
  }

  // Coriolis term
  if (this->angular_velocity_ptr != nullptr)
    scratch.angular_velocity_value = this->angular_velocity_ptr->value();

  // reference density
  if (reference_density_ptr != nullptr)
    reference_density_ptr->gradient_list(scratch.fe_values.get_quadrature_points(),
                                         *scratch.reference_density_gradients);

  // gravity field
  gravity_field_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                scratch.gravity_field_values);

  // stabilization
  if (this->stabilization & (apply_supg|apply_pspg))
    compute_strong_hydrodynamic_residual(scratch.present_velocity_values,
                                         scratch.present_velocity_gradients,
                                         scratch.present_velocity_laplaceans.value(),
                                         scratch.present_pressure_gradients.value(),
                                         scratch.gravity_field_values,
                                         scratch.present_density_values,
                                         scratch.present_strong_residuals.value(),
                                         nu,
                                         this->froude_number,
                                         scratch.background_velocity_values,
                                         scratch.background_velocity_gradients,
                                         scratch.body_force_values,
                                         scratch.angular_velocity_value,
                                         this->rossby_number);

  compute_strong_density_residual(scratch.present_density_gradients,
                                  scratch.present_velocity_values,
                                  scratch.present_strong_density_residuals,
                                  scratch.reference_density_gradients,
                                  stratification_number,
                                  scratch.background_velocity_values);

  std::optional<Tensor<1, dim>> background_velocity_value;
  std::optional<Tensor<2, dim>> background_velocity_gradient;
  std::optional<Tensor<1, dim>> body_force_value;

  std::optional<Tensor<1 ,dim>> pressure_test_function_gradient;
  std::optional<Tensor<2, dim>> optional_velocity_test_function_gradient;

  std::optional<Tensor<1, dim>> reference_density_gradient;

  for (const auto q: scratch.fe_values.quadrature_point_indices())
  {
    for (const auto i: scratch.fe_values.dof_indices())
    {
      scratch.phi_velocity[i] = scratch.fe_values[velocity].value(i, q);
      scratch.grad_phi_velocity[i] = scratch.fe_values[velocity].gradient(i, q);
      scratch.div_phi_velocity[i] = scratch.fe_values[velocity].divergence(i, q);
      scratch.phi_pressure[i] = scratch.fe_values[pressure].value(i, q);
      scratch.phi_density[i] = scratch.fe_values[density].value(i, q);
      scratch.grad_phi_density[i] = scratch.fe_values[density].gradient(i, q);

      // stabilization related shape functions
      if (this->stabilization & (apply_supg|apply_pspg))
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
    if (scratch.reference_density_gradients)
      reference_density_gradient = scratch.reference_density_gradients->at(q);

    const double JxW{scratch.fe_values.JxW(q)};

    for (const auto i: scratch.fe_values.dof_indices())
    {
      const Tensor<1, dim> &velocity_test_function = scratch.phi_velocity[i];
      const Tensor<2, dim> &velocity_test_function_gradient = scratch.grad_phi_velocity[i];

      const double          pressure_test_function = scratch.phi_pressure[i];

      const Tensor<1, dim> &density_test_function_gradient = scratch.grad_phi_density[i];
      const double          density_test_function = scratch.phi_density[i];


      for (const auto j: scratch.fe_values.dof_indices())
      {
        // matrix step 1: hydrodynamic part
        double matrix = compute_hydrodynamic_matrix(scratch.phi_velocity[j],
                                                    scratch.grad_phi_velocity[j],
                                                    velocity_test_function,
                                                    velocity_test_function_gradient,
                                                    scratch.present_velocity_values[q],
                                                    scratch.present_velocity_gradients[q],
                                                    scratch.gravity_field_values[q],
                                                    scratch.phi_pressure[j],
                                                    scratch.phi_density[j],
                                                    pressure_test_function,
                                                    nu,
                                                    this->froude_number,
                                                    use_newton_linearization,
                                                    background_velocity_value,
                                                    background_velocity_gradient,
                                                    scratch.angular_velocity_value,
                                                    this->rossby_number);

        if (this->stabilization & (apply_supg|apply_pspg))
        {
          if (this->stabilization & apply_supg)
            optional_velocity_test_function_gradient = velocity_test_function_gradient;

          if (this->stabilization & apply_pspg)
            pressure_test_function_gradient = scratch.grad_phi_pressure->at(i);

          matrix += delta *
                    compute_hydrodynamic_residual_linearization_matrix(scratch.phi_velocity[j],
                                                                       scratch.grad_phi_velocity[j],
                                                                       scratch.laplace_phi_velocity->at(j),
                                                                       scratch.grad_phi_pressure->at(j),
                                                                       scratch.present_velocity_values[q],
                                                                       scratch.present_velocity_gradients[q],
                                                                       scratch.gravity_field_values[q],
                                                                       scratch.phi_density[j],
                                                                       nu,
                                                                       this->froude_number,
                                                                       use_newton_linearization,
                                                                       optional_velocity_test_function_gradient,
                                                                       pressure_test_function_gradient,
                                                                       background_velocity_value,
                                                                       background_velocity_gradient,
                                                                       scratch.angular_velocity_value,
                                                                       this->rossby_number);

          if (this->stabilization & apply_supg)
            matrix += delta * scratch.present_strong_residuals->at(q) *
                      (*optional_velocity_test_function_gradient * scratch.phi_velocity[j]);
        }

        if (this->stabilization & apply_grad_div)
          matrix += this->mu * Hydrodynamic::
                    compute_grad_div_matrix(scratch.grad_phi_velocity[j],
                                            velocity_test_function_gradient);

        // matrix step 2: density part
        matrix += compute_density_matrix(scratch.grad_phi_density[j],
                                         scratch.phi_velocity[j],
                                         scratch.present_density_gradients[q],
                                         scratch.present_velocity_values[q],
                                         density_test_function,
                                         use_newton_linearization,
                                         reference_density_gradient,
                                         stratification_number,
                                         background_velocity_value);
        // standard stabilization terms
        matrix += delta_density *
                  compute_density_residual_linearization_matrix(scratch.grad_phi_density[j],
                                                                density_test_function_gradient,
                                                                scratch.phi_velocity[j],
                                                                scratch.present_density_gradients[q],
                                                                scratch.present_velocity_values[q],
                                                                nu_density,
                                                                use_newton_linearization,
                                                                reference_density_gradient,
                                                                stratification_number,
                                                                background_velocity_value);
        matrix += delta_density * scratch.present_strong_density_residuals[q] *
                  (scratch.phi_velocity[j] * density_test_function_gradient);

        data.local_matrix(i, j) += matrix * JxW;

      }

      // rhs step 1: hydrodynamic part
      double rhs = compute_hydrodynamic_rhs(velocity_test_function,
                                            velocity_test_function_gradient,
                                            scratch.present_velocity_values[q],
                                            scratch.present_velocity_gradients[q],
                                            scratch.gravity_field_values[q],
                                            scratch.present_pressure_values[q],
                                            scratch.present_density_values[q],
                                            pressure_test_function,
                                            nu,
                                            this->froude_number,
                                            background_velocity_value,
                                            background_velocity_gradient,
                                            body_force_value,
                                            scratch.angular_velocity_value,
                                            this->rossby_number);

      if (this->stabilization & (apply_supg|apply_pspg))
      {
        Tensor<1, dim> stabilization_test_function;

        if (this->stabilization & apply_supg)
        {
          stabilization_test_function += velocity_test_function_gradient * scratch.present_velocity_values[q];
          if (background_velocity_value)
            stabilization_test_function += velocity_test_function_gradient * *background_velocity_value;
        }
        if (this->stabilization & apply_pspg)
          stabilization_test_function += scratch.grad_phi_pressure->at(i);

        rhs -= delta * scratch.present_strong_residuals->at(q) * stabilization_test_function;
      }

      if (this->stabilization & apply_grad_div)
        rhs += this->mu * Hydrodynamic::
               compute_grad_div_rhs(scratch.present_velocity_gradients[q],
                                    velocity_test_function_gradient);


      // rhs step 2: density part
      rhs += compute_density_rhs(scratch.present_density_gradients[q],
                                 scratch.present_velocity_values[q],
                                 density_test_function,
                                 reference_density_gradient,
                                 stratification_number,
                                 background_velocity_value);

      // standard stabilization terms
      {
        double stabilization_test_function{scratch.present_velocity_values[q] * density_test_function_gradient};

        if (background_velocity_value)
          stabilization_test_function += *background_velocity_value * density_test_function_gradient;

        rhs -= delta_density * scratch.present_strong_density_residuals[q] * stabilization_test_function;
      }

      data.local_rhs(i) += rhs * JxW;
    }
  } // end loop over cell quadrature points

  // Loop over the faces of the cell
  const typename VectorBoundaryConditions<dim>::NeumannBCMapping
  &neumann_bcs = this->velocity_boundary_conditions.neumann_bcs;
  const typename ScalarBoundaryConditions<dim>::BCMapping
  &dirichlet_bcs = density_boundary_conditions.dirichlet_bcs;
  if (!neumann_bcs.empty() || !dirichlet_bcs.empty())
    if (cell->at_boundary())
      for (const auto &face : cell->face_iterators())
      {
        if (face->at_boundary() &&
            (neumann_bcs.find(face->boundary_id()) != neumann_bcs.end() ||
             dirichlet_bcs.find(face->boundary_id()) != dirichlet_bcs.end()))
        {
          scratch.fe_face_values.reinit(cell, face);

          const types::boundary_id  boundary_id{face->boundary_id()};

          // Neumann boundary condition
          if (neumann_bcs.find(face->boundary_id()) != neumann_bcs.end())
          {
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
          }

          // Dirichlet boundary condition
          if (dirichlet_bcs.find(face->boundary_id()) != dirichlet_bcs.end())
          {
            dirichlet_bcs.at(boundary_id)->value_list(scratch.fe_face_values.get_quadrature_points(),
                                                      *scratch.density_boundary_values);
            // evaluate solution
            scratch.fe_face_values[density].get_function_values(this->evaluation_point,
                                                                *scratch.present_density_face_values);
            scratch.fe_face_values[velocity].get_function_values(this->evaluation_point,
                                                                 *scratch.present_velocity_face_values);
            // normal vectors
            scratch.face_normal_vectors = scratch.fe_face_values.get_normal_vectors();

            // Loop over face quadrature points
            for (const auto q: scratch.fe_face_values.quadrature_point_indices())
              if (scratch.face_normal_vectors->at(q) * scratch.present_velocity_face_values->at(q) < 0.)
              {
                // Extract the test function's values at the face quadrature points
                for (const auto i: scratch.fe_face_values.dof_indices())
                  scratch.phi_density[i] = scratch.fe_face_values[density].value(i,q);

                const double JxW_face{scratch.fe_face_values.JxW(q)};
                // Loop over the degrees of freedom
                for (const auto i: scratch.fe_face_values.dof_indices())
                {
                  for (const auto j: scratch.fe_face_values.dof_indices())
                    data.local_matrix(i, j) -= scratch.face_normal_vectors->at(q) *
                                               scratch.present_velocity_face_values->at(q) *
                                               scratch.phi_density[i] *
                                               scratch.phi_density[j] *
                                               JxW_face;
                  data.local_rhs(i) += scratch.present_velocity_face_values->at(q) *
                                       scratch.face_normal_vectors->at(q) *
                                       scratch.phi_density[i] *
                                       (scratch.present_density_face_values->at(q) - scratch.density_boundary_values->at(q)) *
                                       JxW_face;
                }
              } // Loop over face quadrature points
          }
        }
      } // Loop over the faces of the cell
}

// explicit instantiation
template void Solver<2>::assemble_local_system
(const typename DoFHandler<2>::active_cell_iterator &cell,
 AssemblyData::Matrix::Scratch<2> &,
 AssemblyBaseData::Matrix::Copy   &,
 const bool) const;
template void Solver<3>::assemble_local_system
(const typename DoFHandler<3>::active_cell_iterator &cell,
 AssemblyData::Matrix::Scratch<3> &,
 AssemblyBaseData::Matrix::Copy   &,
 const bool) const;

template void Solver<2>::assemble_system(const bool, const bool);
template void Solver<3>::assemble_system(const bool, const bool);

}  // namespace BuoyantHydrodynamic

