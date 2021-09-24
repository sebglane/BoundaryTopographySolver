/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>
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

  const AffineConstraints<double> &constraints =
      (use_homogeneous_constraints ? this->zero_constraints: this->nonzero_constraints);

  const FEValuesExtractors::Vector  velocity(0);
  const FEValuesExtractors::Scalar  pressure(dim);

  const QGauss<dim>   quadrature_formula(velocity_fe_degree + 1);

  UpdateFlags update_flags = update_values|
                             update_gradients|
                             update_JxW_values;
  if (body_force_ptr != nullptr)
    update_flags |= update_quadrature_points;
  if (stabilization & (apply_supg|apply_pspg))
    update_flags |= update_hessians;

  FEValues<dim> fe_values(this->mapping,
                          *this->fe_system,
                          quadrature_formula,
                          update_flags);

  const QGauss<dim-1>   face_quadrature_formula(velocity_fe_degree + 1);

  FEFaceValues<dim>     fe_face_values(this->mapping,
                                       *this->fe_system,
                                       face_quadrature_formula,
                                       update_values|
                                       update_quadrature_points|
                                       update_JxW_values);

  const typename VectorBoundaryConditions<dim>::NeumannBCMapping
  &neumann_bcs = velocity_boundary_conditions.neumann_bcs;

  const unsigned int dofs_per_cell{this->fe_system->n_dofs_per_cell()};
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // shape functions
  std::vector<Tensor<1, dim>> phi_velocity(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_velocity(dofs_per_cell);
  std::vector<double>         div_phi_velocity(dofs_per_cell);
  std::vector<double>         phi_pressure(dofs_per_cell);

  // stabilization related shape functions
  std::vector<Tensor<1, dim>> grad_phi_pressure;
  std::vector<Tensor<1, dim>> laplace_phi_velocity;
  if (stabilization & (apply_supg|apply_pspg))
    grad_phi_pressure.resize(dofs_per_cell);
  if (stabilization & (apply_supg|apply_pspg))
    laplace_phi_velocity.resize(dofs_per_cell);

  // solution values
  const unsigned int n_q_points = quadrature_formula.size();
  std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
  std::vector<double>         present_pressure_values(n_q_points);

  // stabilization related solution values
  std::vector<Tensor<1, dim>> present_velocity_laplaceans;
  std::vector<Tensor<1, dim>> present_pressure_gradients;
  if (stabilization & (apply_supg|apply_pspg))
  {
    present_velocity_laplaceans.resize(n_q_points);
    present_pressure_gradients.resize(n_q_points);
  }

  // source term values
  std::vector<Tensor<1,dim>>  body_force_values;
  if (body_force_ptr != nullptr)
    body_force_values.resize(n_q_points);
  typename Utility::AngularVelocity<dim>::value_type angular_velocity_value;
  if (angular_velocity_ptr != nullptr)
    angular_velocity_value = angular_velocity_ptr->value();

  // source term face values
  const unsigned int n_face_q_points{face_quadrature_formula.size()};
  std::vector<Tensor<1, dim>> boundary_traction_values;
  if (!neumann_bcs.empty())
    boundary_traction_values.resize(n_face_q_points);

  const double nu{1.0 / reynolds_number};

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    const double delta{c * std::pow(cell->diameter(), 2)};

    cell_matrix = 0;
    cell_rhs = 0;

    // solution values
    fe_values[velocity].get_function_values(this->evaluation_point,
                                            present_velocity_values);
    fe_values[velocity].get_function_gradients(this->evaluation_point,
                                               present_velocity_gradients);

    fe_values[pressure].get_function_values(this->evaluation_point,
                                            present_pressure_values);

    // stabilization related solution values
    if (stabilization & (apply_supg|apply_pspg))
    {
      fe_values[velocity].get_function_laplacians(this->evaluation_point,
                                                  present_velocity_laplaceans);

      fe_values[pressure].get_function_gradients(this->evaluation_point,
                                                 present_pressure_gradients);
    }

    // body force
    if (body_force_ptr != nullptr)
    {
      body_force_ptr->value_list(fe_values.get_quadrature_points(),
                                 body_force_values);

    }

    for (const auto q: fe_values.quadrature_point_indices())
    {
      for (const auto i: fe_values.dof_indices())
      {
        phi_velocity[i] = fe_values[velocity].value(i, q);
        grad_phi_velocity[i] = fe_values[velocity].gradient(i, q);
        div_phi_velocity[i] = fe_values[velocity].divergence(i, q);
        phi_pressure[i] = fe_values[pressure].value(i, q);

        // stabilization related shape functions
        if (stabilization & (apply_supg|apply_pspg))
          grad_phi_pressure[i] = fe_values[pressure].gradient(i, q);
        if (stabilization & (apply_supg|apply_pspg))
        {
          const Tensor<3, dim> shape_hessian(fe_values[velocity].hessian(i, q));
          for (unsigned int d=0; d<dim; ++d)
            laplace_phi_velocity[i][d] = trace(shape_hessian[d]);
        }
      }

      const double JxW{fe_values.JxW(q)};

      for (const auto i: fe_values.dof_indices())
      {
        const Tensor<1, dim> &velocity_test_function = phi_velocity[i];
        const Tensor<2, dim> &velocity_test_function_gradient = grad_phi_velocity[i];

        const double          pressure_test_function = phi_pressure[i];

        for (const auto j: fe_values.dof_indices())
        {
          double matrix = compute_matrix(phi_velocity[j],
                                         grad_phi_velocity[j],
                                         velocity_test_function,
                                         velocity_test_function_gradient,
                                         present_velocity_values[q],
                                         present_velocity_gradients[q],
                                         phi_pressure[j],
                                         pressure_test_function,
                                         nu);

          if (stabilization & apply_supg)
            matrix += delta * compute_supg_matrix(phi_velocity[j],
                                                  grad_phi_velocity[j],
                                                  laplace_phi_velocity[j],
                                                  velocity_test_function_gradient,
                                                  present_velocity_values[q],
                                                  present_velocity_gradients[q],
                                                  present_velocity_laplaceans[q],
                                                  grad_phi_pressure[j],
                                                  present_pressure_gradients[q],
                                                  nu);
          if (stabilization & apply_pspg)
            matrix += delta * compute_pspg_matrix(phi_velocity[j],
                                                  grad_phi_velocity[j],
                                                  laplace_phi_velocity[j],
                                                  present_velocity_values[q],
                                                  present_velocity_gradients[q],
                                                  grad_phi_pressure[i],
                                                  grad_phi_pressure[j],
                                                  nu);
          if (stabilization & apply_grad_div)
            matrix += mu * compute_grad_div_matrix(grad_phi_velocity[j],
                                                   velocity_test_function_gradient);

          if (body_force_ptr != nullptr && (stabilization & apply_supg))
            matrix -= delta * body_force_values[q] *
                      velocity_test_function_gradient * phi_velocity[j] /
                      std::pow(this->froude_number, 2);

          // Coriolis term
          if (angular_velocity_ptr != nullptr)
          {
            Tensor<1, dim> coriolis_term_test_function(velocity_test_function);

            if (stabilization & apply_supg)
              coriolis_term_test_function += delta * velocity_test_function_gradient * present_velocity_values[q];

            if (stabilization & apply_pspg)
              coriolis_term_test_function += delta * grad_phi_pressure[i];

            if constexpr(dim == 2)
              matrix += 2.0 / rossby_number * angular_velocity_value[0] *
                        cross_product_2d(-phi_velocity[j]) *
                        coriolis_term_test_function;
            else if constexpr(dim == 3)
              matrix += 2.0 / rossby_number *
                        cross_product_3d(angular_velocity_value , phi_velocity[j]) *
                        coriolis_term_test_function;

            if (stabilization & apply_supg)
            {
              if constexpr(dim == 2)
                matrix += 2.0 * delta / rossby_number * angular_velocity_value[0] *
                          cross_product_2d(-present_velocity_values[q]) *
                          velocity_test_function_gradient * phi_velocity[j];
              else if constexpr(dim == 3)
                matrix += 2.0 * delta / rossby_number *
                          cross_product_3d(angular_velocity_value, present_velocity_values[q]) *
                          velocity_test_function_gradient * phi_velocity[j];
            }
          }

          cell_matrix(i, j) +=  matrix * JxW;
        }
        double rhs = compute_rhs(velocity_test_function,
                                 velocity_test_function_gradient,
                                 present_velocity_values[q],
                                 present_velocity_gradients[q],
                                 present_pressure_values[q],
                                 pressure_test_function,
                                 nu);

        if (stabilization & apply_supg)
          rhs += delta * compute_supg_rhs(velocity_test_function_gradient,
                                          present_velocity_values[q],
                                          present_velocity_gradients[q],
                                          present_velocity_laplaceans[q],
                                          present_pressure_gradients[q],
                                          nu);
        if (stabilization & apply_pspg)
          rhs += delta * compute_pspg_rhs(present_velocity_values[q],
                                          present_velocity_gradients[q],
                                          present_velocity_laplaceans[q],
                                          grad_phi_pressure[i],
                                          present_pressure_gradients[q],
                                          nu);
        if (stabilization & apply_grad_div)
          rhs += mu * compute_grad_div_rhs(present_velocity_gradients[q],
                                           velocity_test_function_gradient);
        // body force term
        if (body_force_ptr != nullptr)
        {
          Tensor<1, dim> body_force_test_function(velocity_test_function);

          if (stabilization & apply_supg)
            body_force_test_function += delta * velocity_test_function_gradient * present_velocity_values[q];

          if (stabilization & apply_pspg)
            body_force_test_function += delta * grad_phi_pressure[i];

          rhs += body_force_values[q] * body_force_test_function / std::pow(this->froude_number, 2);
        }

        // Coriolis term
        if (angular_velocity_ptr != nullptr)
        {
          Tensor<1, dim> coriolis_term_test_function(velocity_test_function);

          if (stabilization & apply_supg)
            coriolis_term_test_function += delta * velocity_test_function_gradient * present_velocity_values[q];

          if (stabilization & apply_pspg)
            coriolis_term_test_function += delta * grad_phi_pressure[i];

          if constexpr(dim == 2)
            rhs -= 2.0 * delta / rossby_number * angular_velocity_value[0] *
                   cross_product_2d(-present_velocity_values[q]) *
                   coriolis_term_test_function;
          else if constexpr(dim == 3)
            rhs -= 2.0 / rossby_number *
                   cross_product_3d(angular_velocity_value, present_velocity_values[q]) *
                   coriolis_term_test_function;
        }

        cell_rhs(i) += rhs * JxW;
      }
    } // end loop over cell quadrature points

    // Loop over the faces of the cell
    if (!neumann_bcs.empty())
      if (cell->at_boundary())
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() &&
              neumann_bcs.find(face->boundary_id()) != neumann_bcs.end())
          {
            // Neumann boundary condition
            fe_face_values.reinit(cell, face);

            const types::boundary_id  boundary_id{face->boundary_id()};
            neumann_bcs.at(boundary_id)->value_list(fe_face_values.get_quadrature_points(),
                                                    boundary_traction_values);

            // Loop over face quadrature points
            for (const auto q: fe_face_values.quadrature_point_indices())
            {
              // Extract the test function's values at the face quadrature points
              for (const auto i: fe_face_values.dof_indices())
                phi_velocity[i] = fe_face_values[velocity].value(i,q);

              const double JxW_face{fe_face_values.JxW(q)};

              // Loop over the degrees of freedom
              for (const auto i: fe_face_values.dof_indices())
                cell_rhs(i) += phi_velocity[i] *
                               boundary_traction_values[q] *
                               JxW_face;

            } // Loop over face quadrature points
          } // Loop over the faces of the cell

    cell->get_dof_indices(local_dof_indices);

    constraints.distribute_local_to_global(cell_matrix,
                                           cell_rhs,
                                           local_dof_indices,
                                           this->system_matrix,
                                           this->system_rhs);
  } // end loop over cells
}

// explicit instantiation
template void Solver<2>::assemble_system(const bool);
template void Solver<3>::assemble_system(const bool);

}  // namespace Hydrodynamic

