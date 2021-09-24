/*
 * assemble_rhs.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <assembly_functions.h>
#include <buoyant_hydrodynamic_solver.h>

namespace BuoyantHydrodynamic {

template <int dim>
void Solver<dim>::assemble_rhs(const bool use_homogeneous_constraints)
{
  if (this->verbose)
    std::cout << "    Assemble rhs..." << std::endl;

  if (this->angular_velocity_ptr != nullptr)
    AssertThrow(this->rossby_number > 0.0,
                ExcMessage("Non-vanishing Rossby number is required if the angular "
                           "velocity vector is specified."));
  AssertThrow(gravity_field_ptr != nullptr,
              ExcMessage("For a buoyant fluid, the gravity field must be specified."));
  AssertThrow(reference_density_ptr != nullptr,
              ExcMessage("For a buoyant fluid, the reference density field must be specified."));

  AssertThrow(this->froude_number > 0.0,
              ExcMessage("For a buoyant fluid, the Froude number must be specified."));
  AssertThrow(this->reynolds_number > 0.0,
              ExcMessage("The Reynolds must not vanish (stabilization is not "
                         "implemented yet)."));

  TimerOutput::Scope timer_section(this->computing_timer, "Assemble rhs");

  this->system_rhs = 0;

  const AffineConstraints<double> &constraints =
      (use_homogeneous_constraints? this->zero_constraints: this->nonzero_constraints);

  const FEValuesExtractors::Vector  velocity(0);
  const FEValuesExtractors::Scalar  pressure(dim);
  const FEValuesExtractors::Scalar  density(dim+1);

  const QGauss<dim>   quadrature_formula(this->velocity_fe_degree + 1);

  UpdateFlags update_flags = update_values|
                             update_gradients|
                             update_quadrature_points|
                             update_JxW_values;
  if (this->stabilization & (Hydrodynamic::apply_supg|Hydrodynamic::apply_pspg))
    update_flags |= update_hessians;

  FEValues<dim> fe_values(this->mapping,
                          *this->fe_system,
                          quadrature_formula,
                          update_flags);

  const QGauss<dim-1>   face_quadrature_formula(this->velocity_fe_degree + 1);

  FEFaceValues<dim>     fe_face_values(this->mapping,
                                       *this->fe_system,
                                       face_quadrature_formula,
                                       update_values|
                                       update_quadrature_points|
                                       update_JxW_values);

  const typename VectorBoundaryConditions<dim>::NeumannBCMapping
  &neumann_bcs = this->velocity_boundary_conditions.neumann_bcs;

  const unsigned int dofs_per_cell{this->fe_system->n_dofs_per_cell()};
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // shape functions
  std::vector<Tensor<1, dim>> phi_velocity(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_velocity(dofs_per_cell);
  std::vector<double>         div_phi_velocity(dofs_per_cell);
  std::vector<double>         phi_pressure(dofs_per_cell);
  std::vector<double>         phi_density(dofs_per_cell);
  std::vector<Tensor<1, dim>> grad_phi_density(dofs_per_cell);

  // stabilization related shape functions
  std::vector<Tensor<1, dim>> grad_phi_pressure;
  if (this->stabilization & Hydrodynamic::apply_pspg)
    grad_phi_pressure.resize(dofs_per_cell);

  // solution values
  const unsigned int n_q_points{quadrature_formula.size()};
  std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
  std::vector<double>         present_pressure_values(n_q_points);
  std::vector<double>         present_density_values(n_q_points);
  std::vector<Tensor<1, dim>> present_density_gradients(n_q_points);

  // stabilization related solution values
  std::vector<Tensor<1, dim>> present_velocity_laplaceans;
  std::vector<Tensor<1, dim>> present_pressure_gradients;
  if (this->stabilization & (Hydrodynamic::apply_supg|Hydrodynamic::apply_pspg))
  {
    present_velocity_laplaceans.resize(n_q_points);
    present_pressure_gradients.resize(n_q_points);
  }

  // source term values
  std::vector<Tensor<1,dim>>  reference_density_gradients(n_q_points);
  std::vector<Tensor<1,dim>>  gravity_field_values(n_q_points);
  std::vector<Tensor<1,dim>>  body_force_values;
  typename Utility::AngularVelocity<dim>::value_type angular_velocity_value;
  if (this->body_force_ptr != nullptr)
    body_force_values.resize(n_q_points);
  if (this->angular_velocity_ptr != nullptr)
    angular_velocity_value = this->angular_velocity_ptr->value();

  // source term face values
  const unsigned int n_face_q_points{face_quadrature_formula.size()};
  std::vector<Tensor<1, dim>> boundary_traction_values;
  if (!neumann_bcs.empty())
    boundary_traction_values.resize(n_face_q_points);

  const double nu{1.0 / this->reynolds_number};

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    const double delta{this->c * std::pow(cell->diameter(), 2)};
    const double delta_density{c_density * cell->diameter()};

    cell_rhs = 0;

    fe_values[velocity].get_function_values(this->evaluation_point,
                                            present_velocity_values);
    fe_values[velocity].get_function_gradients(this->evaluation_point,
                                               present_velocity_gradients);

    fe_values[pressure].get_function_values(this->evaluation_point,
                                            present_pressure_values);

    fe_values[density].get_function_values(this->evaluation_point,
                                           present_density_values);
    fe_values[density].get_function_gradients(this->evaluation_point,
                                              present_density_gradients);


    // stabilization related solution values
    if (this->stabilization & (Hydrodynamic::apply_supg|Hydrodynamic::apply_pspg))
    {
      fe_values[velocity].get_function_laplacians(this->evaluation_point,
                                                  present_velocity_laplaceans);
      fe_values[pressure].get_function_gradients(this->evaluation_point,
                                                 present_pressure_gradients);
    }

    // body force
    if (this->body_force_ptr != nullptr)
    {
      this->body_force_ptr->value_list(fe_values.get_quadrature_points(),
                                       body_force_values);

    }

    // reference density
    reference_density_ptr->gradient_list(fe_values.get_quadrature_points(),
                                         reference_density_gradients);

    // gravity field
    gravity_field_ptr->value_list(fe_values.get_quadrature_points(),
                                  gravity_field_values);

    for (const auto q: fe_values.quadrature_point_indices())
    {
      for (const auto i : fe_values.dof_indices())
      {
        phi_velocity[i] = fe_values[velocity].value(i, q);
        grad_phi_velocity[i] = fe_values[velocity].gradient(i, q);
        div_phi_velocity[i] = fe_values[velocity].divergence(i, q);
        phi_pressure[i] = fe_values[pressure].value(i, q);
        phi_density[i] = fe_values[density].value(i, q);
        grad_phi_density[i] = fe_values[density].gradient(i, q);

        // stabilization related shape functions
        if (this->stabilization & Hydrodynamic::apply_pspg)
          grad_phi_pressure[i] = fe_values[pressure].gradient(i, q);
      }

      const double JxW{fe_values.JxW(q)};

      for (const auto i: fe_values.dof_indices())
      {
        // rhs step 1: hydrodynamic part
        double rhs = Hydrodynamic::
                     compute_rhs(phi_velocity[i],
                                 grad_phi_velocity[i],
                                 present_velocity_values[q],
                                 present_velocity_gradients[q],
                                 present_pressure_values[q],
                                 phi_pressure[i],
                                 nu);

        // standard stabilization terms
        if (this->stabilization & Hydrodynamic::apply_supg)
          rhs += delta * Hydrodynamic::
                 compute_supg_rhs(grad_phi_velocity[i],
                                  present_velocity_values[q],
                                  present_velocity_gradients[q],
                                  present_velocity_laplaceans[q],
                                  present_pressure_gradients[q],
                                  nu);
        if (this->stabilization & Hydrodynamic::apply_pspg)
          rhs += delta * Hydrodynamic::
                 compute_pspg_rhs(present_velocity_values[q],
                                  present_velocity_gradients[q],
                                  present_velocity_laplaceans[q],
                                  grad_phi_pressure[i],
                                  present_pressure_gradients[q],
                                  nu);
        if (this->stabilization & Hydrodynamic::apply_grad_div)
          rhs += this->mu * Hydrodynamic::
                 compute_grad_div_rhs(present_velocity_gradients[q],
                                      grad_phi_velocity[i]);

        // body force
        if (this->body_force_ptr != nullptr)
        {
          Tensor<1, dim> body_force_test_function(phi_velocity[i]);

          // body force stabilization terms
          if (this->stabilization & Hydrodynamic::apply_supg)
            body_force_test_function += delta * grad_phi_velocity[i] * present_velocity_values[q];
          if (this->stabilization & Hydrodynamic::apply_pspg)
            body_force_test_function += delta * grad_phi_pressure[i];

          rhs += body_force_values[q] * body_force_test_function / std::pow(this->froude_number, 2);
        }

        // buoyancy term
        {
          Tensor<1, dim> buoyancy_test_function(phi_velocity[i]);

          // buoyancy stabilization terms
          if (this->stabilization & Hydrodynamic::apply_supg)
            buoyancy_test_function += delta * grad_phi_velocity[i] * present_velocity_values[q];
          if (this->stabilization & Hydrodynamic::apply_pspg)
            buoyancy_test_function += delta * grad_phi_pressure[i];

          rhs += present_density_values[q] * gravity_field_values[q] *
                 buoyancy_test_function / std::pow(this->froude_number, 2);
        }

        // Coriolis term
        if (this->angular_velocity_ptr != nullptr)
        {
          Tensor<1, dim> coriolis_term_test_function(phi_velocity[i]);

          // Coriolis stabilization terms
          if (this->stabilization & Hydrodynamic::apply_supg)
            coriolis_term_test_function += delta * grad_phi_velocity[i] *
                                           present_velocity_values[q];
          if (this->stabilization & Hydrodynamic::apply_pspg)
            coriolis_term_test_function += delta * grad_phi_pressure[i];

          if constexpr(dim == 2)
            rhs -= 2.0 / this->rossby_number * angular_velocity_value[0] *
                   cross_product_2d(-present_velocity_values[q]) *
                   coriolis_term_test_function;
          else if constexpr(dim == 3)
            rhs -= 2.0 / this->rossby_number *
                   cross_product_3d(angular_velocity_value, present_velocity_values[q]) *
                   coriolis_term_test_function;
        }

        // rhs step 2: density part
        rhs += compute_density_rhs(present_density_gradients[q],
                                   present_velocity_values[q],
                                   reference_density_gradients[q],
                                   phi_density[i],
                                   stratification_number);
        // standard stabilization terms
        rhs += delta_density *
               compute_density_supg_rhs(grad_phi_density[i],
                                        present_density_gradients[q],
                                        present_velocity_values[q],
                                        reference_density_gradients[q],
                                        stratification_number,
                                        nu_density);

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
              for (const auto i : fe_face_values.dof_indices())
                phi_velocity[i] = fe_face_values[velocity].value(i,q);

              const double JxW_face{fe_face_values.JxW(q)};

              // Loop over the degrees of freedom
              for (const auto i : fe_face_values.dof_indices())
                cell_rhs(i) += phi_velocity[i] *
                               boundary_traction_values[q] *
                               JxW_face;

            } // Loop over face quadrature points
          } // Loop over the faces of the cell

    cell->get_dof_indices(local_dof_indices);

    constraints.distribute_local_to_global(cell_rhs,
                                           local_dof_indices,
                                           this->system_rhs);
  } // end loop over cells
}

// explicit instantiation
template void Solver<2>::assemble_rhs(const bool);
template void Solver<3>::assemble_rhs(const bool);

}  // namespace BuoyantHydrodynamic

