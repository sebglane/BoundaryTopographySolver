/*
 * assemble_system.cc
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
void Solver<dim>::assemble_system(const bool use_homogeneous_constraints)
{
  if (this->verbose)
    std::cout << "    Assemble linear system..." << std::endl;

  AssertThrow(gravity_field_ptr != nullptr,
              ExcMessage("For a buoyant fluid, the gravity field must be specified."));
  AssertThrow(reference_density_ptr != nullptr,
              ExcMessage("For a buoyant fluid, the reference density field must be specified."));

  AssertThrow(this->froude_number > 0.0,
              ExcMessage("For a buoyant fluid, the Froude number must be specified."));
//  AssertThrow(stratification_number != 0.0,
//              ExcMessage("For a buoyant fluid, the stratification number must "
//                         "be specified."));
  AssertThrow(this->reynolds_number > 0.0,
              ExcMessage("The Reynolds must not vanish (stabilization is not "
                         "implemented yet)."));

  TimerOutput::Scope timer_section(this->computing_timer, "Assemble system");

  this->system_matrix = 0;
  this->system_rhs = 0;

  const AffineConstraints<double> &constraints =
      (use_homogeneous_constraints ? this->zero_constraints: this->nonzero_constraints);

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
                                       update_normal_vectors|
                                       update_quadrature_points|
                                       update_JxW_values);

  const typename VectorBoundaryConditions<dim>::NeumannBCMapping
  &neumann_bcs = this->velocity_boundary_conditions.neumann_bcs;

  const typename ScalarBoundaryConditions<dim>::BCMapping
  &dirichlet_bcs = density_boundary_conditions.dirichlet_bcs;

  const unsigned int dofs_per_cell{this->fe_system->n_dofs_per_cell()};
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
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
  std::vector<Tensor<1, dim>> laplace_phi_velocity;
  if (this->stabilization & (Hydrodynamic::apply_supg|Hydrodynamic::apply_pspg))
    grad_phi_pressure.resize(dofs_per_cell);
  if (this->stabilization & (Hydrodynamic::apply_supg|Hydrodynamic::apply_pspg))
    laplace_phi_velocity.resize(dofs_per_cell);

  // solution values
  const unsigned int n_q_points = quadrature_formula.size();
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
  if (this->body_force_ptr != nullptr)
    body_force_values.resize(n_q_points);

  // solution face values
  const unsigned int n_face_q_points{face_quadrature_formula.size()};
  std::vector<double>         present_density_face_values;
  std::vector<Tensor<1, dim>> present_velocity_face_values;
  std::vector<Tensor<1, dim>> face_normal_vectors;
  if (!dirichlet_bcs.empty())
  {
    present_density_face_values.resize(n_face_q_points);
    face_normal_vectors.resize(n_face_q_points);
    present_velocity_face_values.resize(n_face_q_points);
  }

  // source term face values
  std::vector<double> boundary_values;
  if (!dirichlet_bcs.empty())
    boundary_values.resize(n_face_q_points);
  std::vector<Tensor<1, dim>> boundary_traction_values;
  if (!neumann_bcs.empty())
    boundary_traction_values.resize(n_face_q_points);

  const double nu{1.0 / this->reynolds_number};

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    const double delta{this->c * std::pow(cell->diameter(), 2)};
    const double delta_density{c_density * cell->diameter()};

    cell_matrix = 0;
    cell_rhs = 0;

    // solution values
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
      for (const auto i: fe_values.dof_indices())
      {
        phi_velocity[i] = fe_values[velocity].value(i, q);
        grad_phi_velocity[i] = fe_values[velocity].gradient(i, q);
        div_phi_velocity[i] = fe_values[velocity].divergence(i, q);
        phi_pressure[i] = fe_values[pressure].value(i, q);
        phi_density[i] = fe_values[density].value(i, q);
        grad_phi_density[i] = fe_values[density].gradient(i, q);

        // stabilization related shape functions
        if (this->stabilization & (Hydrodynamic::apply_supg|Hydrodynamic::apply_pspg))
          grad_phi_pressure[i] = fe_values[pressure].gradient(i, q);
        if (this->stabilization & (Hydrodynamic::apply_supg|Hydrodynamic::apply_pspg))
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

        const Tensor<1, dim> &density_test_function_gradient = grad_phi_density[i];
        const double          density_test_function = phi_density[i];


        for (const auto j: fe_values.dof_indices())
        {
          // matrix step 1: hydrodynamic part
          double matrix = Hydrodynamic::
                          compute_matrix(phi_velocity[j],
                                         grad_phi_velocity[j],
                                         velocity_test_function,
                                         velocity_test_function_gradient,
                                         present_velocity_values[q],
                                         present_velocity_gradients[q],
                                         phi_pressure[j],
                                         pressure_test_function,
                                         nu);

          // standard stabilization terms
          if (this->stabilization & Hydrodynamic::apply_supg)
            matrix += delta * Hydrodynamic::
                      compute_supg_matrix(phi_velocity[j],
                                          grad_phi_velocity[j],
                                          laplace_phi_velocity[j],
                                          velocity_test_function_gradient,
                                          present_velocity_values[q],
                                          present_velocity_gradients[q],
                                          present_velocity_laplaceans[q],
                                          grad_phi_pressure[j],
                                          present_pressure_gradients[q],
                                          nu);
          if (this->stabilization & Hydrodynamic::apply_pspg)
            matrix += delta * Hydrodynamic::
                      compute_pspg_matrix(phi_velocity[j],
                                          grad_phi_velocity[j],
                                          laplace_phi_velocity[j],
                                          present_velocity_values[q],
                                          present_velocity_gradients[q],
                                          grad_phi_pressure[i],
                                          grad_phi_pressure[j],
                                          nu);
          if (this->stabilization & Hydrodynamic::apply_grad_div)
            matrix += this->mu * Hydrodynamic::
                      compute_grad_div_matrix(grad_phi_velocity[j],
                                              velocity_test_function_gradient);

          // body force stabilization terms
          if (this->body_force_ptr != nullptr && (this->stabilization & Hydrodynamic::apply_supg))
            matrix -= delta * body_force_values[q] *
                      velocity_test_function_gradient * phi_velocity[j] /
                      std::pow(this->froude_number, 2);

          // buoyancy term
          matrix -= phi_density[j] * gravity_field_values[q] *
                    velocity_test_function / std::pow(this->froude_number, 2);

          // buoyancy stabilization terms
          if (this->stabilization & (Hydrodynamic::apply_supg|Hydrodynamic::apply_pspg))
          {
            Tensor<1, dim> buoyancy_test_function;

            if (this->stabilization & Hydrodynamic::apply_supg)
              buoyancy_test_function += velocity_test_function_gradient * present_velocity_values[q];

            if (this->stabilization & Hydrodynamic::apply_pspg)
              buoyancy_test_function += grad_phi_pressure[i];

            matrix -= delta * (  present_density_values[q] * gravity_field_values[q] *
                                 velocity_test_function_gradient * phi_velocity[j]
                               + phi_density[j] * gravity_field_values[q] * buoyancy_test_function) /
                              std::pow(this->froude_number, 2);
          }

          // matrix step 2: density part
          matrix += compute_density_matrix(grad_phi_density[j],
                                           phi_velocity[j],
                                           present_density_gradients[q],
                                           present_velocity_values[q],
                                           reference_density_gradients[q],
                                           density_test_function,
                                           stratification_number);
          // standard stabilization terms
          matrix += delta_density * compute_density_supg_matrix(grad_phi_density[j],
                                                                density_test_function_gradient,
                                                                phi_velocity[j],
                                                                present_density_gradients[q],
                                                                present_velocity_values[q],
                                                                reference_density_gradients[q],
                                                                stratification_number,
                                                                nu_density);

          cell_matrix(i, j) += matrix * JxW;

        }

        // rhs step 1: hydrodynamic part
        double rhs = Hydrodynamic::
                     compute_rhs(velocity_test_function,
                                 velocity_test_function_gradient,
                                 present_velocity_values[q],
                                 present_velocity_gradients[q],
                                 present_pressure_values[q],
                                 pressure_test_function,
                                 nu);

        // standard stabilization terms
        if (this->stabilization & Hydrodynamic::apply_supg)
          rhs += delta * Hydrodynamic::
                 compute_supg_rhs(velocity_test_function_gradient,
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
                                      velocity_test_function_gradient);

        // body force
        if (this->body_force_ptr != nullptr)
        {
          Tensor<1, dim> body_force_test_function(phi_velocity[i]);

          // body force stabilization terms
          if (this->stabilization & Hydrodynamic::apply_supg)
            body_force_test_function += delta * velocity_test_function_gradient * present_velocity_values[q];
          if (this->stabilization & Hydrodynamic::apply_pspg)
            body_force_test_function += delta * grad_phi_pressure[i];

          rhs += body_force_values[q] * body_force_test_function / std::pow(this->froude_number, 2);
        }

        /*
         *

        // buoyancy term
        {
          Tensor<1, dim> buoyancy_test_function(velocity_test_function);

          // buoyancy stabilization terms
          if (this->stabilization & Hydrodynamic::apply_supg)
            buoyancy_test_function += delta * velocity_test_function_gradient * present_velocity_values[q];
          if (this->stabilization & Hydrodynamic::apply_pspg)
            buoyancy_test_function += delta * grad_phi_pressure[i];

          rhs += present_density_values[q] * gravity_field_values[q] *
                 buoyancy_test_function / std::pow(this->froude_number, 2);
        }

         *
         */


        // rhs step 2: density part
        rhs += compute_density_rhs(present_density_gradients[q],
                                   present_velocity_values[q],
                                   reference_density_gradients[q],
                                   density_test_function,
                                   stratification_number);
        // standard stabilization terms
        rhs += delta_density * compute_density_supg_rhs(density_test_function_gradient,
                                                        present_density_gradients[q],
                                                        present_velocity_values[q],
                                                        reference_density_gradients[q],
                                                        stratification_number,
                                                        nu_density);

        cell_rhs(i) += rhs * JxW;
      }
    } // end loop over cell quadrature points

    // Loop over the faces of the cell
    if (!neumann_bcs.empty() || !dirichlet_bcs.empty())
      if (cell->at_boundary())
        for (const auto &face : cell->face_iterators())
        {
          if (face->at_boundary() &&
              (neumann_bcs.find(face->boundary_id()) != neumann_bcs.end() ||
               dirichlet_bcs.find(face->boundary_id()) != dirichlet_bcs.end()))
          {
            fe_face_values.reinit(cell, face);

            const types::boundary_id  boundary_id{face->boundary_id()};

            // Neumann boundary condition
            if (neumann_bcs.find(face->boundary_id()) != neumann_bcs.end())
            {
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
            }

            // Dirichlet boundary condition
            if (dirichlet_bcs.find(face->boundary_id()) != dirichlet_bcs.end())
            {
              dirichlet_bcs.at(boundary_id)->value_list(fe_face_values.get_quadrature_points(),
                                                        boundary_values);
              // evaluate solution
              fe_face_values[density].get_function_values(this->evaluation_point,
                                                          present_density_face_values);
              fe_face_values[velocity].get_function_values(this->evaluation_point,
                                                           present_velocity_face_values);
              // normal vectors
              face_normal_vectors = fe_face_values.get_normal_vectors();

              // Loop over face quadrature points
              for (const auto q: fe_face_values.quadrature_point_indices())
                if (face_normal_vectors[q] * present_velocity_face_values[q] < 0.)
                {
                  // Extract the test function's values at the face quadrature points
                  for (const auto i: fe_face_values.dof_indices())
                    phi_density[i] = fe_face_values[density].value(i,q);

                  const double JxW_face{fe_face_values.JxW(q)};
                  // Loop over the degrees of freedom
                  for (const auto i: fe_face_values.dof_indices())
                  {
                    for (const auto j: fe_face_values.dof_indices())
                      cell_matrix(i, j) -= face_normal_vectors[q] *
                                           present_velocity_face_values[q] *
                                           phi_density[i] *
                                           phi_density[j] *
                                           JxW_face;
                    cell_rhs(i) += present_velocity_face_values[q] *
                                   face_normal_vectors[q] *
                                   phi_density[i] *
                                   (present_density_face_values[q] - boundary_values[q]) *
                                   JxW_face;
                  }
                } // Loop over face quadrature points
            }
          }
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

}  // namespace BuoyantHydrodynamic

