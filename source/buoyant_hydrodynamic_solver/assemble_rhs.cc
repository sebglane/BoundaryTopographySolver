/*
 * assemble_rhs.cc
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

  // Initiate the quadrature formula
  const QGauss<dim>   quadrature_formula(this->velocity_fe_degree + 1);

  // Initiate the face quadrature formula
  const QGauss<dim-1>   face_quadrature_formula(this->velocity_fe_degree + 1);

  // Set up the lambda function for the local assembly operation
  using Scratch = AssemblyData::RightHandSide::Scratch<dim>;
  using Copy = AssemblyBaseData::RightHandSide::Copy;
  auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
         Scratch  &scratch,
         Copy     &data)
         {
            assemble_local_rhs(cell, scratch, data);
         };

  // Set up the lambda function for the copy local to global operation
  auto copier = [this, use_homogeneous_constraints](const Copy   &data)
      {
        this->copy_local_to_global_rhs(data, use_homogeneous_constraints);
      };

  // Assemble using the WorkStream approach
  UpdateFlags update_flags = update_values|
                             update_gradients|
                             update_quadrature_points|
                             update_JxW_values;
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
           !density_boundary_conditions.dirichlet_bcs.empty()),
   Copy(this->fe_system->n_dofs_per_cell()));
}



template <int dim>
void Solver<dim>::assemble_local_rhs
(const typename DoFHandler<dim>::active_cell_iterator &cell,
 AssemblyData::RightHandSide::Scratch<dim> &scratch,
 AssemblyBaseData::RightHandSide::Copy     &data) const
{
  cell->get_dof_indices(data.local_dof_indices);

  const FEValuesExtractors::Vector  velocity(0);
  const FEValuesExtractors::Scalar  pressure(dim);
  const FEValuesExtractors::Scalar  density(dim+1);

  const double nu{1.0 / this->reynolds_number};
  const double delta{this->c * std::pow(cell->diameter(), 2)};
  const double delta_density{c_density * cell->diameter()};

  scratch.fe_values.reinit(cell);

  data.local_rhs = 0;

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
                                                        scratch.present_velocity_laplaceans);
    scratch.fe_values[pressure].get_function_gradients(this->evaluation_point,
                                                       scratch.present_pressure_gradients);
  }

  // body force
  if (this->body_force_ptr != nullptr)
    this->body_force_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                     scratch.body_force_values);

  // Coriolis term
  if (this->angular_velocity_ptr != nullptr)
    scratch.angular_velocity_value = this->angular_velocity_ptr->value();

  // reference density
  reference_density_ptr->gradient_list(scratch.fe_values.get_quadrature_points(),
                                       scratch.reference_density_gradients);

  // gravity field
  gravity_field_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                scratch.gravity_field_values);

  for (const auto q: scratch.fe_values.quadrature_point_indices())
  {
    for (const auto i : scratch.fe_values.dof_indices())
    {
      scratch.phi_velocity[i] = scratch.fe_values[velocity].value(i, q);
      scratch.grad_phi_velocity[i] = scratch.fe_values[velocity].gradient(i, q);
      scratch.div_phi_velocity[i] = scratch.fe_values[velocity].divergence(i, q);
      scratch.phi_pressure[i] = scratch.fe_values[pressure].value(i, q);
      scratch.phi_density[i] = scratch.fe_values[density].value(i, q);
      scratch.grad_phi_density[i] = scratch.fe_values[density].gradient(i, q);

      // stabilization related shape functions
      if (this->stabilization & apply_pspg)
        scratch.grad_phi_pressure[i] = scratch.fe_values[pressure].gradient(i, q);
    }

    const double JxW{scratch.fe_values.JxW(q)};

    for (const auto i: scratch.fe_values.dof_indices())
    {
      // rhs step 1: hydrodynamic part
      double rhs = Hydrodynamic::
                   compute_rhs(scratch.phi_velocity[i],
                               scratch.grad_phi_velocity[i],
                               scratch.present_velocity_values[q],
                               scratch.present_velocity_gradients[q],
                               scratch.present_pressure_values[q],
                               scratch.phi_pressure[i],
                               nu);

      // standard stabilization terms
      if (this->stabilization & apply_supg)
        rhs += delta * Hydrodynamic::
               compute_supg_rhs(scratch.grad_phi_velocity[i],
                                scratch.present_velocity_values[q],
                                scratch.present_velocity_gradients[q],
                                scratch.present_velocity_laplaceans[q],
                                scratch.present_pressure_gradients[q],
                                nu);
      if (this->stabilization & apply_pspg)
        rhs += delta * Hydrodynamic::
               compute_pspg_rhs(scratch.present_velocity_values[q],
                                scratch.present_velocity_gradients[q],
                                scratch.present_velocity_laplaceans[q],
                                scratch.grad_phi_pressure[i],
                                scratch.present_pressure_gradients[q],
                                nu);
      if (this->stabilization & apply_grad_div)
        rhs += this->mu * Hydrodynamic::
               compute_grad_div_rhs(scratch.present_velocity_gradients[q],
                                    scratch.grad_phi_velocity[i]);

      // body force
      if (this->body_force_ptr != nullptr)
      {
        Tensor<1, dim> body_force_test_function(scratch.phi_velocity[i]);

        // body force stabilization terms
        if (this->stabilization & apply_supg)
          body_force_test_function += delta * scratch.grad_phi_velocity[i] * scratch.present_velocity_values[q];
        if (this->stabilization & apply_pspg)
          body_force_test_function += delta * scratch.grad_phi_pressure[i];

        rhs += scratch.body_force_values[q] * body_force_test_function / std::pow(this->froude_number, 2);
      }

      // buoyancy term
      {
        Tensor<1, dim> buoyancy_test_function(scratch.phi_velocity[i]);

        // buoyancy stabilization terms
        if (this->stabilization & apply_supg)
          buoyancy_test_function += delta * scratch.grad_phi_velocity[i] * scratch.present_velocity_values[q];
        if (this->stabilization & apply_pspg)
          buoyancy_test_function += delta * scratch.grad_phi_pressure[i];

        rhs += scratch.present_density_values[q] * scratch.gravity_field_values[q] *
               buoyancy_test_function / std::pow(this->froude_number, 2);
      }

      // Coriolis term
      if (this->angular_velocity_ptr != nullptr)
      {
        Tensor<1, dim> coriolis_term_test_function(scratch.phi_velocity[i]);

        // Coriolis stabilization terms
        if (this->stabilization & apply_supg)
          coriolis_term_test_function += delta * scratch.grad_phi_velocity[i] *
                                         scratch.present_velocity_values[q];
        if (this->stabilization & apply_pspg)
          coriolis_term_test_function += delta * scratch.grad_phi_pressure[i];

        if constexpr(dim == 2)
          rhs -= 2.0 / this->rossby_number * scratch.angular_velocity_value[0] *
                 cross_product_2d(-scratch.present_velocity_values[q]) *
                 coriolis_term_test_function;
        else if constexpr(dim == 3)
          rhs -= 2.0 / this->rossby_number *
                 cross_product_3d(scratch.angular_velocity_value, scratch.present_velocity_values[q]) *
                 coriolis_term_test_function;
      }

      // rhs step 2: density part
      rhs += compute_density_rhs(scratch.present_density_gradients[q],
                                 scratch.present_velocity_values[q],
                                 scratch.reference_density_gradients[q],
                                 scratch.phi_density[i],
                                 stratification_number);
      // standard stabilization terms
      rhs += delta_density *
             compute_density_supg_rhs(scratch.grad_phi_density[i],
                                      scratch.present_density_gradients[q],
                                      scratch.present_velocity_values[q],
                                      scratch.reference_density_gradients[q],
                                      stratification_number,
                                      nu_density);

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
                                                    scratch.boundary_traction_values);
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
                                     scratch.boundary_traction_values[q] *
                                     JxW_face;

            } // Loop over face quadrature points
          }

          // Dirichlet boundary condition
          if (dirichlet_bcs.find(face->boundary_id()) != dirichlet_bcs.end())
          {
            dirichlet_bcs.at(boundary_id)->value_list(scratch.fe_face_values.get_quadrature_points(),
                                                      scratch.density_boundary_values);
            // evaluate solution
            scratch.fe_face_values[density].get_function_values(this->evaluation_point,
                                                                scratch.present_density_face_values);
            scratch.fe_face_values[velocity].get_function_values(this->evaluation_point,
                                                                 scratch.present_velocity_face_values);
            // normal vectors
            scratch.face_normal_vectors = scratch.fe_face_values.get_normal_vectors();

            // Loop over face quadrature points
            for (const auto q: scratch.fe_face_values.quadrature_point_indices())
              if (scratch.face_normal_vectors[q] * scratch.present_velocity_face_values[q] < 0.)
              {
                // Extract the test function's values at the face quadrature points
                for (const auto i: scratch.fe_face_values.dof_indices())
                  scratch.phi_density[i] = scratch.fe_face_values[density].value(i,q);

                const double JxW_face{scratch.fe_face_values.JxW(q)};
                // Loop over the degrees of freedom
                for (const auto i: scratch.fe_face_values.dof_indices())
                  data.local_rhs(i) += scratch.present_velocity_face_values[q] *
                                       scratch.face_normal_vectors[q] *
                                       scratch.phi_density[i] *
                                       (scratch.present_density_face_values[q] - scratch.density_boundary_values[q]) *
                                       JxW_face;
              } // Loop over face quadrature points
          }
        }
      } // Loop over the faces of the cell
}

// explicit instantiation
template void Solver<2>::assemble_local_rhs
(const typename DoFHandler<2>::active_cell_iterator &cell,
 AssemblyData::RightHandSide::Scratch<2> &scratch,
 AssemblyBaseData::RightHandSide::Copy     &data) const;
template void Solver<3>::assemble_local_rhs
(const typename DoFHandler<3>::active_cell_iterator &cell,
 AssemblyData::RightHandSide::Scratch<3> &scratch,
 AssemblyBaseData::RightHandSide::Copy     &data) const;

template void Solver<2>::assemble_rhs(const bool);
template void Solver<3>::assemble_rhs(const bool);

}  // namespace BuoyantHydrodynamic

