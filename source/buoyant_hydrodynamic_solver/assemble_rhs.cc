/*
 * assemble_rhs.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/filtered_iterator.h>

#include <assembly_functions.h>
#include <buoyant_hydrodynamic_solver.h>

namespace BuoyantHydrodynamic {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::assemble_rhs(const bool use_homogeneous_constraints)
{
  if (this->verbose)
    this->pcout << "    Assemble rhs..." << std::endl;

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

  TimerOutput::Scope timer_section(this->computing_timer, "Assemble rhs");

  const bool use_stress_form{this->viscous_term_weak_form == Hydrodynamic::ViscousTermWeakForm::stress};

  this->system_rhs = 0;

  // Initiate the quadrature formula
  const QGauss<dim>   quadrature_formula(this->velocity_fe_degree + 1);

  // Initiate the face quadrature formula
  const QGauss<dim-1>   face_quadrature_formula(this->velocity_fe_degree + 1);

  // Set up the lambda function for the local assembly operation
  using Scratch = AssemblyData::RightHandSide::Scratch<dim>;
  using Copy = AssemblyBaseData::RightHandSide::Copy;
  auto worker =
      [this, use_stress_form]
       (const typename DoFHandler<dim>::active_cell_iterator &cell,
        Scratch  &scratch,
        Copy     &data)
         {
            assemble_local_rhs(cell, scratch, data, use_stress_form);
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
  if (this->include_boundary_stress_terms)
    face_update_flags |= update_gradients|
                         update_normal_vectors;
  if (!density_boundary_conditions.dirichlet_bcs.empty())
    face_update_flags |= update_normal_vectors;

  using CellFilter = FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              this->dof_handler.begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              this->dof_handler.end()),
   worker,
   copier,
   Scratch(this->mapping,
           quadrature_formula,
           *this->fe_system,
           update_flags,
           face_quadrature_formula,
           face_update_flags,
           this->stabilization,
           use_stress_form,
           this->background_velocity_ptr != nullptr,
           this->body_force_ptr != nullptr,
           this->include_boundary_stress_terms,
           !this->velocity_boundary_conditions.neumann_bcs.empty(),
           gravity_field_ptr != nullptr,
           reference_density_ptr != nullptr,
           !density_boundary_conditions.dirichlet_bcs.empty()),
   Copy(this->fe_system->n_dofs_per_cell()));

  this->system_rhs.compress(VectorOperation::add);
}



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::assemble_local_rhs
(const typename DoFHandler<dim>::active_cell_iterator &cell,
 AssemblyData::RightHandSide::Scratch<dim> &scratch,
 AssemblyBaseData::RightHandSide::Copy     &data,
 const bool use_stress_form) const
{
  scratch.fe_values.reinit(cell);

  data.local_rhs = 0;

  cell->get_dof_indices(data.local_dof_indices);

  const FEValuesExtractors::Vector  velocity(0);
  const FEValuesExtractors::Scalar  pressure(dim);
  const FEValuesExtractors::Scalar  density(dim+1);

  const double nu{1.0 / this->reynolds_number};
  const double delta{this->c * std::pow(cell->diameter(), 2)};
  const double delta_density{c_density * std::pow(cell->diameter(), 2)};

  Hydrodynamic::OptionalArgumentsWeakForm<dim>   &weak_form_options = scratch.hydrodynamic_weak_form_options;
  Hydrodynamic::OptionalArgumentsStrongForm<dim> &strong_form_options = scratch.hydrodynamic_strong_form_options;
  weak_form_options.use_stress_form = use_stress_form;
  strong_form_options.use_stress_form = use_stress_form;

  BuoyantHydrodynamic::OptionalArgumentsWeakForm<dim>   &buoyancy_weak_form_options = scratch.weak_form_options;
  BuoyantHydrodynamic::OptionalArgumentsStrongForm<dim> &buoyancy_strong_form_options = scratch.strong_form_options;

  scratch.fe_values[velocity].get_function_values(this->evaluation_point,
                                                  scratch.present_velocity_values);
  scratch.fe_values[velocity].get_function_gradients(this->evaluation_point,
                                                     scratch.present_velocity_gradients);

  scratch.fe_values[pressure].get_function_values(this->evaluation_point,
                                                  scratch.present_pressure_values);

  // stress form
  if (use_stress_form)
    scratch.fe_values[velocity].get_function_symmetric_gradients(this->evaluation_point,
                                                                 scratch.present_sym_velocity_gradients);

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

    // stress form
    if (use_stress_form)
    {
      std::vector<Tensor<3, dim>> present_hessians(scratch.n_q_points);
      scratch.fe_values[velocity].get_function_hessians(this->evaluation_point,
                                                        present_hessians);

      std::vector<Tensor<1, dim>> &present_velocity_grad_divergences =
          strong_form_options.present_velocity_grad_divergences.value();
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
    this->body_force_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                     *strong_form_options.body_force_values);
    strong_form_options.froude_number = this->froude_number;
    weak_form_options.froude_number = this->froude_number;
  }

  // background field
  if (this->background_velocity_ptr != nullptr)
  {
    this->background_velocity_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                              *strong_form_options.background_velocity_values);
    this->background_velocity_ptr->gradient_list(scratch.fe_values.get_quadrature_points(),
                                                 *strong_form_options.background_velocity_gradients);
  }

  // Coriolis term
  if (this->angular_velocity_ptr != nullptr)
  {
    strong_form_options.angular_velocity = this->angular_velocity_ptr->value();
    strong_form_options.rossby_number = this->rossby_number;

    weak_form_options.angular_velocity = this->angular_velocity_ptr->value();
    weak_form_options.rossby_number = this->rossby_number;
  }

  // reference density
  if (reference_density_ptr != nullptr)
  {
    reference_density_ptr->gradient_list(scratch.fe_values.get_quadrature_points(),
                                         *buoyancy_strong_form_options.reference_density_gradients);

    buoyancy_strong_form_options.stratification_number = stratification_number;
    buoyancy_weak_form_options.stratification_number = stratification_number;
  }

  // gravity field
  if (gravity_field_ptr != nullptr)
  {
    gravity_field_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                  *buoyancy_strong_form_options.gravity_field_values);

    strong_form_options.froude_number = this->froude_number;
    weak_form_options.froude_number = this->froude_number;
  }

  // stabilization
  if (this->stabilization & (apply_supg|apply_pspg))
    compute_strong_hydrodynamic_residual(scratch.present_velocity_values,
                                         scratch.present_velocity_gradients,
                                         scratch.present_velocity_laplaceans,
                                         scratch.present_pressure_gradients,
                                         scratch.present_density_values,
                                         scratch.present_strong_residuals,
                                         nu,
                                         strong_form_options,
                                         buoyancy_strong_form_options);

  compute_strong_density_residual(scratch.present_density_gradients,
                                  scratch.present_velocity_values,
                                  scratch.present_strong_density_residuals,
                                  strong_form_options,
                                  buoyancy_strong_form_options);

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

      // stress form
      if (use_stress_form)
        scratch.sym_grad_phi_velocity[i] = scratch.fe_values[velocity].symmetric_gradient(i, q);

      // stabilization related shape functions
      if (this->stabilization & apply_pspg)
        scratch.grad_phi_pressure[i] = scratch.fe_values[pressure].gradient(i, q);
    }

    // stress form
    if (use_stress_form)
      weak_form_options.present_symmetric_velocity_gradient =
          scratch.present_sym_velocity_gradients[q];

    // background field
    if (strong_form_options.background_velocity_values)
      weak_form_options.background_velocity_value =
          strong_form_options.background_velocity_values->at(q);
    if (strong_form_options.background_velocity_gradients)
      weak_form_options.background_velocity_gradient =
          strong_form_options.background_velocity_gradients->at(q);

    // body force
    if (strong_form_options.body_force_values)
      weak_form_options.body_force_value =
          strong_form_options.body_force_values->at(q);

    // reference density
    if (buoyancy_strong_form_options.reference_density_gradients)
      buoyancy_weak_form_options.reference_density_gradient =
          buoyancy_strong_form_options.reference_density_gradients->at(q);

    // gravity field
    if (buoyancy_strong_form_options.gravity_field_values)
      buoyancy_weak_form_options.gravity_field_value =
          buoyancy_strong_form_options.gravity_field_values->at(q);


    const double JxW{scratch.fe_values.JxW(q)};

    for (const auto i: scratch.fe_values.dof_indices())
    {
      // stress form
      if (use_stress_form)
        weak_form_options.velocity_test_function_symmetric_gradient =
            scratch.sym_grad_phi_velocity[i];

      // rhs step 1: hydrodynamic part
      double rhs = compute_hydrodynamic_rhs(scratch.phi_velocity[i],
                                            scratch.grad_phi_velocity[i],
                                            scratch.present_velocity_values[q],
                                            scratch.present_velocity_gradients[q],
                                            scratch.present_pressure_values[q],
                                            scratch.present_density_values[q],
                                            scratch.phi_pressure[i],
                                            nu,
                                            weak_form_options,
                                            buoyancy_weak_form_options);

      if (this->stabilization & (apply_supg|apply_pspg))
      {
        Tensor<1, dim> stabilization_test_function;

        if (this->stabilization & apply_supg)
        {
          stabilization_test_function += scratch.grad_phi_velocity[i] *
                                         scratch.present_velocity_values[q];
          if (weak_form_options.background_velocity_value)
            stabilization_test_function += scratch.grad_phi_velocity[i] *
                                           *weak_form_options.background_velocity_value;
        }
        if (this->stabilization & apply_pspg)
          stabilization_test_function += scratch.grad_phi_pressure[i];

        rhs -= delta * scratch.present_strong_residuals[q] * stabilization_test_function;
      }

      if (this->stabilization & apply_grad_div)
        rhs += this->mu * Hydrodynamic::
               compute_grad_div_rhs(scratch.present_velocity_gradients[q],
                                    scratch.grad_phi_velocity[i]);


      // rhs step 2: density part
      rhs += compute_density_rhs(scratch.present_density_gradients[q],
                                 scratch.present_velocity_values[q],
                                 scratch.phi_density[i],
                                 weak_form_options,
                                 buoyancy_weak_form_options);

      // standard stabilization terms
      {
        double stabilization_test_function{scratch.present_velocity_values[q] * scratch.grad_phi_density[i]};

        if (weak_form_options.background_velocity_value)
          stabilization_test_function += *weak_form_options.background_velocity_value *
                                         scratch.grad_phi_density[i];

        rhs -= delta_density * scratch.present_strong_density_residuals[q] *
               stabilization_test_function;
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

  if (this->include_boundary_stress_terms && cell->at_boundary())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() &&
          std::find(this->boundary_stress_ids.begin(),
                    this->boundary_stress_ids.end(),
                    face->boundary_id()) != this->boundary_stress_ids.end())
      {
        // unconstrained boundary condition
        scratch.fe_face_values.reinit(cell, face);

        scratch.fe_face_values[pressure].get_function_values(this->evaluation_point,
                                                             scratch.present_pressure_face_values);

        // normal vectors
        scratch.face_normal_vectors = scratch.fe_face_values.get_normal_vectors();

        // compute present boundary traction
        if (use_stress_form)
        {
          scratch.fe_face_values[velocity].get_function_symmetric_gradients(this->evaluation_point,
                                                                            scratch.present_velocity_sym_face_gradients);

          for (const auto q: scratch.fe_face_values.quadrature_point_indices())
            scratch.boundary_traction_values[q] =
                - scratch.present_pressure_face_values[q] * scratch.face_normal_vectors[q]
                + 2.0 * nu * scratch.present_velocity_sym_face_gradients[q] * scratch.face_normal_vectors[q];
        }
        else
        {
          scratch.fe_face_values[velocity].get_function_gradients(this->evaluation_point,
                                                                  scratch.present_velocity_face_gradients);

          for (const auto q: scratch.fe_face_values.quadrature_point_indices())
            scratch.boundary_traction_values[q] =
                - scratch.present_pressure_face_values[q] * scratch.face_normal_vectors[q]
                + nu * scratch.present_velocity_face_gradients[q] * scratch.face_normal_vectors[q];
        }

        // Loop over face quadrature points
        for (const auto q: scratch.fe_face_values.quadrature_point_indices())
        {
          const double JxW_face{scratch.fe_face_values.JxW(q)};
          // Extract the test function's values at the face quadrature points
          for (const auto i: scratch.fe_face_values.dof_indices())
            scratch.phi_velocity[i] = scratch.fe_face_values[velocity].value(i, q);

          // Loop over the degrees of freedom
          for (const auto i: scratch.fe_face_values.dof_indices())
            data.local_rhs(i) += scratch.phi_velocity[i] *
                                 scratch.boundary_traction_values[q] *
                                 JxW_face;
        } // Loop over face quadrature points
      } // Loop over the faces of the cell
}

// explicit instantiation
template void Solver<2>::assemble_local_rhs
(const typename DoFHandler<2>::active_cell_iterator &cell,
 AssemblyData::RightHandSide::Scratch<2> &,
 AssemblyBaseData::RightHandSide::Copy   &,
 const bool) const;
template void Solver<3>::assemble_local_rhs
(const typename DoFHandler<3>::active_cell_iterator &cell,
 AssemblyData::RightHandSide::Scratch<3> &,
 AssemblyBaseData::RightHandSide::Copy   &,
 const bool) const;

template void Solver<2>::assemble_rhs(const bool);
template void Solver<3>::assemble_rhs(const bool);

}  // namespace BuoyantHydrodynamic

