/*
 * assemble_system.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <assembly_functions.h>
#include <hydrodynamic_solver.h>

namespace Hydrodynamic {

using TrilinosContainer = typename
                          SolverBase::
                          LinearAlgebraContainer<TrilinosWrappers::MPI::Vector,
                                                 TrilinosWrappers::SparseMatrix,
                                                 TrilinosWrappers::SparsityPattern>;



template <int dim>
using ParallelTriangulation =  parallel::distributed::Triangulation<dim>;



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void Solver<dim, TriangulationType, LinearAlgebraContainer>::assemble_system
(const bool use_homogeneous_constraints,
 const bool use_newton_linearization)
{
  if (this->verbose)
    this->pcout << "    Assemble linear system..." << std::endl;

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

  const bool use_stress_form{viscous_term_weak_form == ViscousTermWeakForm::stress};

  this->container.system_matrix = 0;
  this->container.system_rhs = 0;

  // Initiate the quadrature formula
  const QGauss<dim>   quadrature_formula(velocity_fe_degree + 1);

  // Initiate the face quadrature formula
  const QGauss<dim-1>   face_quadrature_formula(velocity_fe_degree + 1);

  // Set up the lambda function for the local assembly operation
  using Scratch = AssemblyData::Matrix::Scratch<dim>;
  using Copy = AssemblyBaseData::Matrix::Copy;
  auto worker =
      [this, use_newton_linearization, use_stress_form]
       (const typename DoFHandler<dim>::active_cell_iterator &cell,
        Scratch  &scratch,
        Copy     &data)
        {
          assemble_local_system(cell, scratch, data,
                                use_newton_linearization, use_stress_form);
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

  UpdateFlags face_update_flags = update_values|
                                  update_quadrature_points|
                                  update_JxW_values;
  if (include_boundary_stress_terms)
    face_update_flags |= update_gradients|
                         update_normal_vectors;

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
           stabilization,
           use_stress_form,
           background_velocity_ptr != nullptr,
           body_force_ptr != nullptr,
           include_boundary_stress_terms,
           include_boundary_stress_terms,
           !velocity_boundary_conditions.neumann_bcs.empty()),
   Copy(this->fe_system->n_dofs_per_cell()));

  this->container.system_matrix.compress(VectorOperation::add);
  this->container.system_rhs.compress(VectorOperation::add);
}



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void Solver<dim, TriangulationType, LinearAlgebraContainer>::assemble_local_system
(const typename DoFHandler<dim>::active_cell_iterator &cell,
 AssemblyData::Matrix::Scratch<dim> &scratch,
 AssemblyBaseData::Matrix::Copy     &data,
 const bool use_newton_linearization,
 const bool use_stress_form) const
{
  scratch.fe_values.reinit(cell);

  data.local_matrix = 0;
  data.local_rhs = 0;

  cell->get_dof_indices(data.local_dof_indices);

  const FEValuesExtractors::Vector  velocity(0);
  const FEValuesExtractors::Scalar  pressure(dim);

  const double nu{1.0 / reynolds_number};
  const double delta{c * std::pow(cell->diameter(), 2)};

  OptionalArgumentsWeakForm<dim> &weak_form_options = scratch.optional_arguments_weak_from;
  OptionalArgumentsStrongForm<dim> &strong_form_options = scratch.optional_arguments_strong_from;
  weak_form_options.use_stress_form = use_stress_form;
  strong_form_options.use_stress_form = use_stress_form;

  // solution values
  scratch.fe_values[velocity].get_function_values(this->container.evaluation_point,
                                                  scratch.present_velocity_values);
  scratch.fe_values[velocity].get_function_gradients(this->container.evaluation_point,
                                                     scratch.present_velocity_gradients);

  scratch.fe_values[pressure].get_function_values(this->container.evaluation_point,
                                                  scratch.present_pressure_values);

  // stress form
  if (use_stress_form)
    scratch.fe_values[velocity].get_function_symmetric_gradients(this->container.evaluation_point,
                                                                 scratch.present_sym_velocity_gradients);

  // stabilization related solution values
  if (stabilization & (apply_supg|apply_pspg))
  {
    scratch.fe_values[velocity].get_function_laplacians(this->container.evaluation_point,
                                                        scratch.present_velocity_laplaceans);

    scratch.fe_values[pressure].get_function_gradients(this->container.evaluation_point,
                                                       scratch.present_pressure_gradients);

    if (use_stress_form)
    {
      std::vector<Tensor<3, dim>> present_hessians(scratch.n_q_points);
      scratch.fe_values[velocity].get_function_hessians(this->container.evaluation_point,
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
  if (body_force_ptr != nullptr)
  {
    body_force_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                               *strong_form_options.body_force_values);
    strong_form_options.froude_number = froude_number;
    weak_form_options.froude_number = froude_number;
  }

  // background field
  if (background_velocity_ptr != nullptr)
  {
    background_velocity_ptr->value_list(scratch.fe_values.get_quadrature_points(),
                                        *strong_form_options.background_velocity_values);
    background_velocity_ptr->gradient_list(scratch.fe_values.get_quadrature_points(),
                                           *strong_form_options.background_velocity_gradients);
  }

  // Coriolis term
  if (angular_velocity_ptr != nullptr)
  {
    strong_form_options.angular_velocity = angular_velocity_ptr->value();
    strong_form_options.rossby_number = rossby_number;

    weak_form_options.angular_velocity = angular_velocity_ptr->value();
    weak_form_options.rossby_number = rossby_number;
  }

  if (stabilization & (apply_supg|apply_pspg))
    compute_strong_residual(scratch.present_velocity_values,
                            scratch.present_velocity_gradients,
                            scratch.present_velocity_laplaceans,
                            scratch.present_pressure_gradients,
                            scratch.present_strong_residuals,
                            nu,
                            strong_form_options);

  for (const auto q: scratch.fe_values.quadrature_point_indices())
  {
    for (const auto i: scratch.fe_values.dof_indices())
    {
      scratch.phi_velocity[i] = scratch.fe_values[velocity].value(i, q);
      scratch.grad_phi_velocity[i] = scratch.fe_values[velocity].gradient(i, q);
      scratch.div_phi_velocity[i] = scratch.fe_values[velocity].divergence(i, q);
      scratch.phi_pressure[i] = scratch.fe_values[pressure].value(i, q);

      // stress form
      if (use_stress_form)
        scratch.sym_grad_phi_velocity[i] = scratch.fe_values[velocity].symmetric_gradient(i, q);

      // stabilization related shape functions
      if (stabilization & (apply_supg|apply_pspg))
      {
        scratch.grad_phi_pressure[i] = scratch.fe_values[pressure].gradient(i, q);

        const Tensor<3, dim> shape_hessian(scratch.fe_values[velocity].hessian(i, q));
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

    const double JxW{scratch.fe_values.JxW(q)};

    for (const auto i: scratch.fe_values.dof_indices())
    {
      const Tensor<1, dim> &velocity_test_function = scratch.phi_velocity[i];
      const Tensor<2, dim> &velocity_test_function_gradient = scratch.grad_phi_velocity[i];

      const double          pressure_test_function = scratch.phi_pressure[i];

      // stress form
      if (use_stress_form)
        weak_form_options.velocity_test_function_symmetric_gradient =
            scratch.sym_grad_phi_velocity[i];

      // stabilization
      if (stabilization & apply_supg)
        weak_form_options.velocity_test_function_gradient =
            velocity_test_function_gradient;
      if (stabilization & apply_pspg)
        weak_form_options.pressure_test_function_gradient =
            scratch.grad_phi_pressure[i];

      for (const auto j: scratch.fe_values.dof_indices())
      {
        // stress form
        if (use_stress_form)
          weak_form_options.velocity_trial_function_symmetric_gradient =
                scratch.sym_grad_phi_velocity[j];

        double matrix = compute_matrix(scratch.phi_velocity[j],
                                       scratch.grad_phi_velocity[j],
                                       velocity_test_function,
                                       velocity_test_function_gradient,
                                       scratch.present_velocity_values[q],
                                       scratch.present_velocity_gradients[q],
                                       scratch.phi_pressure[j],
                                       pressure_test_function,
                                       nu,
                                       weak_form_options,
                                       use_newton_linearization);

        if (stabilization & (apply_supg|apply_pspg))
        {
          // stress form
          if (use_stress_form)
            weak_form_options.velocity_trial_function_grad_divergence =
                  scratch.grad_div_phi_velocity[j];

          matrix += delta *
                    compute_residual_linearization_matrix(scratch.phi_velocity[j],
                                                          scratch.grad_phi_velocity[j],
                                                          scratch.laplace_phi_velocity[j],
                                                          scratch.grad_phi_pressure[j],
                                                          scratch.present_velocity_values[q],
                                                          scratch.present_velocity_gradients[q],
                                                          nu,
                                                          weak_form_options,
                                                          use_newton_linearization);
          if (stabilization & apply_supg)
            matrix += delta * scratch.present_strong_residuals[q] *
                      (velocity_test_function_gradient * scratch.phi_velocity[j]);
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
                               weak_form_options);

      if (stabilization & (apply_supg|apply_pspg))
      {
        Tensor<1, dim> stabilization_test_function;

        if (stabilization & apply_supg)
        {
          stabilization_test_function += velocity_test_function_gradient *
                                         scratch.present_velocity_values[q];
          if (weak_form_options.background_velocity_value)
            stabilization_test_function += velocity_test_function_gradient *
                                           *weak_form_options.background_velocity_value;
        }
        if (stabilization & apply_pspg)
          stabilization_test_function += scratch.grad_phi_pressure[i];

        rhs -= delta * scratch.present_strong_residuals[q] * stabilization_test_function;
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
        } // Loop over the faces of the cell


  if (include_boundary_stress_terms && cell->at_boundary())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() &&
          std::find(boundary_stress_ids.begin(),
                    boundary_stress_ids.end(),
                    face->boundary_id()) != boundary_stress_ids.end())
      {
        // unconstrained boundary condition
        scratch.fe_face_values.reinit(cell, face);

        scratch.fe_face_values[pressure].get_function_values(this->container.evaluation_point,
                                                             scratch.present_pressure_face_values);

        // normal vectors
        scratch.face_normal_vectors = scratch.fe_face_values.get_normal_vectors();

        // compute present boundary traction
        if (use_stress_form)
        {
          scratch.fe_face_values[velocity].get_function_symmetric_gradients(this->container.evaluation_point,
                                                                            scratch.present_velocity_sym_face_gradients);

          for (const auto q: scratch.fe_face_values.quadrature_point_indices())
            scratch.boundary_traction_values[q] =
                - scratch.present_pressure_face_values[q] * scratch.face_normal_vectors[q]
                + 2.0 * nu * scratch.present_velocity_sym_face_gradients[q] * scratch.face_normal_vectors[q];
        }
        else
        {
          scratch.fe_face_values[velocity].get_function_gradients(this->container.evaluation_point,
                                                                  scratch.present_velocity_face_gradients);

          for (const auto q: scratch.fe_face_values.quadrature_point_indices())
            scratch.boundary_traction_values[q] =
                - scratch.present_pressure_face_values[q] * scratch.face_normal_vectors[q]
                + nu * scratch.present_velocity_face_gradients[q] * scratch.face_normal_vectors[q];
        }

        // Loop over face quadrature points
        for (const auto q: scratch.fe_face_values.quadrature_point_indices())
        {
          // Extract the test function's values at the face quadrature points
          for (const auto i: scratch.fe_face_values.dof_indices())
          {
            scratch.phi_velocity[i] = scratch.fe_face_values[velocity].value(i, q);
            scratch.phi_pressure[i] = scratch.fe_face_values[pressure].value(i, q);

            if (use_stress_form)
              scratch.sym_grad_phi_velocity[i] = scratch.fe_face_values[velocity].symmetric_gradient(i, q);
            else
              scratch.grad_phi_velocity[i] = scratch.fe_face_values[velocity].gradient(i, q);

          }

          const double JxW_face{scratch.fe_face_values.JxW(q)};

          // Loop over the degrees of freedom
          for (const auto i: scratch.fe_face_values.dof_indices())
          {
            if (use_stress_form)
              for (const auto j: scratch.fe_face_values.dof_indices())
                data.local_matrix(i, j) -=
                    (-scratch.phi_pressure[j] * scratch.face_normal_vectors[q] +
                     2.0 * nu * scratch.sym_grad_phi_velocity[j] * scratch.face_normal_vectors[q]) *
                     scratch.phi_velocity[i] * JxW_face;
            else
              for (const auto j: scratch.fe_face_values.dof_indices())
                data.local_matrix(i, j) -=
                    (-scratch.phi_pressure[j] * scratch.face_normal_vectors[q] +
                     nu * scratch.grad_phi_velocity[j] * scratch.face_normal_vectors[q]) *
                     scratch.phi_velocity[i] * JxW_face;

            data.local_rhs(i) += scratch.phi_velocity[i] *
                                 scratch.boundary_traction_values[q] *
                                 JxW_face;
          }

        } // Loop over face quadrature points
      } // Loop over the faces of the cell
}



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void Solver<dim, TriangulationType, LinearAlgebraContainer>::copy_local_to_global_system
(const AssemblyBaseData::Matrix::Copy     &data,
 const bool use_homogeneous_constraints)
{
  const AffineConstraints<double> &constraints =
      (use_homogeneous_constraints ? this->zero_constraints: this->nonzero_constraints);

  constraints.distribute_local_to_global(data.local_matrix,
                                         data.local_rhs,
                                         data.local_dof_indices,
                                         this->container.system_matrix,
                                         this->container.system_rhs);
}

// explicit instantiation
template
void
Solver<2>::
assemble_local_system
(const typename DoFHandler<2>::active_cell_iterator &cell,
 AssemblyData::Matrix::Scratch<2> &,
 AssemblyBaseData::Matrix::Copy   &,
 const bool, const bool) const;
template
void
Solver<3>::
assemble_local_system
(const typename DoFHandler<3>::active_cell_iterator &cell,
 AssemblyData::Matrix::Scratch<3> &,
 AssemblyBaseData::Matrix::Copy   &,
 const bool, const bool) const;

template
void
Solver<2, ParallelTriangulation<2>, TrilinosContainer>::
assemble_local_system
(const typename DoFHandler<2>::active_cell_iterator &cell,
 AssemblyData::Matrix::Scratch<2> &,
 AssemblyBaseData::Matrix::Copy   &,
 const bool, const bool) const;
template
void
Solver<3, ParallelTriangulation<3>, TrilinosContainer>::
assemble_local_system
(const typename DoFHandler<3>::active_cell_iterator &cell,
 AssemblyData::Matrix::Scratch<3> &,
 AssemblyBaseData::Matrix::Copy   &,
 const bool, const bool) const;

template
void
Solver<2>::
copy_local_to_global_system
(const AssemblyBaseData::Matrix::Copy &, const bool);
template
void
Solver<3>::
copy_local_to_global_system
(const AssemblyBaseData::Matrix::Copy &, const bool);

template
void
Solver<2, ParallelTriangulation<2>, TrilinosContainer>::
copy_local_to_global_system
(const AssemblyBaseData::Matrix::Copy &, const bool);
template
void
Solver<3, ParallelTriangulation<3>, TrilinosContainer>::
copy_local_to_global_system
(const AssemblyBaseData::Matrix::Copy &, const bool);

template
void
Solver<2>::
assemble_system
(const bool, const bool);
template
void
Solver<3>::
assemble_system(const bool, const bool);

template
void
Solver<2, ParallelTriangulation<2>, TrilinosContainer>::
assemble_system
(const bool, const bool);
template
void
Solver<3, ParallelTriangulation<3>, TrilinosContainer>::
assemble_system(const bool, const bool);

}  // namespace Hydrodynamic

