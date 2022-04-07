/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <assembly_functions.h>
#include <advection_solver.h>

namespace Advection {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::assemble_rhs(const bool use_homogeneous_constraints)
{
  if (this->verbose)
    this->pcout << "    Assemble rhs..." << std::endl;

  AssertThrow(advection_field_ptr != nullptr,
              ExcMessage("The advection field must be specified."));

  TimerOutput::Scope timer_section(this->computing_timer, "Assemble rhs");

  this->system_rhs = 0;

  const AffineConstraints<double> &constraints =
      (use_homogeneous_constraints? this->zero_constraints: this->nonzero_constraints);

  const FEValuesExtractors::Scalar  field(0);

  const QGauss<dim>   quadrature_formula(fe_degree + 1);

  FEValues<dim> fe_values(this->mapping,
                          *this->fe_system,
                          quadrature_formula,
                          update_values|
                          update_gradients|
                          update_quadrature_points|
                          update_JxW_values);

  const QGauss<dim-1>   face_quadrature_formula(fe_degree + 1);

  FEFaceValues<dim>     fe_face_values(this->mapping,
                                       *this->fe_system,
                                       face_quadrature_formula,
                                       update_values|
                                       update_normal_vectors|
                                       update_quadrature_points|
                                       update_JxW_values);

  const typename ScalarBoundaryConditions<dim>::BCMapping
  &dirichlet_bcs = boundary_conditions.dirichlet_bcs;

  const unsigned int dofs_per_cell{this->fe_system->n_dofs_per_cell()};
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double>         phi(dofs_per_cell);
  std::vector<Tensor<1, dim>> grad_phi(dofs_per_cell);

  const unsigned int n_q_points = quadrature_formula.size();
  std::vector<double>         present_values(n_q_points);
  std::vector<Tensor<1, dim>> present_gradients(n_q_points);
  std::vector<Tensor<1,dim>>  advection_field_values(n_q_points);

  std::vector<double>         source_term_values;
  if (source_term_ptr != nullptr)
    source_term_values.resize(n_q_points);

  const unsigned int n_face_q_points{face_quadrature_formula.size()};
  std::vector<double> present_face_values;
  std::vector<double> boundary_values;
  std::vector<Tensor<1, dim>> face_normal_vectors;
  std::vector<Tensor<1, dim>> face_advection_field_values;
  if (!dirichlet_bcs.empty())
  {
    present_face_values.resize(n_face_q_points);
    boundary_values.resize(n_face_q_points);
    face_normal_vectors.resize(n_face_q_points);
    face_advection_field_values.resize(n_face_q_points);
  }

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  if (cell->is_locally_owned())
  {
    fe_values.reinit(cell);

    cell_rhs = 0;

    fe_values[field].get_function_values(this->evaluation_point,
                                         present_values);
    fe_values[field].get_function_gradients(this->evaluation_point,
                                            present_gradients);

    // body force
    if (source_term_ptr != nullptr)
    {
      source_term_ptr->value_list(fe_values.get_quadrature_points(),
                                  source_term_values);

    }

    // advection field
    advection_field_ptr->value_list(fe_values.get_quadrature_points(),
                                    advection_field_values);

    // stabilization parameter
    const double delta = compute_stabilization_parameter(advection_field_values,
                                                         cell->diameter());
    Assert(delta > 0.0, ExcLowerRangeType<double>(0.0, delta));

    for (const auto q: fe_values.quadrature_point_indices())
    {
      for (const auto i: fe_values.dof_indices())
      {
        phi[i] = fe_values[field].value(i, q);
        grad_phi[i] = fe_values[field].gradient(i, q);
      }

      const double JxW{fe_values.JxW(q)};

      for (const auto i: fe_values.dof_indices())
      {
        double rhs = compute_rhs(grad_phi[i],
                                 present_gradients[q],
                                 advection_field_values[q],
                                 phi[i],
                                 delta);

        if (source_term_ptr != nullptr)
          rhs += source_term_values[q] *
                 (phi[i] +
                  delta * advection_field_values[q] * grad_phi[i]);

        cell_rhs(i) += rhs * JxW;
      }
    } // end loop over cell quadrature points

    // Loop over the faces of the cell
    if (!dirichlet_bcs.empty())
      if (cell->at_boundary())
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() &&
              dirichlet_bcs.find(face->boundary_id()) != dirichlet_bcs.end())
          {
            fe_face_values.reinit(cell, face);

            // evaluate solution
            fe_face_values[field].get_function_values(this->evaluation_point,
                                                      present_face_values);
            // Dirichlet boundary condition
            const types::boundary_id  boundary_id{face->boundary_id()};
            dirichlet_bcs.at(boundary_id)->value_list(fe_face_values.get_quadrature_points(),
                                                      boundary_values);
            // advection field
            advection_field_ptr->value_list(fe_face_values.get_quadrature_points(),
                                            face_advection_field_values);
            // normal vectors
            face_normal_vectors = fe_face_values.get_normal_vectors();
            // Loop over face quadrature points
            for (const auto q: fe_face_values.quadrature_point_indices())
              if (face_normal_vectors[q] * face_advection_field_values[q] < 0.)
              {
                // Extract the test function's values at the face quadrature points
                for (const auto i: fe_face_values.dof_indices())
                  phi[i] = fe_face_values[field].value(i, q);

                const double JxW_face{fe_face_values.JxW(q)};

                // Loop over the degrees of freedom
                for (const auto i: fe_face_values.dof_indices())
                {
                  cell_rhs(i) += face_advection_field_values[q] *
                                 face_normal_vectors[q] *
                                 phi[i] *
                                 (present_face_values[q] - boundary_values[q]) *
                                 JxW_face;
                }

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

}  // namespace Advection

