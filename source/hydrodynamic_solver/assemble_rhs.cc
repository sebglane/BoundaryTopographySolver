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

namespace TopographyProblem {

template <int dim>
void HydrodynamicSolver<dim>::assemble_rhs(const bool use_homogeneous_constraints)
{
  if (this->verbose)
    std::cout << "    Assemble rhs..." << std::endl;

  TimerOutput::Scope timer_section(this->computing_timer, "Assemble rhs");

  this->system_rhs = 0;

  const AffineConstraints<double> &constraints =
      (use_homogeneous_constraints? this->zero_constraints: this->nonzero_constraints);

  const FEValuesExtractors::Vector  velocity(0);
  const FEValuesExtractors::Scalar  pressure(dim);

  const QGauss<dim>   quadrature_formula(velocity_fe_degree + 1);

  FEValues<dim> fe_values(this->mapping,
                          *this->fe_system,
                          quadrature_formula,
                          update_values|
                          update_gradients|
                          update_JxW_values);

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
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Tensor<1, dim>> phi_velocity(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_velocity(dofs_per_cell);
  std::vector<double>         div_phi_velocity(dofs_per_cell);
  std::vector<double>         phi_pressure(dofs_per_cell);

  std::vector<Tensor<1, dim>> phi_face_velocity;
  if (!neumann_bcs.empty())
    phi_face_velocity.resize(dofs_per_cell);

  const unsigned int n_q_points{quadrature_formula.size()};
  std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
  std::vector<double>         present_pressure_values(n_q_points);

  const unsigned int n_face_q_points{face_quadrature_formula.size()};
  std::vector<Tensor<1, dim>> boundary_traction_values;
  if (!neumann_bcs.empty())
    boundary_traction_values.resize(n_face_q_points);

  const double nu{1.0 / reynolds_number};

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    cell_rhs = 0;

    fe_values[velocity].get_function_values(this->evaluation_point,
                                            present_velocity_values);
    fe_values[velocity].get_function_gradients(this->evaluation_point,
                                               present_velocity_gradients);

    fe_values[pressure].get_function_values(this->evaluation_point,
                                            present_pressure_values);

    for (const unsigned int q: fe_values.quadrature_point_indices())
    {
      for (const unsigned int i : fe_values.dof_indices())
      {
        phi_velocity[i] = fe_values[velocity].value(i, q);
        grad_phi_velocity[i] = fe_values[velocity].gradient(i, q);
        div_phi_velocity[i] = fe_values[velocity].divergence(i, q);
        phi_pressure[i] = fe_values[pressure].value(i, q);
      }

      const double JxW{fe_values.JxW(q)};

      for (const unsigned int i : fe_values.dof_indices())
        cell_rhs(i) += compute_hydrodynamic_rhs(phi_velocity[i],
                                                grad_phi_velocity[i],
                                                present_velocity_values[q],
                                                present_velocity_gradients[q],
                                                present_pressure_values[q],
                                                phi_pressure[i],
                                                nu) * JxW;
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
            for (const unsigned int q: fe_face_values.quadrature_point_indices())
            {
              // Extract the test function's values at the face quadrature points
              for (const unsigned int i : fe_face_values.dof_indices())
                phi_face_velocity[i] = fe_face_values[velocity].value(i,q);

              const double JxW_face{fe_face_values.JxW(q)};

              // Loop over the degrees of freedom
              for (const unsigned int i : fe_face_values.dof_indices())
                cell_rhs(i) += phi_face_velocity[i] *
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
template void HydrodynamicSolver<2>::assemble_rhs(const bool);
template void HydrodynamicSolver<3>::assemble_rhs(const bool);

}  // namespace TopographyProblem

