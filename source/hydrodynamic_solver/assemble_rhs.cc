/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

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

  const unsigned int dofs_per_cell{this->fe_system->n_dofs_per_cell()};
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Tensor<1, dim>> phi_velocity(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_velocity(dofs_per_cell);
  std::vector<double>         div_phi_velocity(dofs_per_cell);
  std::vector<double>         phi_pressure(dofs_per_cell);

  const unsigned int n_q_points = quadrature_formula.size();
  std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
  std::vector<double>         present_velocity_divergences(n_q_points);
  std::vector<double>         present_pressure_values(n_q_points);

  const double nu{1.0 / reynolds_number};

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    cell_rhs = 0;

    fe_values[velocity].get_function_values(this->evaluation_point,
                                            present_velocity_values);
    fe_values[velocity].get_function_gradients(this->evaluation_point,
                                               present_velocity_gradients);
    fe_values[velocity].get_function_divergences(this->evaluation_point,
                                                 present_velocity_divergences);

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
      {
        cell_rhs(i) +=
            (
              // incompressibility equation
              present_velocity_divergences[q] * phi_pressure[i]
              // momentum equation
              - nu * scalar_product(present_velocity_gradients[q], grad_phi_velocity[i])
              - (present_velocity_gradients[q] * present_velocity_values[q]) * phi_velocity[i]
              + present_pressure_values[q] * div_phi_velocity[i]
            ) * JxW;
      }
    } // end loop over quadrature points

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

