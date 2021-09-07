/*
 * entropy_viscosity.cc
 *
 *  Created on: Sep 7, 2021
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <buoyant_hydrodynamic_solver.h>

namespace BuoyantHydrodynamic {

template<int dim>
void Solver<dim>::preprocess_newton_iteration()
{
  global_entropy_variation = compute_entropy_variation();
  return;
}



template<int dim>
double Solver<dim>::compute_entropy_variation() const
{
  const QGauss<dim> quadrature_formula(this->velocity_fe_degree + 1);

  FEValues<dim> fe_values(this->mapping,
                          *this->fe_system,
                          quadrature_formula,
                          update_values|
                          update_JxW_values);

  std::vector<double> density_values(quadrature_formula.size());

  double  min_entropy = std::numeric_limits<double>::max(),
          max_entropy = std::numeric_limits<double>::min(),
          volume = 0,
          integrated_entropy = 0;

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    fe_values.get_function_values(this->present_solution,
                                  density_values);

    for (const unsigned int q: fe_values.quadrature_point_indices())
    {
      const double entropy = 0.5 * density_values[q] * density_values[q];
      min_entropy = std::min(min_entropy, entropy);
      max_entropy = std::max(max_entropy, entropy);
      volume += fe_values.JxW(q);
      integrated_entropy += fe_values.JxW(q) * entropy;
    }
  }

  const double average_entropy = integrated_entropy / volume;

  const double entropy_variation = std::max(max_entropy - average_entropy,
                                            average_entropy - min_entropy);

  return (entropy_variation);
}

// explicit instantiations
template void Solver<2>::preprocess_newton_iteration();
template void Solver<3>::preprocess_newton_iteration();

template double Solver<2>::compute_entropy_variation() const;
template double Solver<3>::compute_entropy_variation() const;

}  // namespace BuoyantHydrodynamic

