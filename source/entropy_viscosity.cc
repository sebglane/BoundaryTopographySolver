/*
 * entropy_viscosity.cc
 *
 *  Created on: Dec 13, 2018
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>

#include <equation_data.h>
#include <solver.h>

namespace TopographyProblem {

template<int dim>
double TopographySolver<dim>::compute_density_viscosity(
        const std::vector<Tensor<1,dim>>   &velocity_values,
        const double                        cell_diameter) const
{
    const unsigned int n_q_points = velocity_values.size();

    double max_velocity = 0;

    for (unsigned int q=0; q < n_q_points; ++q)
    {
        const Tensor<1,dim>     velocity = velocity_values[q];
        max_velocity = std::max(std::sqrt(velocity*velocity), max_velocity);
    }

    const double max_viscosity = parameters.c_density * cell_diameter * max_velocity;

    if (max_viscosity > 0.0)
        return max_viscosity;
    else
        return parameters.default_viscosity;
}

template<int dim>
double TopographySolver<dim>::compute_velocity_viscosity(
        const std::vector<Tensor<1,dim>>   &velocity_values,
        const double                        cell_diameter) const
{
    const unsigned int n_q_points = velocity_values.size();

    double max_velocity = 0;

    for (unsigned int q=0; q < n_q_points; ++q)
    {
        const Tensor<1,dim>     velocity = velocity_values[q];
        max_velocity = std::max(std::sqrt(velocity*velocity), max_velocity);
    }

    const double max_viscosity = parameters.c_velocity * cell_diameter * max_velocity;

    if (max_viscosity > 0.0)
        return max_viscosity;
    else
        return parameters.default_viscosity;
}

}  // namespace TopographyProblem


// explicit instantiation
template double TopographyProblem::TopographySolver<3>::compute_density_viscosity(
        const std::vector<dealii::Tensor<1,3>>  &,
        const double                            ) const;

template double TopographyProblem::TopographySolver<3>::compute_velocity_viscosity(
        const std::vector<dealii::Tensor<1,3>>  &,
        const double                            ) const;
