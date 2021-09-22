/*
 * assembly_functions.h
 *
 *  Created on: Sep 2, 2021
 *      Author: sg
 */

#ifndef INCLUDE_ASSEMBLY_FUNCTIONS_H_
#define INCLUDE_ASSEMBLY_FUNCTIONS_H_

#include <deal.II/base/tensor.h>

namespace Hydrodynamic {

using namespace dealii;

template <int dim>
inline double compute_matrix
(const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<2, dim> &velocity_trial_function_gradient,
 const Tensor<1, dim> &velocity_test_function_value,
 const Tensor<2, dim> &velocity_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const double          pressure_trial_function,
 const double          pressure_test_function,
 const double          nu)
{
  const double velocity_trial_function_divergence{trace(velocity_trial_function_gradient)};
  const double velocity_test_function_divergence{trace(velocity_test_function_gradient)};

  return (// incompressibility equation
          - velocity_trial_function_divergence * pressure_test_function
          // momentum equation
          + nu * scalar_product(velocity_trial_function_gradient,
                                velocity_test_function_gradient)
          + (velocity_trial_function_gradient * present_velocity_value) * velocity_test_function_value
          + (present_velocity_gradient * velocity_trial_function_value) * velocity_test_function_value
          - pressure_trial_function * velocity_test_function_divergence
          );
}



template <int dim>
inline double compute_rhs
(const Tensor<1, dim> &velocity_test_function_value,
 const Tensor<2, dim> &velocity_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const double          present_pressure_value,
 const double          pressure_test_function,
 const double          nu)
{
  const double present_velocity_divergence{trace(present_velocity_gradient)};
  const double velocity_test_function_divergence{trace(velocity_test_function_gradient)};

  return (// incompressibility equation
          present_velocity_divergence * pressure_test_function
          // momentum equation
          - nu * scalar_product(present_velocity_gradient,
                                velocity_test_function_gradient)
          - (present_velocity_gradient * present_velocity_value) * velocity_test_function_value
          + present_pressure_value * velocity_test_function_divergence
          );
}

}  // namespace Hydrodynamic

namespace BuoyantHydrodynamic {

using namespace dealii;

template <int dim>
inline double compute_density_matrix
(const Tensor<1, dim> &density_trial_function_gradient,
 const Tensor<1, dim> &density_test_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<1, dim> &reference_density_gradient,
 const double          density_test_function_value,
 const double          stratification_parameter,
 const double          delta,
 const bool            vanishing_velocity)
{
  if (vanishing_velocity)
    return (stratification_parameter * velocity_trial_function_value * reference_density_gradient +
            velocity_trial_function_value * present_density_gradient +
            present_velocity_value * density_trial_function_gradient) *
           (density_test_function_value +
            delta * present_velocity_value * density_test_function_gradient) +
            delta * density_trial_function_gradient * density_test_function_gradient +
           (stratification_parameter * present_velocity_value * reference_density_gradient +
            present_velocity_value * present_density_gradient) *
           (delta * velocity_trial_function_value * density_trial_function_gradient);
  else
    return (stratification_parameter * velocity_trial_function_value * reference_density_gradient +
            velocity_trial_function_value * present_density_gradient +
            present_velocity_value * density_trial_function_gradient) *
           (density_test_function_value +
            delta * present_velocity_value * density_test_function_gradient) +
           (present_velocity_value * present_density_gradient) *
           (delta * velocity_trial_function_value * density_trial_function_gradient);
}



template <int dim>
inline double compute_density_rhs
(const Tensor<1, dim> &density_test_function_gradient,
 const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<1, dim> &reference_density_gradient,
 const double          density_test_function_value,
 const double          stratification_parameter,
 const double          delta)
{
  return -(stratification_parameter * present_velocity_value * reference_density_gradient +
           present_velocity_value * present_density_gradient) *
          (density_test_function_value +
           delta * present_velocity_value * density_test_function_gradient);
}



template<int dim>
inline std::pair<const double, bool> compute_stabilization_parameter
(const std::vector<Tensor<1, dim>> &present_velocity_values,
 const double                       cell_diameter)
{
  double max_velocity = 0;

  for (std::size_t q=0; q<present_velocity_values.size(); ++q)
    max_velocity = std::max(present_velocity_values[q].norm(), max_velocity);

  if (max_velocity > 0.0)
    return std::pair<const double, bool>(0.5 * cell_diameter / max_velocity, false);
  else
    return std::pair<const double, bool>(0.1 * cell_diameter, true);
}

}  // namespace BuoyantHydrodynamic

namespace Advection {

using namespace dealii;

template <int dim>
inline double compute_matrix
(const Tensor<1, dim> &trial_function_gradient,
 const Tensor<1, dim> &test_function_gradient,
 const Tensor<1, dim> &advection_field_value,
 const double          test_function_value,
 const double          delta)
{
  return (advection_field_value * trial_function_gradient) *
         (test_function_value +
          delta * advection_field_value * test_function_gradient);
}



template <int dim>
inline double compute_rhs
(const Tensor<1, dim> &test_function_gradient,
 const Tensor<1, dim> &present_gradient,
 const Tensor<1, dim> &advection_field_value,
 const double          test_function_value,
 const double          delta)
{
  return -(advection_field_value * present_gradient) *
          (test_function_value +
           delta * advection_field_value * test_function_gradient);
}



template<int dim>
inline double compute_stabilization_parameter
(const std::vector<Tensor<1, dim>> &advection_field_values,
 const double                       cell_diameter)
{
  double max_velocity = 0;

  for (std::size_t q=0; q<advection_field_values.size(); ++q)
    max_velocity = std::max(advection_field_values[q].norm(), max_velocity);

  const double max_viscosity = 0.5 * cell_diameter / max_velocity;

  return (max_viscosity);
}

}  // namespace Advection

#endif /* INCLUDE_ASSEMBLY_FUNCTIONS_H_ */
