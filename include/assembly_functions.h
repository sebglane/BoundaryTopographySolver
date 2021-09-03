/*
 * assembly_functions.h
 *
 *  Created on: Sep 2, 2021
 *      Author: sg
 */

#ifndef INCLUDE_ASSEMBLY_FUNCTIONS_H_
#define INCLUDE_ASSEMBLY_FUNCTIONS_H_

#include <deal.II/base/tensor.h>

namespace TopographyProblem {

using namespace dealii;

template <int dim>
inline double compute_hydrodynamic_matrix
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
inline double compute_hydrodynamic_rhs
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

}  // namespace TopographyProblem



#endif /* INCLUDE_ASSEMBLY_FUNCTIONS_H_ */