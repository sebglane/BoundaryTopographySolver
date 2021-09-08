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
 const double          nu)
{
  return (// density equation
          (stratification_parameter *
           velocity_trial_function_value * reference_density_gradient +
           velocity_trial_function_value * present_density_gradient +
           present_velocity_value * density_trial_function_gradient
          ) * density_test_function_value +
          // stabilization term
          nu * density_trial_function_gradient * density_test_function_gradient
          );
}



template <int dim>
inline double compute_density_rhs
(const Tensor<1, dim> &density_test_function_gradient,
 const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<1, dim> &reference_density_gradient,
 const double          density_test_function_value,
 const double          stratification_parameter,
 const double          nu)
{
  return -(// density equation
           (stratification_parameter *
            present_velocity_value * reference_density_gradient +
            present_velocity_value * present_density_gradient
           ) * density_test_function_value +
           // stabilization term
           nu * present_density_gradient * density_test_function_gradient);
}



template<int dim>
inline double compute_entropy_viscosity
(const std::vector<Tensor<1, dim>> &present_velocity_values,
 const std::vector<Tensor<1, dim>> &present_density_gradients,
 const std::vector<double>         &present_density_values,
 const double                       cell_diameter,
 const double                       entropy_variation,
 const double                       c_max,
 const double                       c_entropy)
{
  AssertDimension(present_velocity_values.size(),
                  present_density_gradients.size());
  AssertDimension(present_velocity_values.size(),
                  present_density_values.size());

  AssertIsFinite(entropy_variation);

  Assert(entropy_variation > 0, ExcLowerRangeType<double>(0, entropy_variation));
  Assert(c_max > 0, ExcLowerRangeType<double>(0, c_max));
  Assert(c_entropy > 0, ExcLowerRangeType<double>(0, c_entropy));

  double max_residual = 0;
  double max_velocity = 0;

  for (std::size_t q=0; q<present_velocity_values.size(); ++q)
  {
    max_velocity = std::max(present_velocity_values[q].norm(), max_velocity);

    double residual = std::abs(present_density_gradients[q] * present_velocity_values[q]);
    residual *= std::abs(present_density_values[q]);
    max_residual = std::max(residual, max_residual);
  }

  const double max_viscosity = c_max * cell_diameter * max_velocity;

  const double entropy_viscosity = (c_entropy * cell_diameter * cell_diameter *
                                    max_residual / entropy_variation);

  if (entropy_viscosity == 0.0)
  {
    if (max_viscosity > 0.0)
      return max_viscosity;
    else
      return 0.1;
  }
  else
    return (std::min(max_viscosity, entropy_viscosity));
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

}  // namespace Advection

#endif /* INCLUDE_ASSEMBLY_FUNCTIONS_H_ */
