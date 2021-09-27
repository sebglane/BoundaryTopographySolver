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



template <int dim>
inline double compute_supg_matrix
(const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<2, dim> &velocity_trial_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_laplacean,
 const Tensor<2, dim> &velocity_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const Tensor<1, dim> &present_velocity_laplacean,
 const Tensor<1, dim> &pressure_trial_function_gradient,
 const Tensor<1, dim> &present_pressure_gradient,
 const double          nu)
{
  const Tensor<1, dim> projected_test_function_gradient(velocity_test_function_gradient * present_velocity_value);
  const Tensor<1, dim> linearized_test_function_gradient(velocity_test_function_gradient * velocity_trial_function_value);

  return (// linearized residual
            (velocity_trial_function_gradient * present_velocity_value) * projected_test_function_gradient
          + (present_velocity_gradient * velocity_trial_function_value) * projected_test_function_gradient
          - nu * velocity_trial_function_laplacean * projected_test_function_gradient
          + pressure_trial_function_gradient * projected_test_function_gradient
          // linearized test function
          + (present_velocity_gradient * present_velocity_value) * linearized_test_function_gradient
          - nu * present_velocity_laplacean * linearized_test_function_gradient
          + present_pressure_gradient * linearized_test_function_gradient
         );
}



template <int dim>
inline double compute_supg_rhs
(const Tensor<2, dim> &velocity_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const Tensor<1, dim> &present_velocity_laplacean,
 const Tensor<1, dim> &present_pressure_gradient,
 const double          nu)
{
  const Tensor<1, dim> projected_test_function_gradient(velocity_test_function_gradient * present_velocity_value);

  return (- (present_velocity_gradient * present_velocity_value) * projected_test_function_gradient
          + nu * present_velocity_laplacean * projected_test_function_gradient
          - present_pressure_gradient * projected_test_function_gradient
         );
}



template <int dim>
inline double compute_pspg_matrix
(const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<2, dim> &velocity_trial_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_laplacean,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const Tensor<1, dim> &pressure_test_function_gradient,
 const Tensor<1, dim> &pressure_trial_function_gradient,
 const double          nu)
{
  return (  (velocity_trial_function_gradient * present_velocity_value) * pressure_test_function_gradient
          + (present_velocity_gradient * velocity_trial_function_value) * pressure_test_function_gradient
          - nu * velocity_trial_function_laplacean * pressure_test_function_gradient
          + pressure_trial_function_gradient * pressure_test_function_gradient
         );
}



template <int dim>
inline double compute_pspg_rhs
(const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const Tensor<1, dim> &present_velocity_laplacean,
 const Tensor<1, dim> &pressure_test_function_gradient,
 const Tensor<1, dim> &present_pressure_gradient,
 const double          nu)
{
  return (- (present_velocity_gradient * present_velocity_value) * pressure_test_function_gradient
          + nu * present_velocity_laplacean * pressure_test_function_gradient
          - present_pressure_gradient * pressure_test_function_gradient
         );
}



template <int dim>
inline double compute_grad_div_matrix
(const Tensor<2, dim> &velocity_trial_function_gradient,
 const Tensor<2, dim> &velocity_test_function_gradient)
{
  return (trace(velocity_trial_function_gradient) *
          trace(velocity_test_function_gradient)
         );
}



template <int dim>
inline double compute_grad_div_rhs
(const Tensor<2, dim> &present_velocity_gradient,
 const Tensor<2, dim> &velocity_test_function_gradient)
{
  return (- trace(present_velocity_gradient) *
            trace(velocity_test_function_gradient)
         );
}


}  // namespace Hydrodynamic

namespace BuoyantHydrodynamic {

using namespace dealii;

template <int dim>
inline double compute_density_matrix
(const Tensor<1, dim> &density_trial_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<1, dim> &reference_density_gradient,
 const double          density_test_function_value,
 const double          stratification_parameter)
{
  return (stratification_parameter * velocity_trial_function_value * reference_density_gradient +
          velocity_trial_function_value * present_density_gradient +
          present_velocity_value * density_trial_function_gradient) * density_test_function_value;
}



template <int dim>
inline double compute_density_rhs
(const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<1, dim> &reference_density_gradient,
 const double          density_test_function_value,
 const double          stratification_parameter)
{
  return -(stratification_parameter * present_velocity_value * reference_density_gradient +
           present_velocity_value * present_density_gradient) * density_test_function_value;
}



template <int dim>
inline double compute_density_supg_matrix
(const Tensor<1, dim> &density_trial_function_gradient,
 const Tensor<1, dim> &density_test_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<1, dim> &reference_density_gradient,
 const double          stratification_parameter,
 const double          nu)
{
  const double projected_test_function{density_test_function_gradient * present_velocity_value};
  const double linearized_test_function{density_test_function_gradient * velocity_trial_function_value};

  if (present_velocity_value.norm() > 0.0)
    return (// linearized residual
            (  stratification_parameter * velocity_trial_function_value * reference_density_gradient
             + velocity_trial_function_value * present_density_gradient
             + present_velocity_value * density_trial_function_gradient
            ) * projected_test_function +
            // linearized test function
            (  stratification_parameter * present_velocity_value * reference_density_gradient
             + present_velocity_value * present_density_gradient
            ) * linearized_test_function
           );
  else
    return (nu * density_trial_function_gradient * density_test_function_gradient);
}



template <int dim>
inline double compute_density_supg_rhs
(const Tensor<1, dim> &density_test_function_gradient,
 const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<1, dim> &reference_density_gradient,
 const double          stratification_parameter,
 const double          nu)
{
  const double projected_test_function{density_test_function_gradient * present_velocity_value};

  if (present_velocity_value.norm() > 0.0)
    return (-(  stratification_parameter * present_velocity_value * reference_density_gradient
              + present_velocity_value * present_density_gradient
             ) * projected_test_function);
  else
    return (-nu * present_density_gradient * density_test_function_gradient);
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
