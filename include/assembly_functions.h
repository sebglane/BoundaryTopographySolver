/*
 * assembly_functions.h
 *
 *  Created on: Sep 2, 2021
 *      Author: sg
 */

#ifndef INCLUDE_ASSEMBLY_FUNCTIONS_H_
#define INCLUDE_ASSEMBLY_FUNCTIONS_H_

#include <deal.II/base/tensor.h>

#include <angular_velocity.h>

#include <optional>

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
 const double          nu,
 const std::optional<Tensor<1, dim>> &background_velocity_value = std::nullopt,
 const std::optional<Tensor<2, dim>> &background_velocity_gradient = std::nullopt,
 const std::optional<typename Utility::AngularVelocity<dim>::value_type> angular_velocity = std::nullopt,
 const std::optional<double>          rossby_number = std::nullopt)
{
  const double velocity_trial_function_divergence{trace(velocity_trial_function_gradient)};
  const double velocity_test_function_divergence{trace(velocity_test_function_gradient)};

  double matrix = // incompressibility equation
                  - velocity_trial_function_divergence * pressure_test_function +
                  // momentum equation
                  (velocity_trial_function_gradient * present_velocity_value +
                   present_velocity_gradient * velocity_trial_function_value ) *
                   velocity_test_function_value -
                  pressure_trial_function * velocity_test_function_divergence +
                  nu * scalar_product(velocity_trial_function_gradient,
                                      velocity_test_function_gradient);

  if (background_velocity_value)
  {
    Assert(background_velocity_gradient, ExcInternalError());

    matrix += (velocity_trial_function_gradient * *background_velocity_value +
               *background_velocity_gradient * velocity_trial_function_value) * velocity_test_function_value;
  }

  if (angular_velocity)
  {
    Assert(rossby_number, ExcInternalError());

    if constexpr(dim == 2)
      matrix += 2.0 / *rossby_number * angular_velocity.value()[0] *
                cross_product_2d(-velocity_trial_function_value) *
                velocity_test_function_value;
    else if constexpr(dim == 3)
      matrix += 2.0 / *rossby_number *
                cross_product_3d(*angular_velocity, velocity_trial_function_value) *
                velocity_test_function_value;
  }

  return (matrix);
}



template <int dim>
inline double compute_rhs
(const Tensor<1, dim> &velocity_test_function_value,
 const Tensor<2, dim> &velocity_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const double          present_pressure_value,
 const double          pressure_test_function,
 const double          nu,
 const std::optional<Tensor<1, dim>> &background_velocity_value = std::nullopt,
 const std::optional<Tensor<2, dim>> &background_velocity_gradient = std::nullopt,
 const std::optional<Tensor<1, dim>> &body_force_value = std::nullopt,
 const std::optional<double>          froude_number = std::nullopt,
 const std::optional<typename Utility::AngularVelocity<dim>::value_type> angular_velocity = std::nullopt,
 const std::optional<double>          rossby_number = std::nullopt)
{
  const double present_velocity_divergence{trace(present_velocity_gradient)};
  const double velocity_test_function_divergence{trace(velocity_test_function_gradient)};

  double rhs = // incompressibility equation
               present_velocity_divergence * pressure_test_function +
               // momentum equation
               present_pressure_value * velocity_test_function_divergence -
               (present_velocity_gradient * present_velocity_value) * velocity_test_function_value -
               nu * scalar_product(present_velocity_gradient,
                                   velocity_test_function_gradient);

  if (background_velocity_value)
  {
    Assert(background_velocity_gradient, ExcInternalError());

    rhs -= (present_velocity_gradient * *background_velocity_value +
            *background_velocity_gradient * present_velocity_value) *
            velocity_test_function_value;
  }

  if (body_force_value)
  {
    Assert(froude_number, ExcInternalError());

    rhs -= *body_force_value * velocity_test_function_value / std::pow(*froude_number, 2);
  }

  if (angular_velocity)
  {
    Assert(rossby_number, ExcInternalError());

    if constexpr(dim == 2)
      rhs -= 2.0 / *rossby_number * angular_velocity.value()[0] *
             cross_product_2d(-present_velocity_value) * velocity_test_function_value;
    else if constexpr(dim == 3)
      rhs -= 2.0 / *rossby_number *
             cross_product_3d(*angular_velocity, present_velocity_value) *
             velocity_test_function_value;
  }

  return (rhs);

}



template <int dim>
inline void compute_strong_residual
(const std::vector<Tensor<1, dim>> &present_velocity_values,
 const std::vector<Tensor<2, dim>> &present_velocity_gradients,
 const std::vector<Tensor<1, dim>> &present_velocity_laplaceans,
 const std::vector<Tensor<1, dim>> &present_pressure_gradients,
 std::vector<Tensor<1, dim>>       &strong_residuals,
 const double          nu,
 const std::optional<std::vector<Tensor<1, dim>>> &background_velocity_values = std::nullopt,
 const std::optional<std::vector<Tensor<2, dim>>> &background_velocity_gradients = std::nullopt,
 const std::optional<std::vector<Tensor<1, dim>>> &body_force_values = std::nullopt,
 const std::optional<double>                       froude_number = std::nullopt,
 const std::optional<typename Utility::AngularVelocity<dim>::value_type> angular_velocity = std::nullopt,
 const std::optional<double>                       rossby_number = std::nullopt)
{
  for (std::size_t q=0; q<present_velocity_values.size(); ++q)
    strong_residuals[q] = (present_velocity_gradients[q] * present_velocity_values[q])
                          - nu * present_velocity_laplaceans[q]
                          + present_pressure_gradients[q];

  if (background_velocity_values)
  {
    Assert(background_velocity_gradients, ExcInternalError());

    for (std::size_t q=0; q<present_velocity_values.size(); ++q)
      strong_residuals[q] += present_velocity_gradients[q] * background_velocity_values->at(q) +
                             background_velocity_gradients->at(q) * present_velocity_values[q];
  }

  if (body_force_values)
  {
    Assert(froude_number, ExcInternalError());

    for (std::size_t q=0; q<present_velocity_values.size(); ++q)
      strong_residuals[q] -= body_force_values->at(q) / std::pow(*froude_number, 2);
  }

  if (angular_velocity)
  {
    Assert(rossby_number, ExcInternalError());

    if constexpr(dim == 2)
        for (std::size_t q=0; q<present_velocity_values.size(); ++q)
          strong_residuals[q] += 2.0 / *rossby_number * angular_velocity.value()[0] *
                                 cross_product_2d(-present_velocity_values[q]);
    else if constexpr(dim == 3)
        for (std::size_t q=0; q<present_velocity_values.size(); ++q)
          strong_residuals[q] += 2.0 / *rossby_number *
                                 cross_product_3d(*angular_velocity, present_velocity_values[q]);
  }
}



template <int dim>
inline double compute_residual_linearization_matrix
(const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<2, dim> &velocity_trial_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_laplacean,
 const Tensor<1, dim> &pressure_trial_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const double          nu,
 const std::optional<Tensor<2, dim>>  &velocity_test_function_gradient,
 const std::optional<Tensor<1, dim>>  &pressure_test_function_gradient,
 const std::optional<Tensor<1, dim>>  &background_velocity_value = std::nullopt,
 const std::optional<Tensor<2, dim>>  &background_velocity_gradient = std::nullopt,
 const std::optional<typename Utility::AngularVelocity<dim>::value_type> angular_velocity = std::nullopt,
 const std::optional<double>           rossby_number = std::nullopt)
{
  if (!velocity_test_function_gradient ||
      !pressure_test_function_gradient)
    return (0.0);

  // linearized residual
  Tensor<1, dim> linearized_residual =
      velocity_trial_function_gradient * present_velocity_value +
      present_velocity_gradient * velocity_trial_function_value -
      nu * velocity_trial_function_laplacean +
      pressure_trial_function_gradient;

  if (background_velocity_value)
  {
    Assert(background_velocity_gradient, ExcInternalError());

    linearized_residual += present_velocity_gradient * *background_velocity_value +
                           *background_velocity_gradient * present_velocity_value;
  }

  if (angular_velocity)
  {
    Assert(rossby_number, ExcInternalError());

    if constexpr(dim == 2)
      linearized_residual += 2.0 / *rossby_number * angular_velocity.value()[0] *
                            cross_product_2d(-present_velocity_value);
    else if constexpr(dim == 3)
      linearized_residual += 2.0 / *rossby_number *
                             cross_product_3d(*angular_velocity, present_velocity_value);
  }

  Tensor<1, dim> test_function;

  if (velocity_test_function_gradient)
  {
    test_function += *velocity_test_function_gradient * present_velocity_value;

    if (background_velocity_value)
      test_function += *velocity_test_function_gradient * *background_velocity_value;
  }
  if (pressure_test_function_gradient)
    test_function += *pressure_test_function_gradient;

  return (linearized_residual * test_function);

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
