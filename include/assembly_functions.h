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
#include <hydrodynamic_options.h>

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
 const OptionalArgumentsWeakForm<dim> &options,
 const bool            apply_newton_linearization = true)
{
  const double velocity_trial_function_divergence{trace(velocity_trial_function_gradient)};
  const double velocity_test_function_divergence{trace(velocity_test_function_gradient)};

  double matrix{-(velocity_trial_function_divergence * pressure_test_function +
                  pressure_trial_function * velocity_test_function_divergence)};

  if (apply_newton_linearization)
    matrix += (present_velocity_gradient * velocity_trial_function_value +
               velocity_trial_function_gradient * present_velocity_value) *
               velocity_test_function_value;
  else
    matrix += velocity_trial_function_gradient * present_velocity_value *
              velocity_test_function_value;

  if (options.use_stress_form)
  {
    Assert(options.velocity_trial_function_symmetric_gradient, ExcInternalError());
    Assert(options.velocity_test_function_symmetric_gradient, ExcInternalError());

    matrix += 2.0 * nu * scalar_product(*options.velocity_trial_function_symmetric_gradient,
                                        *options.velocity_test_function_symmetric_gradient);
  }
  else
    matrix += nu * scalar_product(velocity_trial_function_gradient,
                                  velocity_test_function_gradient);

  if (options.background_velocity_value)
  {
    Assert(options.background_velocity_gradient, ExcInternalError());

    if (apply_newton_linearization)
      matrix += (velocity_trial_function_gradient * *options.background_velocity_value +
                 *options.background_velocity_gradient * velocity_trial_function_value) *
                velocity_test_function_value;
    else
      matrix += velocity_trial_function_gradient * *options.background_velocity_value *
                velocity_test_function_value;
  }

  if (options.angular_velocity)
  {
    Assert(options.rossby_number, ExcInternalError());

    if constexpr(dim == 2)
      matrix += 2.0 / *options.rossby_number * options.angular_velocity.value()[0] *
                cross_product_2d(-velocity_trial_function_value) *
                velocity_test_function_value;
    else if constexpr(dim == 3)
      matrix += 2.0 / *options.rossby_number *
                cross_product_3d(*options.angular_velocity, velocity_trial_function_value) *
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
 const OptionalArgumentsWeakForm<dim> &options)
{
  const double present_velocity_divergence{trace(present_velocity_gradient)};
  const double velocity_test_function_divergence{trace(velocity_test_function_gradient)};

  double rhs{present_velocity_divergence * pressure_test_function +
             present_pressure_value * velocity_test_function_divergence -
             (present_velocity_gradient * present_velocity_value) *
             velocity_test_function_value};

  if (options.use_stress_form)
  {
    Assert(options.present_symmetric_velocity_gradient, ExcInternalError());
    Assert(options.velocity_test_function_symmetric_gradient, ExcInternalError());

    rhs -= 2.0 * nu * scalar_product(*options.present_symmetric_velocity_gradient,
                                     *options.velocity_test_function_symmetric_gradient);
  }
  else
    rhs -= nu * scalar_product(present_velocity_gradient,
                               velocity_test_function_gradient);

  if (options.background_velocity_value)
  {
    Assert(options.background_velocity_gradient, ExcInternalError());

    rhs -= (present_velocity_gradient * *options.background_velocity_value +
            *options.background_velocity_gradient * present_velocity_value) *
            velocity_test_function_value;
  }

  if (options.body_force_value)
  {
    Assert(options.froude_number, ExcInternalError());

    rhs -= *options.body_force_value * velocity_test_function_value /
           (*options.froude_number * *options.froude_number);
  }

  if (options.angular_velocity)
  {
    Assert(options.rossby_number, ExcInternalError());

    if constexpr(dim == 2)
      rhs -= 2.0 / *options.rossby_number * options.angular_velocity.value()[0] *
             cross_product_2d(-present_velocity_value) * velocity_test_function_value;
    else if constexpr(dim == 3)
      rhs -= 2.0 / *options.rossby_number *
             cross_product_3d(*options.angular_velocity, present_velocity_value) *
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
 const OptionalArgumentsStrongForm<dim> &options)
{
  if (options.use_stress_form)
  {
    Assert(options.present_velocity_grad_divergences, ExcInternalError());

    for (std::size_t q=0; q<present_velocity_values.size(); ++q)
      strong_residuals[q] = (present_velocity_gradients[q] * present_velocity_values[q]) -
                            nu * present_velocity_laplaceans[q] -
                            nu * options.present_velocity_grad_divergences->at(q) +
                            present_pressure_gradients[q];
  }
  else
    for (std::size_t q=0; q<present_velocity_values.size(); ++q)
      strong_residuals[q] = (present_velocity_gradients[q] * present_velocity_values[q]) -
                            nu * present_velocity_laplaceans[q] +
                            present_pressure_gradients[q];


  if (options.background_velocity_values)
  {
    Assert(options.background_velocity_gradients, ExcInternalError());

    for (std::size_t q=0; q<present_velocity_values.size(); ++q)
      strong_residuals[q] += present_velocity_gradients[q] * options.background_velocity_values->at(q) +
                             options.background_velocity_gradients->at(q) * present_velocity_values[q];
  }

  if (options.body_force_values)
  {
    Assert(options.froude_number, ExcInternalError());

    for (std::size_t q=0; q<present_velocity_values.size(); ++q)
      strong_residuals[q] -= options.body_force_values->at(q) / std::pow(*options.froude_number, 2);
  }

  if (options.angular_velocity)
  {
    Assert(options.rossby_number, ExcInternalError());

    if constexpr(dim == 2)
        for (std::size_t q=0; q<present_velocity_values.size(); ++q)
          strong_residuals[q] += 2.0 / *options.rossby_number * options.angular_velocity.value()[0] *
                                 cross_product_2d(-present_velocity_values[q]);
    else if constexpr(dim == 3)
        for (std::size_t q=0; q<present_velocity_values.size(); ++q)
          strong_residuals[q] += 2.0 / *options.rossby_number *
                                 cross_product_3d(*options.angular_velocity, present_velocity_values[q]);
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
 const OptionalArgumentsWeakForm<dim> &options,
 const bool            apply_newton_linearization = true)
{
  if (!options.velocity_test_function_gradient && !options.pressure_test_function_gradient)
    return (0.0);

  // linearized residual
  Tensor<1, dim> linearized_residual{velocity_trial_function_gradient * present_velocity_value +
                                     pressure_trial_function_gradient};

  if (apply_newton_linearization)
    linearized_residual += present_velocity_gradient * velocity_trial_function_value;

  if (options.use_stress_form)
  {
    Assert(options.velocity_trial_function_grad_divergence, ExcInternalError());

    linearized_residual -= nu * (velocity_trial_function_laplacean +
                                 *options.velocity_trial_function_grad_divergence);
  }
  else
    linearized_residual -= nu * velocity_trial_function_laplacean;

  if (options.background_velocity_value)
  {
    Assert(options.background_velocity_gradient, ExcInternalError());

    linearized_residual += velocity_trial_function_gradient * *options.background_velocity_value;

    if (apply_newton_linearization)
      linearized_residual += *options.background_velocity_gradient * velocity_trial_function_value;
  }

  if (options.angular_velocity)
  {
    Assert(options.rossby_number, ExcInternalError());

    if constexpr(dim == 2)
      linearized_residual += 2.0 / *options.rossby_number * options.angular_velocity.value()[0] *
                            cross_product_2d(-present_velocity_value);
    else if constexpr(dim == 3)
      linearized_residual += 2.0 / *options.rossby_number *
                             cross_product_3d(*options.angular_velocity, present_velocity_value);
  }

  Tensor<1, dim> test_function;

  if (options.velocity_test_function_gradient)
  {
    test_function += *options.velocity_test_function_gradient * present_velocity_value;

    if (options.background_velocity_value)
      test_function += *options.velocity_test_function_gradient * *options.background_velocity_value;
  }
  if (options.pressure_test_function_gradient)
    test_function += *options.pressure_test_function_gradient;

  return (linearized_residual * test_function);

}



template <int dim>
inline double compute_residual_linearization_matrix_stress_form
(const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<2, dim> &velocity_trial_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_laplacean,
 const Tensor<1, dim> &velocity_trial_function_grad_divergence,
 const Tensor<1, dim> &pressure_trial_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const double          nu,
 const bool            apply_newton_linearization,
 const std::optional<Tensor<2, dim>>  &velocity_test_function_gradient,
 const std::optional<Tensor<1, dim>>  &pressure_test_function_gradient,
 const std::optional<Tensor<1, dim>>  &background_velocity_value = std::nullopt,
 const std::optional<Tensor<2, dim>>  &background_velocity_gradient = std::nullopt,
 const std::optional<typename Utility::AngularVelocity<dim>::value_type> angular_velocity = std::nullopt,
 const std::optional<double>           rossby_number = std::nullopt)
{
  if (!velocity_test_function_gradient &&
      !pressure_test_function_gradient)
    return (0.0);

  // linearized residual
  Tensor<1, dim> linearized_residual;
  if (apply_newton_linearization)
    linearized_residual = velocity_trial_function_gradient * present_velocity_value +
                          present_velocity_gradient * velocity_trial_function_value -
                          nu * velocity_trial_function_laplacean -
                          nu * velocity_trial_function_grad_divergence +
                          pressure_trial_function_gradient;
  else
    linearized_residual = velocity_trial_function_gradient * present_velocity_value -
                          nu * velocity_trial_function_laplacean -
                          nu * velocity_trial_function_grad_divergence +
                          pressure_trial_function_gradient;

  if (background_velocity_value)
  {
    Assert(background_velocity_gradient, ExcInternalError());

    if (apply_newton_linearization)
      linearized_residual += velocity_trial_function_gradient * *background_velocity_value +
                             *background_velocity_gradient * velocity_trial_function_value;
    else
      linearized_residual += velocity_trial_function_gradient * *background_velocity_value;
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
inline void compute_strong_hydrodynamic_residual
(const std::vector<Tensor<1, dim>> &present_velocity_values,
 const std::vector<Tensor<2, dim>> &present_velocity_gradients,
 const std::vector<Tensor<1, dim>> &present_velocity_laplaceans,
 const std::vector<Tensor<1, dim>> &present_pressure_gradients,
 const std::vector<Tensor<1, dim>> &gravity_field_values,
 const std::vector<double>         &present_density_values,
 std::vector<Tensor<1, dim>>       &strong_residuals,
 const double                       nu,
 const double                       froude_number,
 const std::optional<std::vector<Tensor<1, dim>>> &background_velocity_values = std::nullopt,
 const std::optional<std::vector<Tensor<2, dim>>> &background_velocity_gradients = std::nullopt,
 const std::optional<std::vector<Tensor<1, dim>>> &body_force_values = std::nullopt,
 const std::optional<typename Utility::AngularVelocity<dim>::value_type> angular_velocity = std::nullopt,
 const std::optional<double>                       rossby_number = std::nullopt)
{
  Hydrodynamic::
  compute_strong_residual(present_velocity_values,
                          present_velocity_gradients,
                          present_velocity_laplaceans,
                          present_pressure_gradients,
                          strong_residuals,
                          nu,
                          background_velocity_values,
                          background_velocity_gradients,
                          body_force_values,
                          froude_number,
                          angular_velocity,
                          rossby_number);

  for (std::size_t q=0; q<present_velocity_values.size(); ++q)
    strong_residuals[q] -= present_density_values[q] * gravity_field_values[q] / std::pow(froude_number, 2);

}



template <int dim>
inline void compute_strong_density_residual
(const std::vector<Tensor<1, dim>> &present_density_gradients,
 const std::vector<Tensor<1, dim>> &present_velocity_values,
 std::vector<double>               &strong_residuals,
 const std::optional<std::vector<Tensor<1, dim>>> &reference_density_gradients = std::nullopt,
 const std::optional<double>                       stratification_number = std::nullopt,
 const std::optional<std::vector<Tensor<1, dim>>> &background_velocity_values = std::nullopt)
{
  for (std::size_t q=0; q<present_density_gradients.size(); ++q)
    strong_residuals[q] = present_velocity_values[q] * present_density_gradients[q];

  if (reference_density_gradients)
  {
    Assert(stratification_number, ExcInternalError());

    for (std::size_t q=0; q<present_density_gradients.size(); ++q)
      strong_residuals[q] += *stratification_number * present_velocity_values[q] *
                             reference_density_gradients->at(q);
  }

  if (background_velocity_values)
  {
    for (std::size_t q=0; q<present_density_gradients.size(); ++q)
      strong_residuals[q] += background_velocity_values->at(q) * present_density_gradients[q];

    if (reference_density_gradients)
      for (std::size_t q=0; q<present_density_gradients.size(); ++q)
        strong_residuals[q] += *stratification_number * background_velocity_values->at(q) *
                               reference_density_gradients->at(q);
  }
}



template <int dim>
inline double compute_density_matrix
(const Tensor<1, dim> &density_trial_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const double          density_test_function_value,
 const bool            apply_newton_linearization = true,
 const std::optional<Tensor<1, dim>> &reference_density_gradient = std::nullopt,
 const std::optional<double>          stratification_number = std::nullopt,
 const std::optional<Tensor<1, dim>> &background_velocity_value = std::nullopt)
{
  double linearized_residual =
      present_velocity_value * density_trial_function_gradient +
      (apply_newton_linearization?
          velocity_trial_function_value * present_density_gradient:
          0.0);

  if (reference_density_gradient && apply_newton_linearization)
  {
    Assert(stratification_number, ExcInternalError());

    linearized_residual += *stratification_number * velocity_trial_function_value *
                           *reference_density_gradient;
  }

  if (background_velocity_value)
    linearized_residual += *background_velocity_value * density_trial_function_gradient;

  return (linearized_residual * density_test_function_value);
}



template <int dim>
inline double compute_density_rhs
(const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const double          density_test_function_value,
 const std::optional<Tensor<1, dim>> &reference_density_gradient = std::nullopt,
 const std::optional<double>          stratification_number = std::nullopt,
 const std::optional<Tensor<1, dim>> &background_velocity_value = std::nullopt)
{
  double residual = -(present_velocity_value * present_density_gradient);

  if (reference_density_gradient)
  {
    Assert(stratification_number, ExcInternalError());

    residual -= *stratification_number * present_velocity_value *
                *reference_density_gradient;
  }

  if (background_velocity_value)
  {
    residual -= *background_velocity_value * present_density_gradient;

    if (reference_density_gradient)
      residual -= *stratification_number * *background_velocity_value * *reference_density_gradient;
  }

  return (residual * density_test_function_value);
}



template <int dim>
inline double compute_density_residual_linearization_matrix
(const Tensor<1, dim> &density_trial_function_gradient,
 const Tensor<1, dim> &density_test_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const double          nu,
 const bool            apply_newton_linearization = true,
 const std::optional<Tensor<1, dim>> &reference_density_gradient = std::nullopt,
 const std::optional<double>          stratification_number = std::nullopt,
 const std::optional<Tensor<1, dim>> &background_velocity_value = std::nullopt)
{
  if (present_velocity_value.norm() > 0.0)
  {
    double linearized_residual =
        present_velocity_value * density_trial_function_gradient +
        (apply_newton_linearization?
            velocity_trial_function_value * present_density_gradient:
            0.0);


    if (reference_density_gradient && apply_newton_linearization)
    {
      Assert(stratification_number, ExcInternalError());


      linearized_residual += *stratification_number * velocity_trial_function_value *
                             *reference_density_gradient;
    }

    if (background_velocity_value)
    {
      linearized_residual += *background_velocity_value * density_trial_function_gradient;

      const double test_function{density_test_function_gradient *
                                 (present_velocity_value + *background_velocity_value)};

      return (linearized_residual *  test_function);
    }
    else
      return (linearized_residual * (density_test_function_gradient * present_velocity_value));
  }
  else if (background_velocity_value && (background_velocity_value->norm() > 0.0))
  {
    double linearized_residual =
        *background_velocity_value * density_trial_function_gradient +
        (apply_newton_linearization?
            velocity_trial_function_value * present_density_gradient:
            0.0);

    if (reference_density_gradient && apply_newton_linearization)
    {
      Assert(stratification_number, ExcInternalError());

      linearized_residual += *stratification_number * velocity_trial_function_value *
                             *reference_density_gradient;
    }

    const double test_function{density_test_function_gradient * *background_velocity_value};

    return (linearized_residual *  test_function);

  }
  else
    return (nu * density_trial_function_gradient * density_test_function_gradient);
}



template <int dim>
inline double compute_hydrodynamic_matrix
(const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<2, dim> &velocity_trial_function_gradient,
 const Tensor<1, dim> &velocity_test_function_value,
 const Tensor<2, dim> &velocity_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const Tensor<1, dim> &present_gravity_field_value,
 const double          pressure_trial_function,
 const double          density_trial_function_value,
 const double          pressure_test_function,
 const double          nu,
 const double          froude_number,
 const bool            apply_newton_linearization = true,
 const std::optional<Tensor<1, dim>> &background_velocity_value = std::nullopt,
 const std::optional<Tensor<2, dim>> &background_velocity_gradient = std::nullopt,
 const std::optional<typename Utility::AngularVelocity<dim>::value_type> angular_velocity = std::nullopt,
 const std::optional<double>          rossby_number = std::nullopt)
{
  double matrix = Hydrodynamic::
                  compute_matrix(velocity_trial_function_value,
                                 velocity_trial_function_gradient,
                                 velocity_test_function_value,
                                 velocity_test_function_gradient,
                                 present_velocity_value,
                                 present_velocity_gradient,
                                 pressure_trial_function,
                                 pressure_test_function,
                                 nu,
                                 apply_newton_linearization,
                                 background_velocity_value,
                                 background_velocity_gradient,
                                 angular_velocity,
                                 rossby_number);

  matrix -= density_trial_function_value *  present_gravity_field_value / std::pow(froude_number, 2) *
            velocity_test_function_value;

  return (matrix);
}



template <int dim>
inline double compute_hydrodynamic_rhs
(const Tensor<1, dim> &velocity_test_function_value,
 const Tensor<2, dim> &velocity_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const Tensor<1, dim> &present_gravity_field_value,
 const double          present_pressure_value,
 const double          present_density_value,
 const double          pressure_test_function,
 const double          nu,
 const double          froude_number,
 const std::optional<Tensor<1, dim>> &background_velocity_value = std::nullopt,
 const std::optional<Tensor<2, dim>> &background_velocity_gradient = std::nullopt,
 const std::optional<Tensor<1, dim>> &body_force_value = std::nullopt,
 const std::optional<typename Utility::AngularVelocity<dim>::value_type> angular_velocity = std::nullopt,
 const std::optional<double>          rossby_number = std::nullopt)
{
  double rhs = Hydrodynamic::
               compute_rhs(velocity_test_function_value,
                           velocity_test_function_gradient,
                           present_velocity_value,
                           present_velocity_gradient,
                           present_pressure_value,
                           pressure_test_function,
                           nu,
                           background_velocity_value,
                           background_velocity_gradient,
                           body_force_value,
                           froude_number,
                           angular_velocity,
                           rossby_number);

  rhs += present_density_value *  present_gravity_field_value / std::pow(froude_number, 2) *
         velocity_test_function_value;

  return (rhs);
}



template <int dim>
inline double compute_hydrodynamic_residual_linearization_matrix
(const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<2, dim> &velocity_trial_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_laplacean,
 const Tensor<1, dim> &pressure_trial_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const Tensor<1, dim> &present_gravity_field_value,
 const double          density_trial_function_value,
 const double          nu,
 const double          froude_number,
 const bool            apply_newton_linearization,
 const std::optional<Tensor<2, dim>>  &velocity_test_function_gradient,
 const std::optional<Tensor<1, dim>>  &pressure_test_function_gradient,
 const std::optional<Tensor<1, dim>>  &background_velocity_value = std::nullopt,
 const std::optional<Tensor<2, dim>>  &background_velocity_gradient = std::nullopt,
 const std::optional<typename Utility::AngularVelocity<dim>::value_type> angular_velocity = std::nullopt,
 const std::optional<double>           rossby_number = std::nullopt)
{
  if (!velocity_test_function_gradient &&
      !pressure_test_function_gradient)
    return (0.0);

  double matrix = Hydrodynamic::
                  compute_residual_linearization_matrix(velocity_trial_function_value,
                                                        velocity_trial_function_gradient,
                                                        velocity_trial_function_laplacean,
                                                        pressure_trial_function_gradient,
                                                        present_velocity_value,
                                                        present_velocity_gradient,
                                                        nu,
                                                        apply_newton_linearization,
                                                        velocity_test_function_gradient,
                                                        pressure_test_function_gradient,
                                                        background_velocity_value,
                                                        background_velocity_gradient,
                                                        angular_velocity,
                                                        rossby_number);
  Tensor<1, dim> test_function;

  if (velocity_test_function_gradient)
  {
    test_function += *velocity_test_function_gradient * present_velocity_value;

    if (background_velocity_value)
      test_function += *velocity_test_function_gradient * *background_velocity_value;
  }
  if (pressure_test_function_gradient)
    test_function += *pressure_test_function_gradient;

  matrix -= density_trial_function_value *  present_gravity_field_value / std::pow(froude_number, 2) *
            test_function;

  return (matrix);

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
