/*
 * assembly_functions.h
 *
 *  Created on: Sep 2, 2021
 *      Author: sg
 */

#ifndef INCLUDE_ASSEMBLY_FUNCTIONS_H_
#define INCLUDE_ASSEMBLY_FUNCTIONS_H_

#include <deal.II/base/tensor.h>

#include <advection_options.h>
#include <angular_velocity.h>
#include <buoyant_hydrodynamic_options.h>
#include <hydrodynamic_options.h>

#include <optional>

namespace Hydrodynamic {

using namespace dealii;

template <int dim>
double compute_matrix
(const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<2, dim> &velocity_trial_function_gradient,
 const Tensor<1, dim> &velocity_test_function_value,
 const Tensor<2, dim> &velocity_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const double          pressure_trial_function,
 const double          pressure_test_function,
 const double          nu,
 const OptionalScalarArguments<dim> &options,
 const bool            apply_newton_linearization = true);



template <int dim>
double compute_rhs
(const StabilizationFlags  &stabilization,
 const Tensor<1, dim>      &velocity_test_function_value,
 const Tensor<2, dim>      &velocity_test_function_gradient,
 const Tensor<1, dim>      &present_velocity_value,
 const Tensor<2, dim>      &present_velocity_gradient,
 const Tensor<1, dim>      &present_strong_residual,
 const double               present_pressure_value,
 const double               pressure_test_function,
 const Tensor<1, dim>      &pressure_test_function_gradient,
 const double               nu,
 const double               mu,
 const double               delta,
 const OptionalScalarArguments<dim> &options);



template <int dim>
void compute_strong_residual
(const std::vector<Tensor<1, dim>>   &present_velocity_values,
 const std::vector<Tensor<2, dim>>   &present_velocity_gradients,
 const OptionalVectorArguments<dim>  &options,
 const double                         nu,
 std::vector<Tensor<1,dim>>          &strong_residuals);


template <int dim>
double compute_residual_linearization_matrix
(const StabilizationFlags  &stabilization,
 const Tensor<1, dim>      &velocity_trial_function_value,
 const Tensor<2, dim>      &velocity_trial_function_gradient,
 const Tensor<1, dim>      &velocity_trial_function_laplacean,
 const Tensor<1, dim>      &pressure_trial_function_gradient,
 const Tensor<1, dim>      &present_velocity_value,
 const Tensor<2, dim>      &present_velocity_gradient,
 const Tensor<1, dim>      &present_strong_residual,
 const Tensor<2, dim>      &velocity_test_function_gradient,
 const Tensor<1, dim>      &pressure_test_function_gradient,
 const double               nu,
 const double               delta,
 const double               mu,
 const OptionalScalarArguments<dim> &options,
 const bool                 apply_newton_linearization = true);

}  // namespace Hydrodynamic




namespace BuoyantHydrodynamic {

using namespace dealii;

template <int dim>
inline void compute_strong_hydrodynamic_residual
(const std::vector<Tensor<1, dim>> &present_velocity_values,
 const std::vector<Tensor<2, dim>> &present_velocity_gradients,
 const std::vector<Tensor<1, dim>> &present_velocity_laplaceans,
 const std::vector<Tensor<1, dim>> &present_pressure_gradients,
 const std::vector<double>         &present_density_values,
 std::vector<Tensor<1, dim>>       &strong_residuals,
 const double                       nu,
 const Hydrodynamic::OptionalVectorArguments<dim>        &options,
 const BuoyantHydrodynamic::OptionalVectorArguments<dim> &buoyancy_options)
{
  Hydrodynamic::
  compute_strong_residual(present_velocity_values,
                          present_velocity_gradients,
                          present_velocity_laplaceans,
                          present_pressure_gradients,
                          strong_residuals,
                          nu,
                          options);

  if (buoyancy_options.gravity_field_values)
  {
    Assert(options.froude_number, ExcInternalError());

    for (std::size_t q=0; q<present_velocity_values.size(); ++q)
      strong_residuals[q] -= present_density_values[q] *
                             buoyancy_options.gravity_field_values->at(q) /
                             (*options.froude_number * *options.froude_number);
  }
}



template <int dim>
inline void compute_strong_density_residual
(const std::vector<Tensor<1, dim>> &present_density_gradients,
 const std::vector<Tensor<1, dim>> &present_velocity_values,
 std::vector<double>               &strong_residuals,
 const Hydrodynamic::OptionalVectorArguments<dim>        &options,
 const BuoyantHydrodynamic::OptionalVectorArguments<dim> &buoyancy_options)
{
  for (std::size_t q=0; q<present_density_gradients.size(); ++q)
    strong_residuals[q] = present_velocity_values[q] * present_density_gradients[q];

  if (buoyancy_options.reference_density_gradients)
  {
    Assert(buoyancy_options.stratification_number, ExcInternalError());

    for (std::size_t q=0; q<present_density_gradients.size(); ++q)
      strong_residuals[q] += *buoyancy_options.stratification_number * present_velocity_values[q] *
                             buoyancy_options.reference_density_gradients->at(q);
  }

  if (options.background_velocity_values)
  {
    for (std::size_t q=0; q<present_density_gradients.size(); ++q)
      strong_residuals[q] += options.background_velocity_values->at(q) *
                             present_density_gradients[q];

    if (buoyancy_options.reference_density_gradients)
      for (std::size_t q=0; q<present_density_gradients.size(); ++q)
        strong_residuals[q] += *buoyancy_options.stratification_number *
                               options.background_velocity_values->at(q) *
                               buoyancy_options.reference_density_gradients->at(q);
  }
}



template <int dim>
inline double compute_density_matrix
(const Tensor<1, dim> &density_trial_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const double          density_test_function_value,
 const Hydrodynamic::OptionalScalarArguments<dim>        &options,
 const Advection::OptionalScalarArguments<dim> &buoyancy_options,
 const bool            apply_newton_linearization = true)
{
  double linearized_residual =
      present_velocity_value * density_trial_function_gradient +
      (apply_newton_linearization?
          velocity_trial_function_value * present_density_gradient:
          0.0);

  if (buoyancy_options.reference_gradient && apply_newton_linearization)
  {
    Assert(buoyancy_options.gradient_scaling, ExcInternalError());

    linearized_residual += *buoyancy_options.gradient_scaling *
                           velocity_trial_function_value *
                           *buoyancy_options.reference_gradient;
  }

  if (options.background_velocity_value)
    linearized_residual += *options.background_velocity_value *
                           density_trial_function_gradient;

  return (linearized_residual * density_test_function_value);
}



template <int dim>
inline double compute_density_rhs
(const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const double          density_test_function_value,
 const Hydrodynamic::OptionalScalarArguments<dim>        &options,
 const Advection::OptionalScalarArguments<dim> &buoyancy_options)
{
  double residual = -(present_velocity_value * present_density_gradient);

  if (buoyancy_options.reference_gradient)
  {
    Assert(buoyancy_options.gradient_scaling, ExcInternalError());

    residual -= *buoyancy_options.gradient_scaling *
                present_velocity_value *
                *buoyancy_options.reference_gradient;
  }

  if (options.background_velocity_value)
  {
    residual -= *options.background_velocity_value * present_density_gradient;

    if (buoyancy_options.reference_gradient)
      residual -= *buoyancy_options.gradient_scaling *
                  *options.background_velocity_value *
                  *buoyancy_options.reference_gradient;
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
 const Hydrodynamic::OptionalScalarArguments<dim>  &options,
 const Advection::OptionalScalarArguments<dim>     &buoyancy_options,
 const bool            apply_newton_linearization = true)
{
  if (present_velocity_value.norm() > 0.0)
  {
    double linearized_residual =
        present_velocity_value * density_trial_function_gradient +
        (apply_newton_linearization?
            velocity_trial_function_value * present_density_gradient:
            0.0);


    if (buoyancy_options.reference_gradient && apply_newton_linearization)
    {
      Assert(buoyancy_options.gradient_scaling, ExcInternalError());


      linearized_residual += *buoyancy_options.gradient_scaling *
                             velocity_trial_function_value *
                             *buoyancy_options.reference_gradient;
    }

    if (options.background_velocity_value)
    {
      linearized_residual += *options.background_velocity_value *
                             density_trial_function_gradient;

      const double test_function{density_test_function_gradient *
                                 (present_velocity_value +
                                  *options.background_velocity_value)};

      return (linearized_residual *  test_function);
    }
    else
      return (linearized_residual * (density_test_function_gradient * present_velocity_value));
  }
  else if (options.background_velocity_value && (options.background_velocity_value->norm() > 0.0))
  {
    double linearized_residual =
        *options.background_velocity_value * density_trial_function_gradient +
        (apply_newton_linearization?
            velocity_trial_function_value * present_density_gradient:
            0.0);

    if (buoyancy_options.reference_gradient && apply_newton_linearization)
    {
      Assert(buoyancy_options.gradient_scaling, ExcInternalError());

      linearized_residual += *buoyancy_options.gradient_scaling *
                             velocity_trial_function_value *
                             *buoyancy_options.reference_gradient;
    }

    const double test_function{density_test_function_gradient *
                               *options.background_velocity_value};

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
 const double          pressure_trial_function,
 const double          density_trial_function_value,
 const double          pressure_test_function,
 const double          nu,
 const Hydrodynamic::OptionalScalarArguments<dim>        &options,
 const BuoyantHydrodynamic::OptionalScalarArguments<dim> &buoyancy_options,
 const bool            apply_newton_linearization = true)
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
                                 options,
                                 apply_newton_linearization);

  if (buoyancy_options.gravity_field_value)
  {
    Assert(options.froude_number, ExcInternalError());

    matrix -= density_trial_function_value * *buoyancy_options.gravity_field_value *
              velocity_test_function_value /
              (*options.froude_number * *options.froude_number);
  }

  return (matrix);
}



template <int dim>
inline double compute_hydrodynamic_rhs
(const Tensor<1, dim> &velocity_test_function_value,
 const Tensor<2, dim> &velocity_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const double          present_pressure_value,
 const double          present_density_value,
 const double          pressure_test_function,
 const double          nu,
 const Hydrodynamic::OptionalScalarArguments<dim>        &options,
 const BuoyantHydrodynamic::OptionalScalarArguments<dim> &buoyancy_options)
{
  double rhs = Hydrodynamic::
               compute_rhs(velocity_test_function_value,
                           velocity_test_function_gradient,
                           present_velocity_value,
                           present_velocity_gradient,
                           present_pressure_value,
                           pressure_test_function,
                           nu,
                           options);

  if (buoyancy_options.gravity_field_value)
  {
    Assert(options.froude_number, ExcInternalError());

    rhs += present_density_value * *buoyancy_options.gravity_field_value *
           velocity_test_function_value /
           (*options.froude_number * *options.froude_number);
  }

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
 const double          density_trial_function_value,
 const double          nu,
 const Hydrodynamic::OptionalScalarArguments<dim>        &options,
 const BuoyantHydrodynamic::OptionalScalarArguments<dim> &buoyancy_options,
 const bool            apply_newton_linearization = true)
{
  if (!options.velocity_test_function_gradient &&
      !options.pressure_test_function_gradient)
    return (0.0);

  double matrix = Hydrodynamic::
                  compute_residual_linearization_matrix(velocity_trial_function_value,
                                                        velocity_trial_function_gradient,
                                                        velocity_trial_function_laplacean,
                                                        pressure_trial_function_gradient,
                                                        present_velocity_value,
                                                        present_velocity_gradient,
                                                        nu,
                                                        options,
                                                        apply_newton_linearization);
  if (buoyancy_options.gravity_field_value)
  {
    Assert(options.froude_number, ExcInternalError());

    Tensor<1, dim> test_function;
    if (options.velocity_test_function_gradient)
    {
      test_function += *options.velocity_test_function_gradient *
                       present_velocity_value;

      if (options.background_velocity_value)
        test_function += *options.velocity_test_function_gradient *
                         *options.background_velocity_value;
    }
    if (options.pressure_test_function_gradient)
      test_function += *options.pressure_test_function_gradient;

    matrix -= density_trial_function_value * *buoyancy_options.gravity_field_value *
              test_function /
              (*options.froude_number * *options.froude_number);
  }

  return (matrix);

}

}  // namespace BuoyantHydrodynamic



namespace Advection {

using namespace dealii;

/*!
 * @brief Computes the matrix entry of the advection equation.
 *
 * @attention The advection field must include contributions due to a possible
 * background field.
 *
 */
template <int dim>
double compute_matrix
(const Tensor<1, dim>                &trial_function_gradient,
 const Tensor<1, dim>                &advection_field_value,
 const double                         test_function_value);


/*!
 * @brief Computes the right-hand side entry of the advection equation.
 *
 * @attention The advection field must include contributions due to a possible
 * background field.
 *
 */
template <int dim>
double compute_rhs
(const double           test_function_value,
 const Tensor<1, dim>  &test_function_gradient,
 const Tensor<1, dim>  &present_gradient,
 const Tensor<1, dim>  &advection_field_value,
 const double           present_strong_residual,
 const double           delta,
 const OptionalScalarArguments<dim> &options);




/*!
 * @brief Computes the strong residual of the advection equation.
 *
 * @attention The advection field must include contributions due to a possible
 * background field.
 *
 */
template<int dim>
void compute_strong_residual
(const std::vector<Tensor<1, dim>>   &present_gradients,
 const std::vector<Tensor<1, dim>>   &advection_field_values,
 std::vector<double>                 &strong_residuals,
 const OptionalVectorArguments<dim>  &options);


/*!
 * @brief Computes the linearization of the strong residual of the advection
 * equation.
 *
 */
template <int dim>
double compute_residual_linearization_matrix
(const Tensor<1, dim>                &trial_function_gradient,
 const Tensor<1, dim>                &advection_field_value,
 const Tensor<1, dim>                &test_function_gradient,
 const double                         delta);

}  // namespace Advection

#endif /* INCLUDE_ASSEMBLY_FUNCTIONS_H_ */
