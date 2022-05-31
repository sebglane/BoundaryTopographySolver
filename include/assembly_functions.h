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
double compute_hydrodynamic_matrix
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
 const bool            apply_newton_linearization = true);



template <int dim>
double compute_hydrodynamic_residual_linearization_matrix
(const StabilizationFlags  &stabilization,
 const Tensor<1, dim>      &velocity_trial_function_value,
 const Tensor<2, dim>      &velocity_trial_function_gradient,
 const Tensor<1, dim>      &velocity_trial_function_laplacean,
 const Tensor<1, dim>      &pressure_trial_function_gradient,
 const Tensor<1, dim>      &present_velocity_value,
 const Tensor<2, dim>      &present_velocity_gradient,
 const Tensor<1, dim>      &present_strong_residual,
 const double               density_trial_function_value,
 const Tensor<2, dim>      &velocity_test_function_gradient,
 const Tensor<1, dim>      &pressure_test_function_gradient,
 const double               nu,
 const double               delta,
 const double               mu,
 const Hydrodynamic::OptionalScalarArguments<dim>        &options,
 const BuoyantHydrodynamic::OptionalScalarArguments<dim> &buoyancy_options,
 const bool                 apply_newton_linearization = true);



template <int dim>
double compute_hydrodynamic_rhs
(const StabilizationFlags & stabilization,
 const Tensor<1, dim> &velocity_test_function_value,
 const Tensor<2, dim> &velocity_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<2, dim> &present_velocity_gradient,
 const Tensor<1, dim> &present_strong_residual,
 const double          present_pressure_value,
 const double          present_density_value,
 const double          pressure_test_function,
 const Tensor<1, dim> &pressure_test_function_gradient,
 const double          nu,
 const double          mu,
 const double          delta,
 const Hydrodynamic::OptionalScalarArguments<dim>        &options,
 const BuoyantHydrodynamic::OptionalScalarArguments<dim> &buoyancy_options);



template <int dim>
void compute_strong_hydrodynamic_residual
(const std::vector<Tensor<1, dim>> &present_velocity_values,
 const std::vector<Tensor<2, dim>> &present_velocity_gradients,
 const std::vector<double>         &present_density_values,
 std::vector<Tensor<1, dim>>       &strong_residuals,
 const double                       nu,
 const Hydrodynamic::OptionalVectorArguments<dim>        &options,
 const BuoyantHydrodynamic::OptionalVectorArguments<dim> &buoyancy_options);



template <int dim>
void compute_strong_density_residual
(const std::vector<Tensor<1, dim>>             &present_density_gradients,
 const std::vector<Tensor<1, dim>>             &present_velocity_values,
 std::vector<double>                           &strong_residuals,
 const Advection::OptionalVectorArguments<dim> &advection_options);



template <int dim>
double compute_density_matrix
(const Tensor<1, dim> &density_trial_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const double          density_test_function_value,
 const Advection::OptionalScalarArguments<dim>      &advection_options,
 const bool            apply_newton_linearization = true);



template <int dim>
double compute_density_rhs
(const Tensor<1, dim> &present_density_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const double          present_strong_residual,
 const double          density_test_function_value,
 const Tensor<1, dim> &density_test_function_gradient,
 const double          delta,
 const Advection::OptionalScalarArguments<dim>  &advection_options);



template <int dim>
double compute_density_residual_linearization_matrix
(const Tensor<1, dim> &density_trial_function_gradient,
 const Tensor<1, dim> &velocity_trial_function_value,
 const Tensor<1, dim> &density_test_function_gradient,
 const Tensor<1, dim> &present_velocity_value,
 const Tensor<1, dim> &present_density_gradient,
 const double          present_strong_residual,
 const double          delta,
 const double          nu,
 const Advection::OptionalScalarArguments<dim>    &advection_options,
 const bool            apply_newton_linearization = true);



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
