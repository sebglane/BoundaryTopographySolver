/*
 * hydrodynamic_options.h
 *
 *  Created on: Oct 5, 2021
 *      Author: sg
 */

#ifndef INCLUDE_HYDRODYNAMIC_OPTIONS_H_
#define INCLUDE_HYDRODYNAMIC_OPTIONS_H_

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

#include <angular_velocity.h>
#include <stabilization_flags.h>

#include <optional>

namespace Hydrodynamic
{

using namespace dealii;

template<int dim>
struct OptionalArguments
{
  OptionalArguments(const bool use_stress_form);

  OptionalArguments(const OptionalArguments<dim> &other);

  // stress-based form
  const bool use_stress_form;

  // body force term
  std::optional<double> froude_number;

  // Coriolis term
  std::optional<typename Utility::AngularVelocity<dim>::value_type> angular_velocity;
  std::optional<double> rossby_number;
};



template<int dim>
struct OptionalScalarArguments : OptionalArguments<dim>
{
  OptionalScalarArguments(const bool use_stress_from);

  OptionalScalarArguments(const OptionalScalarArguments<dim> &other);

  // stress-based form
  std::optional<SymmetricTensor<2, dim>>  velocity_trial_function_symmetric_gradient;
  std::optional<SymmetricTensor<2, dim>>  velocity_test_function_symmetric_gradient;
  std::optional<SymmetricTensor<2, dim>>  present_symmetric_velocity_gradient;

  // stabilization and stress-based related trial functions
  std::optional<Tensor<1, dim>>           velocity_trial_function_grad_divergence;

  // body force term
  std::optional<Tensor<1, dim>>           body_force_value;
};



template<int dim>
struct OptionalVectorArguments : OptionalArguments<dim>
{
  OptionalVectorArguments(const StabilizationFlags  &stabilization,
                          const bool use_stress_form,
                          const bool allocate_background_velocity,
                          const bool allocate_body_force,
                          const bool allocate_traction,
                          const unsigned int n_q_points,
                          const unsigned int n_face_q_points);

  OptionalVectorArguments(const OptionalVectorArguments<dim> &other);

  // stabilization related solution values
  std::optional<std::vector<Tensor<1, dim>>>  present_velocity_laplaceans;
  std::optional<std::vector<Tensor<1, dim>>>  present_pressure_gradients;

  // stress-based form
  std::optional<std::vector<SymmetricTensor<2, dim>>> present_sym_velocity_gradients;
  std::optional<std::vector<Tensor<1, dim>>>          present_velocity_grad_divergences;

  // background velocity term
  std::optional<std::vector<Tensor<1, dim>>>  background_velocity_values;
  std::optional<std::vector<Tensor<2, dim>>>  background_velocity_gradients;

  // body force term
  std::optional<std::vector<Tensor<1, dim>>>  body_force_values;

  // source term face values
  std::vector<Tensor<1, dim>>                 boundary_traction_values;
};

}  // namespace Hydrodynamic


#endif /* INCLUDE_HYDRODYNAMIC_OPTIONS_H_ */
