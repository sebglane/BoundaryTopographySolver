/*
 * hydrodynamic_options.cc
 *
 *  Created on: Oct 5, 2021
 *      Author: sg
 */

#include <hydrodynamic_options.h>

namespace Hydrodynamic
{

template<int dim>
OptionalArguments<dim>::OptionalArguments(const bool use_stress_form)
:
use_stress_form(use_stress_form),
froude_number(),
angular_velocity(),
rossby_number()
{}



template<int dim>
OptionalArguments<dim>::OptionalArguments(const OptionalArguments<dim> &other)
:
use_stress_form(other.use_stress_form),
froude_number(other.froude_number),
angular_velocity(other.angular_velocity),
rossby_number(other.rossby_number)
{}



template<int dim>
OptionalScalarArguments<dim>::OptionalScalarArguments(const bool use_stress_form)
:
OptionalArguments<dim>(use_stress_form),
velocity_trial_function_symmetric_gradient(),
velocity_test_function_symmetric_gradient(),
present_symmetric_velocity_gradient(),
velocity_trial_function_grad_divergence(),
background_velocity_value(),
background_velocity_gradient(),
body_force_value()
{}



template<int dim>
OptionalScalarArguments<dim>::OptionalScalarArguments
(const OptionalScalarArguments<dim> &other)
:
OptionalArguments<dim>(other),
velocity_trial_function_symmetric_gradient(other.velocity_trial_function_symmetric_gradient),
velocity_test_function_symmetric_gradient(other.velocity_test_function_symmetric_gradient),
present_symmetric_velocity_gradient(other.present_symmetric_velocity_gradient),
velocity_trial_function_grad_divergence(other.velocity_trial_function_grad_divergence),
background_velocity_value(other.background_velocity_value),
background_velocity_gradient(other.background_velocity_gradient),
body_force_value(other.body_force_value)
{}



template<int dim>
OptionalVectorArguments<dim>::OptionalVectorArguments
(const StabilizationFlags &stabilization,
 const bool use_stress_form,
 const bool allocate_background_velocity,
 const bool allocate_body_force,
 const bool allocate_traction,
 const unsigned int n_q_points,
 const unsigned int n_face_q_points)
:
OptionalArguments<dim>(use_stress_form),
present_velocity_grad_divergences(),
background_velocity_values(),
background_velocity_gradients(),
body_force_values(),
boundary_traction_values()
{
  if (use_stress_form && (stabilization & (apply_supg|apply_pspg)))
    present_velocity_grad_divergences.emplace(n_q_points);

  if (allocate_background_velocity)
  {
    background_velocity_values.emplace(n_q_points);
    background_velocity_gradients.emplace(n_q_points);
  }

  if (allocate_body_force)
    body_force_values.emplace(n_q_points);

  if (allocate_traction)
    boundary_traction_values.resize(n_face_q_points);

}



template<int dim>
OptionalVectorArguments<dim>::OptionalVectorArguments
(const OptionalVectorArguments<dim> &other)
:
OptionalArguments<dim>(other),
present_velocity_grad_divergences(other.present_velocity_grad_divergences),
background_velocity_values(other.background_velocity_values),
background_velocity_gradients(other.background_velocity_gradients),
body_force_values(other.body_force_values),
boundary_traction_values(other.boundary_traction_values)
{}

// explicit instantiations
template struct OptionalArguments<2>;
template struct OptionalArguments<3>;

template struct OptionalScalarArguments<2>;
template struct OptionalScalarArguments<3>;

template struct OptionalVectorArguments<2>;
template struct OptionalVectorArguments<3>;

}  // namespace Hydrodynamic


