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
OptionalArgumentsWeakForm<dim>::OptionalArgumentsWeakForm(const bool use_stress_form)
:
OptionalArguments<dim>(use_stress_form),
velocity_trial_function_symmetric_gradient(),
velocity_test_function_symmetric_gradient(),
present_symmetric_velocity_gradient(),
pressure_test_function_gradient(),
velocity_test_function_gradient(),
velocity_trial_function_grad_divergence(),
background_velocity_value(),
background_velocity_gradient(),
body_force_value()
{}



template<int dim>
OptionalArgumentsWeakForm<dim>::OptionalArgumentsWeakForm
(const OptionalArgumentsWeakForm<dim> &other)
:
OptionalArguments<dim>(other),
velocity_trial_function_symmetric_gradient(other.velocity_trial_function_symmetric_gradient),
velocity_test_function_symmetric_gradient(other.velocity_test_function_symmetric_gradient),
present_symmetric_velocity_gradient(other.present_symmetric_velocity_gradient),
pressure_test_function_gradient(other.pressure_test_function_gradient),
velocity_test_function_gradient(other.velocity_test_function_gradient),
velocity_trial_function_grad_divergence(other.velocity_trial_function_grad_divergence),
background_velocity_value(other.background_velocity_value),
background_velocity_gradient(other.background_velocity_gradient),
body_force_value(other.body_force_value)
{}



template<int dim>
OptionalArgumentsStrongForm<dim>::OptionalArgumentsStrongForm
(const StabilizationFlags stabilization,
 const bool use_stress_form,
 const bool allocate_background_velocity,
 const bool allocate_body_force,
 const unsigned int n_q_points)
:
OptionalArguments<dim>(use_stress_form),
present_velocity_grad_divergences(),
background_velocity_values(),
background_velocity_gradients(),
body_force_values()
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
}



template<int dim>
OptionalArgumentsStrongForm<dim>::OptionalArgumentsStrongForm
(const OptionalArgumentsStrongForm<dim> &other)
:
OptionalArguments<dim>(other),
present_velocity_grad_divergences(other.present_velocity_grad_divergences),
background_velocity_values(other.background_velocity_values),
background_velocity_gradients(other.background_velocity_gradients),
body_force_values(other.body_force_values)
{}

// explicit instantiations
template struct OptionalArguments<2>;
template struct OptionalArguments<3>;

template struct OptionalArgumentsWeakForm<2>;
template struct OptionalArgumentsWeakForm<3>;

template struct OptionalArgumentsStrongForm<2>;
template struct OptionalArgumentsStrongForm<3>;

}  // namespace Hydrodynamic

