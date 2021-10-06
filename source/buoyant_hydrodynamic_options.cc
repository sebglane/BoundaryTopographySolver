/*
 * buoyant_hydrodynamic_options.cc
 *
 *  Created on: Oct 6, 2021
 *      Author: sg
 */

#include <buoyant_hydrodynamic_options.h>

namespace BuoyantHydrodynamic
{

OptionalArguments::OptionalArguments()
:
stratification_number()
{}



OptionalArguments::OptionalArguments(const OptionalArguments &other)
:
stratification_number(other.stratification_number)
{}



template<int dim>
OptionalArgumentsWeakForm<dim>::OptionalArgumentsWeakForm()
:
OptionalArguments(),
gravity_field_value(),
reference_density_gradient()
{}



template<int dim>
OptionalArgumentsWeakForm<dim>::OptionalArgumentsWeakForm
(const OptionalArgumentsWeakForm<dim> &other)
:
OptionalArguments(other),
gravity_field_value(other.gravity_field_value),
reference_density_gradient(other.reference_density_gradient)
{}



template<int dim>
OptionalArgumentsStrongForm<dim>::OptionalArgumentsStrongForm
(const bool allocate_gravity_field,
 const bool allocate_reference_density,
 const unsigned int n_q_points)
:
OptionalArguments(),
gravity_field_values(),
reference_density_gradients()
{
  if (allocate_gravity_field)
    gravity_field_values.emplace(n_q_points);

  if (allocate_reference_density)
    reference_density_gradients.emplace(n_q_points);
}




template<int dim>
OptionalArgumentsStrongForm<dim>::OptionalArgumentsStrongForm(const OptionalArgumentsStrongForm<dim> &other)
:
OptionalArguments(other),
gravity_field_values(other.gravity_field_values),
reference_density_gradients(other.reference_density_gradients)
{}

// explicit instantiations
template struct OptionalArgumentsWeakForm<2>;
template struct OptionalArgumentsWeakForm<3>;

template struct OptionalArgumentsStrongForm<2>;
template struct OptionalArgumentsStrongForm<3>;

}  // namespace BuoyantHydrodynamic



