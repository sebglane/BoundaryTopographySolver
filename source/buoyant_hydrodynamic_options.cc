/*
 * buoyant_hydrodynamic_options.cc
 *
 *  Created on: Oct 6, 2021
 *      Author: sg
 */

#include <buoyant_hydrodynamic_options.h>

namespace BuoyantHydrodynamic
{

template<int dim>
OptionalScalarArguments<dim>::OptionalScalarArguments()
:
gravity_field_value()
{}



template<int dim>
OptionalScalarArguments<dim>::OptionalScalarArguments
(const OptionalScalarArguments<dim> &other)
:
gravity_field_value(other.gravity_field_value)
{}



template<int dim>
OptionalVectorArguments<dim>::OptionalVectorArguments
(const unsigned int n_q_points,
 const bool allocate_gravity_field)
:
gravity_field_values()
{
  if (allocate_gravity_field)
    gravity_field_values.emplace(n_q_points);
}




template<int dim>
OptionalVectorArguments<dim>::OptionalVectorArguments(const OptionalVectorArguments<dim> &other)
:
gravity_field_values(other.gravity_field_values)
{}

// explicit instantiations
template struct OptionalScalarArguments<2>;
template struct OptionalScalarArguments<3>;

template struct OptionalVectorArguments<2>;
template struct OptionalVectorArguments<3>;

}  // namespace BuoyantHydrodynamic



