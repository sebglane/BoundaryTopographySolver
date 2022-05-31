/*
 * advection_options.cc
 *
 *  Created on: Apr 12, 2022
 *      Author: sg
 */

#include <advection_options.h>

namespace Advection
{

OptionalArguments::OptionalArguments()
:
gradient_scaling()
{}



OptionalArguments::OptionalArguments
(const OptionalArguments  &other)
:
gradient_scaling(other.gradient_scaling)
{}



template<int dim>
OptionalScalarArguments<dim>::OptionalScalarArguments
()
:
OptionalArguments(),
source_term_value(),
reference_gradient()
{}



template<int dim>
OptionalScalarArguments<dim>::OptionalScalarArguments
(const OptionalScalarArguments<dim> &other)
:
OptionalArguments(other),
source_term_value(other.source_term_value),
reference_gradient(other.reference_gradient)
{}



template<int dim>
OptionalVectorArguments<dim>::OptionalVectorArguments
(const unsigned int n_q_points,
 const unsigned int n_face_q_points,
 const bool         allocate_source_term,
 const bool         allocate_boundary_values,
 const bool         allocate_background_velocity,
 const bool         allocate_reference_gradient)
:
OptionalArguments(),
source_term_values(),
boundary_values(),
advection_field_face_values(),
background_advection_values(),
reference_gradients()
{
  if (allocate_source_term)
    source_term_values.emplace(n_q_points);

  if (allocate_boundary_values)
  {
    boundary_values.resize(n_face_q_points);
    advection_field_face_values.resize(n_face_q_points);
  }

  if (allocate_background_velocity)
    background_advection_values.emplace(n_q_points);

  if (allocate_reference_gradient)
    reference_gradients.emplace(n_q_points);
}



template<int dim>
OptionalVectorArguments<dim>::OptionalVectorArguments
(const OptionalVectorArguments<dim> &other)
:
OptionalArguments(other),
source_term_values(other.source_term_values),
boundary_values(other.boundary_values),
advection_field_face_values(other.advection_field_face_values),
background_advection_values(other.background_advection_values),
reference_gradients(other.reference_gradients)
{}

// explicit instantiations
template struct OptionalScalarArguments<2>;
template struct OptionalScalarArguments<3>;

template struct OptionalVectorArguments<2>;
template struct OptionalVectorArguments<3>;

}  // Advection
