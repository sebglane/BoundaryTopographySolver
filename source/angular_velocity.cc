/*
 * angular_velocity.cc
 *
 *  Created on: Aug 18, 2021
 *      Author: sg
 */

#include <angular_velocity.h>

namespace TopographyProblem
{

template <int dim>
AngularVelocity<dim>::AngularVelocity(const double time)
:
FunctionTime<double>(time)
{}


template <int dim>
typename AngularVelocity<dim>::value_type
AngularVelocity<dim>::value() const
{
  value_type  value;
  value = 0;
  return (value);
}

// explicit instantiations
template class AngularVelocity<2>;
template class AngularVelocity<3>;

}  // namespace TopographyProblem
