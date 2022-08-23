/*
 * magnetic_induction_options.cc
 *
 *  Created on: Aug 11, 2022
 *      Author: sg
 */

#include <magnetic_induction_options.h>

namespace MagneticInduction
{

template<int dim>
VectorOptions<dim>::VectorOptions
(const unsigned int n_q_points,
 const bool         allocate_velocity_field,
 const bool         allocate_background_magnetic_field)
:
background_magnetic_field_values(),
background_magnetic_field_curls(),
background_magnetic_field_divergences(),
velocity_field_values()
{
  if (allocate_background_magnetic_field)
  {
    background_magnetic_field_values.emplace(n_q_points);
    background_magnetic_field_curls.emplace(n_q_points);
    background_magnetic_field_divergences.emplace(n_q_points);
  }
  if (allocate_velocity_field)
    velocity_field_values.emplace(n_q_points);
}



template<int dim>
VectorOptions<dim>::VectorOptions
(const VectorOptions<dim> &other)
:
background_magnetic_field_values(other.background_magnetic_field_values),
background_magnetic_field_curls(other.background_magnetic_field_curls),
background_magnetic_field_divergences(other.background_magnetic_field_divergences),
velocity_field_values(other.velocity_field_values)
{}



// explicit instantiations
template struct VectorOptions<2>;
template struct VectorOptions<3>;

}  // namespace MagneticInduction
