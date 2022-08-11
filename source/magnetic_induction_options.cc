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
 const bool         allocate_background_magnetic_field)
:
background_magnetic_field_values(),
background_magnetic_field_gradients()
{
  if (allocate_background_magnetic_field)
  {
    background_magnetic_field_values.emplace(n_q_points);
    background_magnetic_field_values.emplace(n_q_points);
  }
}



template<int dim>
VectorOptions<dim>::VectorOptions
(const VectorOptions<dim> &other)
:
background_magnetic_field_values(other.background_magnetic_field_values),
background_magnetic_field_gradients(other.background_magnetic_field_gradients)
{}



// explicit instantiations
template struct VectorOptions<2>;
template struct VectorOptions<3>;

}  // namespace MagneticInduction
