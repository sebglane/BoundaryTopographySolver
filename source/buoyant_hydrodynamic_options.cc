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
VectorOptions<dim>::VectorOptions
(const unsigned int n_q_points,
 const bool allocate_gravity_field)
:
gravity_field_values()
{
  if (allocate_gravity_field)
    gravity_field_values.emplace(n_q_points);
}



template<int dim>
VectorOptions<dim>::VectorOptions(const VectorOptions<dim> &other)
:
gravity_field_values(other.gravity_field_values)
{}

// explicit instantiations
template struct VectorOptions<2>;
template struct VectorOptions<3>;

}  // namespace BuoyantHydrodynamic



