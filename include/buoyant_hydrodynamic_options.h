/*
 * buoyant_hydrodynamic_options.h
 *
 *  Created on: Oct 6, 2021
 *      Author: sg
 */

#ifndef INCLUDE_BUOYANT_HYDRODYNAMIC_OPTIONS_H_
#define INCLUDE_BUOYANT_HYDRODYNAMIC_OPTIONS_H_

#include <deal.II/base/tensor.h>

#include <optional>

namespace BuoyantHydrodynamic
{

using namespace dealii;

/**
 * @todo Add documentation.
 */
template<int dim>
struct VectorOptions
{
  VectorOptions(const unsigned int n_q_points,
                          const bool allocate_gravity_field = false);

  VectorOptions(const VectorOptions<dim> &other);

  // gravity term
  std::optional<std::vector<Tensor<1, dim>>> gravity_field_values;

};

}  // namespace BuoyantHydrodynamic


#endif /* INCLUDE_BUOYANT_HYDRODYNAMIC_OPTIONS_H_ */
