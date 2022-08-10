/*
 * magnetic_induction_options.h
 *
 *  Created on: Oct 5, 2021
 *      Author: sg
 */

#ifndef INCLUDE_MAGNETIC_INDUCTION_OPTIONS_H_
#define INCLUDE_MAGNETIC_INDUCTION_OPTIONS_H_

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

#include <angular_velocity.h>
#include <stabilization_flags.h>

#include <optional>

namespace MagneticInduction
{

using namespace dealii;

template<int dim>
struct VectorOptions
{
  VectorOptions(const unsigned int n_q_points,
                const bool background_magnetic_field = false);

  VectorOptions(const VectorOptions<dim> &other);

  // background magnetic field
  std::optional<std::vector<Tensor<1, dim>>>  background_magnetic_field_values;
  std::optional<std::vector<Tensor<2, dim>>>  background_magnetic_field_gradients;

};

}  // namespace MagneticInduction


#endif /* INCLUDE_MAGNETIC_INDUCTION_OPTIONS_H_ */
