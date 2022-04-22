/*
 * advection_options.h
 *
 *  Created on: Apr 12, 2022
 *      Author: sg
 */

#ifndef INCLUDE_ADVECTION_OPTIONS_H_
#define INCLUDE_ADVECTION_OPTIONS_H_

#include <deal.II/base/tensor.h>

#include <optional>

namespace Advection {

using namespace dealii;

struct OptionalArguments
{
  OptionalArguments();

  OptionalArguments(const OptionalArguments &other);

  // scale factor of the gradient of the reference field
  std::optional<double> gradient_scaling;
};



template<int dim>
struct OptionalScalarArguments : OptionalArguments
{
  OptionalScalarArguments();

  OptionalScalarArguments(const OptionalScalarArguments<dim> &other);

  // source term
  std::optional<double>           source_term_value;

  // reference gradient
  std::optional<Tensor<1, dim>>   reference_gradient;
};



template<int dim>
struct OptionalVectorArguments : OptionalArguments
{
  OptionalVectorArguments(
      const unsigned int n_q_points,
      const unsigned int n_face_q_points,
      const bool         allocate_source_term = false,
      const bool         allocate_boundary_values = false,
      const bool         allocate_background_velocity = false,
      const bool         allocate_reference_gradient = false);

  OptionalVectorArguments(const OptionalVectorArguments<dim> &other);

  // source term
  std::optional<std::vector<double>>          source_term_values;

  // boundary term
  std::vector<double>         boundary_values;
  std::vector<Tensor<1,dim>>  advection_field_face_values;

  // background advection field
  std::optional<std::vector<Tensor<1, dim>>>  background_advection_values;

  // reference gradient
  std::optional<std::vector<Tensor<1, dim>>>  reference_gradients;
};

}  // namespace Advection


#endif /* INCLUDE_ADVECTION_OPTIONS_H_ */