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

struct OptionalArguments
{
  OptionalArguments();

  OptionalArguments(const OptionalArguments &other);

  std::optional<double> stratification_number;
};



template<int dim>
struct OptionalArgumentsWeakForm : OptionalArguments
{
  OptionalArgumentsWeakForm();

  OptionalArgumentsWeakForm(const OptionalArgumentsWeakForm<dim> &other);

  // gravity term
  std::optional<Tensor<1, dim>> gravity_field_value;

  // source term
  std::optional<Tensor<1, dim>> reference_density_gradient;
};



template<int dim>
struct OptionalArgumentsStrongForm : OptionalArguments
{
  OptionalArgumentsStrongForm(const bool allocate_gravity_field,
                              const bool allocate_reference_density,
                              const unsigned int n_q_points);

  OptionalArgumentsStrongForm(const OptionalArgumentsStrongForm<dim> &other);

  // gravity term
  std::optional<std::vector<Tensor<1, dim>>> gravity_field_values;

  // source term
  std::optional<std::vector<Tensor<1, dim>>> reference_density_gradients;
};

}  // namespace BuoyantHydrodynamic


#endif /* INCLUDE_BUOYANT_HYDRODYNAMIC_OPTIONS_H_ */
