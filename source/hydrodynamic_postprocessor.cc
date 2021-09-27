/*
 * hydrodynamic_postprocessor.cc
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#include <hydrodynamic_postprocessor.h>

namespace Hydrodynamic {

template<int dim>
Postprocessor<dim>::Postprocessor
(const unsigned int velocity_start_index,
 const unsigned int pressure_index)
:
DataPostprocessor<dim>(),
background_velocity_ptr(nullptr),
velocity_start_index(velocity_start_index),
pressure_index(pressure_index)
{}



template<int dim>
std::vector<std::string> Postprocessor<dim>::get_names() const
{
  std::vector<std::string> solution_names;

  // velocity
  if (background_velocity_ptr != nullptr)
    for (unsigned int d=0; d<dim; ++d)
      solution_names.push_back("velocity_perturbation");
  else
    for (unsigned int d=0; d<dim; ++d)
      solution_names.push_back("velocity");
  // vorticity
  if constexpr(dim == 2)
    solution_names.emplace_back("vorticity");
  else if constexpr(dim == 3)
    for (unsigned int d=0; d<dim; ++d)
      solution_names.emplace_back("vorticity");
  // pressure
  solution_names.push_back("pressure");
  // pressure gradient
  for (unsigned int d=0; d<dim; ++d)
    solution_names.push_back("pressure_gradient");
  // background velocity
  if (background_velocity_ptr != nullptr)
    for (unsigned int d=0; d<dim; ++d)
      solution_names.push_back("velocity");

  return (solution_names);
}



template<int dim>
UpdateFlags Postprocessor<dim>::get_needed_update_flags() const
{
  UpdateFlags update_flags{update_values|update_gradients};
  if (background_velocity_ptr != nullptr)
    update_flags |= update_quadrature_points;

  return (update_flags);
}



template<int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
Postprocessor<dim>::get_data_component_interpretation() const
{
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation;

  // velocity
  for (unsigned int d=0; d<dim; ++d)
    component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
  // vorticity
  if constexpr(dim == 2)
  component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  else if constexpr(dim == 3)
    for (unsigned int d=0; d<dim; ++d)
      component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
  // pressure
  component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  // pressure gradient
  for (unsigned int d=0; d<dim; ++d)
    component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
  // background velocity
  if (background_velocity_ptr != nullptr)
    for (unsigned int d=0; d<dim; ++d)
      component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);

  return (component_interpretation);
}



template<int dim>
void Postprocessor<dim>::evaluate_vector_field
(const DataPostprocessorInputs::Vector<dim>  &inputs,
 std::vector<Vector<double>>                 &computed_quantities) const
{
  const std::size_t n_quadrature_points{inputs.solution_values.size()};

  AssertDimension(computed_quantities.size(), n_quadrature_points);
  {
    const std::size_t n_solution_components{inputs.solution_values[0].size()};
    Assert(dim + 1 <= n_solution_components,
           ExcLowerRange(dim + 1, n_solution_components));
    Assert(velocity_start_index < n_solution_components,
           ExcLowerRange(velocity_start_index, n_solution_components));
    Assert(velocity_start_index + dim < n_solution_components,
           ExcLowerRange(velocity_start_index + dim, n_solution_components));
    Assert(pressure_index < n_solution_components,
           ExcLowerRange(pressure_index, n_solution_components));
  }

  Tensor<1, dim> background_velocity;

  for (unsigned int q=0; q<n_quadrature_points; ++q)
  {
    unsigned int cnt = 0;

    // velocity
    for (unsigned int d=0; d<dim; ++d)
      computed_quantities[q](d) = inputs.solution_values[q][velocity_start_index + d];
    cnt += dim;
    // vorticity
    if constexpr(dim == 2)
    {
      computed_quantities[q](cnt) = inputs.solution_gradients[q][velocity_start_index + 1][0]
                                  - inputs.solution_gradients[q][velocity_start_index][1];
      cnt += 1;
    }
    else if constexpr(dim == 3)
    {
      computed_quantities[q](cnt) = inputs.solution_gradients[q][velocity_start_index + 2][1]
                                  - inputs.solution_gradients[q][velocity_start_index + 1][2];
      computed_quantities[q](cnt) = inputs.solution_gradients[q][velocity_start_index][2]
                                  - inputs.solution_gradients[q][velocity_start_index + 2][0];
      computed_quantities[q](cnt) = inputs.solution_gradients[q][velocity_start_index + 1][0]
                                  - inputs.solution_gradients[q][velocity_start_index][1];
      cnt += 3;
    }
    // pressure
    computed_quantities[q](cnt) = inputs.solution_values[q][pressure_index];
    cnt += 1;
    // pressure gradient
    for (unsigned int d=0; d<dim; ++d)
      computed_quantities[q](cnt) = inputs.solution_gradients[q][pressure_index][d];
    cnt += dim;
    // background value
    if (background_velocity_ptr != nullptr)
    {
      background_velocity = background_velocity_ptr->value(inputs.evaluation_points[q]);
      for (unsigned int d=0; d<dim; ++d)
        computed_quantities[q](cnt) = inputs.solution_values[q][velocity_start_index + d] +
                                      background_velocity[d];
      cnt += dim;
    }
  }
}

// explicit instantiation
template class Postprocessor<2>;
template class Postprocessor<3>;

}  // namespace Hydrodynamic

