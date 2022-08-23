/*
 * magnetic_postprocessor.cc
 *
 *  Created on: Aug 22, 2022
 *      Author: sg
 */

#include <magnetic_postprocessor.h>

namespace MagneticInduction {

template<int dim>
Postprocessor<dim>::Postprocessor
(const unsigned int magnetic_field_start_index,
 const unsigned int magnetic_pressure_index)
:
DataPostprocessor<dim>(),
background_magnetic_field_ptr(),
magnetic_field_start_index(magnetic_field_start_index),
magnetic_pressure_index(magnetic_pressure_index)
{}



template<int dim>
std::vector<std::string> Postprocessor<dim>::get_names() const
{
  std::vector<std::string> solution_names;

  // magnetic field
  if (background_magnetic_field_ptr)
    for (unsigned int d=0; d<dim; ++d)
      solution_names.push_back("magnetic_field_perturbation");
  else
    for (unsigned int d=0; d<dim; ++d)
      solution_names.push_back("magnetic_field");
  // magnetic pressure
  solution_names.push_back("magnetic_pressure");
  // magnetic pressure gradient
  for (unsigned int d=0; d<dim; ++d)
    solution_names.push_back("magnetic_pressure_gradient");
  // background magnetic field
  if (background_magnetic_field_ptr)
    for (unsigned int d=0; d<dim; ++d)
      solution_names.push_back("magnetic_field");

  return (solution_names);
}



template<int dim>
UpdateFlags Postprocessor<dim>::get_needed_update_flags() const
{
  UpdateFlags update_flags{update_values|update_gradients};
  if (background_magnetic_field_ptr)
    update_flags |= update_quadrature_points;

  return (update_flags);
}



template<int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
Postprocessor<dim>::get_data_component_interpretation() const
{
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation;

  // magnetic field
  for (unsigned int d=0; d<dim; ++d)
    component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
  // magnetic pressure
  component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  // magnetic pressure gradient
  for (unsigned int d=0; d<dim; ++d)
    component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
  // background magnetic field
  if (background_magnetic_field_ptr)
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
    Assert(magnetic_field_start_index < n_solution_components,
           ExcLowerRange(magnetic_field_start_index, n_solution_components));
    Assert(magnetic_field_start_index + dim < n_solution_components,
           ExcLowerRange(magnetic_field_start_index + dim, n_solution_components));
    Assert(magnetic_pressure_index < n_solution_components,
           ExcLowerRange(magnetic_pressure_index, n_solution_components));
  }

  Tensor<1, dim> background_magnetic_field;

  for (unsigned int q=0; q<n_quadrature_points; ++q)
  {
    unsigned int cnt = 0;

    // magnetic field
    for (unsigned int d=0; d<dim; ++d)
      computed_quantities[q](d) = inputs.solution_values[q][magnetic_field_start_index + d];
    cnt += dim;
    // magnetic pressure
    computed_quantities[q](cnt) = inputs.solution_values[q][magnetic_pressure_index];
    cnt += 1;
    // magnetic pressure gradient
    for (unsigned int d=0; d<dim; ++d)
      computed_quantities[q](cnt + d) = inputs.solution_gradients[q][magnetic_pressure_index][d];
    cnt += dim;
    // background magnetic field
    if (background_magnetic_field_ptr)
    {
      background_magnetic_field = background_magnetic_field_ptr->value(inputs.evaluation_points[q]);
      for (unsigned int d=0; d<dim; ++d)
        computed_quantities[q](cnt + d) = inputs.solution_values[q][magnetic_field_start_index + d] +
                                          background_magnetic_field[d];
      cnt += dim;
    }
  }
}

// explicit instantiation
template class Postprocessor<2>;
template class Postprocessor<3>;

}  // namespace Hydrodynamic
