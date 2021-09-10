/*
 * postprocessor_base.cc
 *
 *  Created on: Sep 7, 2021
 *      Author: sg
 */

#include <postprocessor_base.h>

namespace Utility {

template <int dim>
PostprocessorScalarField<dim>::PostprocessorScalarField
(const std::string &name,
 const unsigned int component_index)
:
DataPostprocessor<dim>(),
name(name),
component_index(component_index)
{}



template <int dim>
void PostprocessorScalarField<dim>::evaluate_vector_field
(const DataPostprocessorInputs::Vector<dim> &inputs,
 std::vector<Vector<double>> &computed_quantities) const
{

  const unsigned int n_quadrature_points = inputs.solution_values.size();
  Assert(computed_quantities.size() == n_quadrature_points,
         ExcDimensionMismatch(computed_quantities.size(),
                              n_quadrature_points));

  for (unsigned int q=0; q<n_quadrature_points; ++q)
  {
    unsigned int cnt = 0;

    // solution value
    computed_quantities[q](cnt) = inputs.solution_values[q][component_index];
    cnt += 1;
    // gradient of the solution
    for (unsigned int d=0; d<dim; ++d)
    {
      computed_quantities[q](cnt) = inputs.solution_gradients[q][component_index][d];
      cnt += 1;
    }
  }
}



template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
PostprocessorScalarField<dim>::get_data_component_interpretation() const
{
  // solution
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation(1, DataComponentInterpretation::component_is_scalar);
  // gradient of the solution
  for (unsigned int d=0; d<dim; ++d)
    component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);

  return (component_interpretation);
}



template <int dim>
std::vector<std::string> PostprocessorScalarField<dim>::get_names() const
{
  // the solution
  std::vector<std::string> solution_names(1, name);
  // gradient of the solution
  for (unsigned int d=0; d<dim; ++d)
    solution_names.emplace_back(name + "_gradient");

  return (solution_names);
}



template <int dim>
UpdateFlags PostprocessorScalarField<dim>::get_needed_update_flags() const
{
  return update_values|update_gradients;
}



template <int dim>
PostprocessorVectorField<dim>::PostprocessorVectorField
(const std::string &name,
 const unsigned int first_index)
:
DataPostprocessor<dim>(),
name(name),
first_index(first_index)
{}



template <int dim>
void PostprocessorVectorField<dim>::evaluate_vector_field
(const DataPostprocessorInputs::Vector<dim> &inputs,
 std::vector<Vector<double>> &computed_quantities) const
{

  const unsigned int n_quadrature_points = inputs.solution_values.size();
  Assert(computed_quantities.size() == n_quadrature_points,
         ExcDimensionMismatch(computed_quantities.size(),
                              n_quadrature_points));
  Assert(first_index + dim <= inputs.solution_values[0].size(),
         ExcDimensionMismatch(first_index + dim,
                              inputs.solution_values[0].size()));

  for (unsigned int q=0; q<n_quadrature_points; ++q)
  {
    unsigned int cnt = 0;

    // solution values
    for (unsigned int d=0; d<dim; ++d)
    {
      computed_quantities[q](cnt) = inputs.solution_values[q](first_index + d);
      cnt += 1;
    }
    // curl of the solution
    if constexpr(dim == 2)
    {
      computed_quantities[q](cnt) = inputs.solution_gradients[q][first_index + 1][0]
                                  - inputs.solution_gradients[q][first_index][1];
      cnt += 1;
    }
    else if constexpr(dim == 3)
    {
      computed_quantities[q](cnt) = inputs.solution_gradients[q][first_index + 2][1]
                                  - inputs.solution_gradients[q][first_index + 1][2];
      computed_quantities[q](cnt) = inputs.solution_gradients[q][first_index + 0][2]
                                  - inputs.solution_gradients[q][first_index + 2][0];
      computed_quantities[q](cnt) = inputs.solution_gradients[q][first_index + 1][0]
                                  - inputs.solution_gradients[q][first_index + 0][1];
      cnt += 3;
    }
  }
}



template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
PostprocessorVectorField<dim>::get_data_component_interpretation() const
{
  // solution
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  // curl of the solution
  if constexpr(dim == 2)
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  else if constexpr(dim == 3)
    for (unsigned int d=0; d<dim; ++d)
      component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);

  return (component_interpretation);
}



template <int dim>
std::vector<std::string> PostprocessorVectorField<dim>::get_names() const
{
  // the solution
  std::vector<std::string> solution_names(dim, name);
  // curl of the solution
  if constexpr(dim == 2)
    solution_names.emplace_back(name + "_curl");
  else if constexpr(dim == 3)
    for (unsigned int d=0; d<dim; ++d)
      solution_names.emplace_back(name + "_curl");

  return (solution_names);
}



template <int dim>
UpdateFlags PostprocessorVectorField<dim>::get_needed_update_flags() const
{
  return update_values|update_gradients;
}



// explicit instantiations
template class PostprocessorScalarField<2>;
template class PostprocessorScalarField<3>;

template class PostprocessorVectorField<2>;
template class PostprocessorVectorField<3>;

}  // namespace Utility
