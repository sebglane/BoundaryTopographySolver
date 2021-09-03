/*
 * postprocessor.cc
 *
 *  Created on: Dec 11, 2018
 *      Author: sg
 */

#include <postprocessor.h>

namespace TopographyProblem {

template<int dim>
PostProcessor<dim>::PostProcessor()
:
DataPostprocessor<dim>(),
background_velocity()
{}

template<int dim>
std::vector<std::string> PostProcessor<dim>::get_names() const
{
    std::vector<std::string> solution_names;
    // density
    solution_names.push_back("density");
    // velocity
    for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back("velocity");
    // pressure
    solution_names.push_back("pressure");
    // total velocity
    for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back("total_velocity");
    // magnetic field
    for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back("magnetic_field");
    // total magnetic field
    for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back("total_magnetic_field");
    // scalar field
    solution_names.push_back("scalar_field");

    return solution_names;
}

template<int dim>
UpdateFlags PostProcessor<dim>::get_needed_update_flags() const
{
    return update_values;
}

template<int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
PostProcessor<dim>::get_data_component_interpretation() const
{
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation;

    // density
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    // velocity
    for (unsigned int d=0; d<dim; ++d)
        component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    // pressure
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    // total velocity
    for (unsigned int d=0; d<dim; ++d)
        component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    // magnetic field
    for (unsigned int d=0; d<dim; ++d)
        component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    // total magnetic field
    for (unsigned int d=0; d<dim; ++d)
        component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    // scalar field
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    return component_interpretation;
}

template<int dim>
void PostProcessor<dim>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim>  &inputs,
        std::vector<Vector<double>>                 &computed_quantities) const
{

    const unsigned int n_quadrature_points = inputs.solution_values.size();

    Vector<double>      background_velocity_values(background_velocity.n_components);
    Vector<double>      background_field_values(background_velocity.n_components);

    AssertDimension(computed_quantities.size(),
                    n_quadrature_points);
    AssertDimension(inputs.solution_values[0].size(),
                    2*dim+3);

    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
        AssertDimension(computed_quantities[q].size(), 4*dim+3);
        // density
        computed_quantities[q][0] = inputs.solution_values[q][0];
        // velocity
        for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q][d+1] = inputs.solution_values[q][d+1];
        // pressure
        computed_quantities[q][dim+1] = inputs.solution_values[q][dim+1];
        // total velocity
        background_velocity.vector_value(inputs.evaluation_points[q],
                                         background_velocity_values);
        for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q][d+dim+2] = inputs.solution_values[q][d+1]
                                            + background_velocity_values[d]  ;
        // magnetic field
        for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q][d+2*dim+2] = inputs.solution_values[q][d+dim+2];
        // total magnetic field
        background_field.vector_value(inputs.evaluation_points[q],
                                      background_field_values);
        for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q][d+3*dim+2] = inputs.solution_values[q][d+dim+2]
                                              + background_velocity_values[d]  ;
        // scalar field
        computed_quantities[q][4*dim+2] = inputs.solution_values[q][2*dim+2];
    }
}
}  // namespace TopographyProblem

// explicit instantiation
template class TopographyProblem::PostProcessor<3>;
