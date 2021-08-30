/*
 * postprocessor.h
 *
 *  Created on: Dec 11, 2018
 *      Author: sg
 */

#ifndef INCLUDE_POSTPROCESSOR_H_
#define INCLUDE_POSTPROCESSOR_H_

#include <deal.II/numerics/data_postprocessor.h>

#include <equation_data.h>

namespace TopographyProblem {

using namespace dealii;

template<int dim>
class PostProcessor : public DataPostprocessor<dim>
{
public:
    PostProcessor();

    virtual void evaluate_vector_field(
                const DataPostprocessorInputs::Vector<dim> &inputs,
                std::vector<Vector<double>>                &computed_quantities) const;

    virtual std::vector<std::string> get_names() const;

    virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const;

    virtual UpdateFlags get_needed_update_flags() const;

private:
    const EquationData::BackgroundVelocity<dim>         background_velocity;
    const EquationData::BackgroundMagneticField<dim>    background_field;
};

}  // namespace TopographyProblem

#endif /* INCLUDE_POSTPROCESSOR_H_ */
