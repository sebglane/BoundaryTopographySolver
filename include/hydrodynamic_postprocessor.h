/*
 * hydrodynamic_postprocessor.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_HYDRODYNAMIC_POSTPROCESSOR_H_
#define INCLUDE_HYDRODYNAMIC_POSTPROCESSOR_H_

#include <deal.II/numerics/data_postprocessor.h>

namespace TopographyProblem {

using namespace dealii;

template<int dim>
class HydrodynamicPostprocessor : public DataPostprocessor<dim>
{
public:
  HydrodynamicPostprocessor(const unsigned int velocity_start_index,
                            const unsigned int pressure_index);

  virtual void evaluate_vector_field
  (const DataPostprocessorInputs::Vector<dim> &inputs,
   std::vector<Vector<double>>                &computed_quantities) const;

  virtual std::vector<std::string> get_names() const;

  virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const;

  virtual UpdateFlags get_needed_update_flags() const;

private:
  const unsigned int velocity_start_index;

  const unsigned int pressure_index;
};

}  // namespace TopographyProblem

#endif /* INCLUDE_HYDRODYNAMIC_POSTPROCESSOR_H_ */
