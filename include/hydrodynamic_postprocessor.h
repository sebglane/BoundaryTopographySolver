/*
 * hydrodynamic_postprocessor.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_HYDRODYNAMIC_POSTPROCESSOR_H_
#define INCLUDE_HYDRODYNAMIC_POSTPROCESSOR_H_

#include <deal.II/base/tensor_function.h>

#include <deal.II/numerics/data_postprocessor.h>

namespace Hydrodynamic {

using namespace dealii;

template<int dim>
class Postprocessor : public DataPostprocessor<dim>
{
public:
  Postprocessor(const unsigned int velocity_start_index,
                const unsigned int pressure_index);

  void set_background_velocity(const TensorFunction<1, dim> &velocity);

  virtual void evaluate_vector_field
  (const DataPostprocessorInputs::Vector<dim> &inputs,
   std::vector<Vector<double>>                &computed_quantities) const;

  virtual std::vector<std::string> get_names() const;

  virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const;

  virtual UpdateFlags get_needed_update_flags() const;

private:
  const TensorFunction<1, dim>  *background_velocity_ptr;

  const unsigned int velocity_start_index;

  const unsigned int pressure_index;
};

// inline functions
template <int dim>
inline void Postprocessor<dim>::set_background_velocity(const TensorFunction<1, dim> &velocity)
{
  background_velocity_ptr = &velocity;
}

}  // namespace Hydrodynamic

#endif /* INCLUDE_HYDRODYNAMIC_POSTPROCESSOR_H_ */
