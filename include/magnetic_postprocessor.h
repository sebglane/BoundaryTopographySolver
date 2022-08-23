/*
 * magnetic_postprocessor.h
 *
 *  Created on: Aug 22, 2022
 *      Author: sg
 */

#ifndef MAGNETIC_POSTPROCESSOR_H_
#define MAGNETIC_POSTPROCESSOR_H_

#include <deal.II/base/tensor_function.h>

#include <deal.II/numerics/data_postprocessor.h>

#include <memory>

namespace MagneticInduction {

using namespace dealii;

/**
 * @todo Add documentation.
 */
template<int dim>
class Postprocessor : public DataPostprocessor<dim>
{
public:
  Postprocessor(const unsigned int magnetic_field_start_index,
                const unsigned int magnetic_pressure_index);

  void set_background_magnetic_field(const std::shared_ptr<const TensorFunction<1, dim>> &magnetic_field);

  virtual void evaluate_vector_field
  (const DataPostprocessorInputs::Vector<dim> &inputs,
   std::vector<Vector<double>>                &computed_quantities) const;

  virtual std::vector<std::string> get_names() const;

  virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const;

  virtual UpdateFlags get_needed_update_flags() const;

private:
  std::shared_ptr<const TensorFunction<1, dim>> background_magnetic_field_ptr;

  const unsigned int magnetic_field_start_index;

  const unsigned int magnetic_pressure_index;
};

// inline functions
template <int dim>
inline void Postprocessor<dim>::set_background_magnetic_field
(const std::shared_ptr<const TensorFunction<1, dim>> &magnetic_field)
{
  background_magnetic_field_ptr = magnetic_field;
}

}  // namespace MagneticInduction




#endif /* MAGNETIC_POSTPROCESSOR_H_ */
