/*
 * postprocessor_base.h
 *
 *  Created on: Sep 7, 2021
 *      Author: sg
 */

#ifndef INCLUDE_POSTPROCESSOR_BASE_H_
#define INCLUDE_POSTPROCESSOR_BASE_H_

#include <deal.II/numerics/data_postprocessor.h>

namespace Utility {

using namespace dealii;

/*!
 * @class PostprocessorScalarField
 *
 * @brief A postprocessor for a scalar finite element field.
 *
 * @details This postprocessor outputs the field itself and the gradient.
 */
template<int dim>
class PostprocessorScalarField : public DataPostprocessor<dim>
{
public:
  /*!
   * @brief Default constructor specifying the name of the field.
   */
  PostprocessorScalarField(const std::string &name,
                           const unsigned int component_index);

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::evaluate_scalar_field`.
   *
   * @details Value and gradient of the field is written as an output.
   */
  virtual void evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_names`.
   */
  virtual std::vector<std::string> get_names() const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_data_component_interpretation`.
   */
  virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_needed_update_flags`.
   */
  virtual UpdateFlags get_needed_update_flags() const override;

private:
  /*!
   * @brief Name of the scalar finite element field.
   */
  const std::string   name;

  const unsigned int  component_index;
};



template<int dim>
class PostprocessorVectorField : public DataPostprocessor<dim>
{
public:
  /*!
   * @brief Default constructor specifying the name of the field.
   */
  PostprocessorVectorField(const std::string &name,
                           const unsigned int first_index);

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::evaluate_scalar_field`.
   *
   * @details The field, its curl, the helicity and the invariants of the gradient
   * are written as an output.
   */
  virtual void evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_data_component_interpretation`.
   */
  virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_names`.
   */
  virtual std::vector<std::string> get_names() const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_needed_update_flags`.
   */
  virtual UpdateFlags get_needed_update_flags() const override;

private:
  /*!
   * @brief Name of the scalar finite element field.
   */
  const std::string   name;

  const unsigned int  first_index;

};


}  // namespace Utility



#endif /* INCLUDE_POSTPROCESSOR_BASE_H_ */
