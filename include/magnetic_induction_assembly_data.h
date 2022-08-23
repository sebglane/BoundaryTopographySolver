/*
 * magnetic_induction_assembly_data.h
 *
 *  Created on: Apr 11, 2022
 *      Author: sg
 */

#ifndef INCLUDE_MAGNETIC_INDUCTION_ASSEMBLY_DATA_H_
#define INCLUDE_MAGNETIC_INDUCTION_ASSEMBLY_DATA_H_

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/meshworker/scratch_data.h>

#include <magnetic_induction_options.h>

#include <memory>

namespace MagneticInduction {

using namespace dealii;

namespace AssemblyData {

template <int dim>
class ScratchData : virtual public MeshWorker::ScratchData<dim>
{
public:
  ScratchData(
    const Mapping<dim>       &mapping,
    const FiniteElement<dim> &fe,
    const Quadrature<dim>    &quadrature,
    const UpdateFlags        &update_flags,
    const bool                allocate_velocity_field = false,
    const bool                allocate_background_magnetic_field = false);

  ScratchData(
    const FiniteElement<dim> &fe,
    const Quadrature<dim>    &quadrature,
    const UpdateFlags        &update_flags,
    const bool                allocate_velocity_field = false,
    const bool                allocate_background_velocity = false);

  ScratchData(const ScratchData<dim>  &data);

  /*
   *

  void assign_vector_options(
    const std::string                                          &name,
    const FEValuesExtractors::Vector                           &magnetic_field,
    const FEValuesExtractors::Scalar                           &magnetic_pressure,
    const std::shared_ptr<const TensorFunction<1,dim>>         &background_magnetic_field_ptr = nullptr);

  void assign_vector_options_boundary(
    const std::string                                          &name,
    const FEValuesExtractors::Vector                           &magnetic_field,
    const FEValuesExtractors::Scalar                           &magnetic_pressure,
    const std::shared_ptr<const TensorFunction<1,dim>>         &background_magnetic_field_ptr = nullptr);

  void adjust_velocity_field_local_cell();

  *
  */
  using curl_type = typename FEValuesViews::Vector<dim>::curl_type;

  VectorOptions<dim>          vector_options;

  // shape functions
  std::vector<Tensor<1, dim>> phi_magnetic_field;
  std::vector<curl_type>      curl_phi_magnetic_field;
  std::vector<double>         div_phi_magnetic_field;
  std::vector<Tensor<1, dim>> grad_phi_magnetic_pressure;

  // present solution values
  std::vector<Tensor<1, dim>> present_magnetic_field_values;
  std::vector<curl_type>      present_magnetic_field_curls;
  std::vector<double>         present_magnetic_field_divergences;
  std::vector<Tensor<1, dim>> present_magnetic_pressure_gradients;

};

namespace Matrix {

template <int dim>
using ScratchData = ScratchData<dim>;

}  // namespace Matrix



namespace RightHandSide {

template <int dim>
using ScratchData = ScratchData<dim>;

}  // namespace RightHandSide

}  // namespace AssemblyData

}  // namespace MagneticInduction



#endif /* INCLUDE_MAGNETIC_INDUCTION_ASSEMBLY_DATA_H_ */