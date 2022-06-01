/*
 * hydrodynamic_assembly_data.h
 *
 *  Created on: Apr 11, 2022
 *      Author: sg
 */

#ifndef INCLUDE_HYDRODYNAMIC_ASSEMBLY_DATA_H_
#define INCLUDE_HYDRODYNAMIC_ASSEMBLY_DATA_H_

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/meshworker/scratch_data.h>

#include <hydrodynamic_options.h>
#include <stabilization_flags.h>

#include <memory>

namespace Hydrodynamic {

using namespace dealii;

namespace AssemblyData {

namespace Matrix {

template <int dim>
class ScratchData : virtual public MeshWorker::ScratchData<dim>
{
public:
  ScratchData(
    const Mapping<dim>       &mapping,
    const FiniteElement<dim> &fe,
    const Quadrature<dim>    &quadrature,
    const UpdateFlags        &update_flags,
    const Quadrature<dim-1>  &face_quadrature = Quadrature<dim-1>(),
    const UpdateFlags        &face_update_flags = update_default,
    const StabilizationFlags &stabilization_flags = apply_none,
    const bool                use_stress_form = false,
    const bool                allocate_background_velocity = false,
    const bool                allocate_body_force = false,
    const bool                allocate_traction = false);

  ScratchData(
    const FiniteElement<dim> &fe,
    const Quadrature<dim>    &quadrature,
    const UpdateFlags        &update_flags,
    const Quadrature<dim-1>  &face_quadrature   = Quadrature<dim-1>(),
    const UpdateFlags        &face_update_flags = update_default,
    const StabilizationFlags &stabilization_flags = apply_none,
    const bool                use_stress_form = false,
    const bool                allocate_background_velocity = false,
    const bool                allocate_body_force = false,
    const bool                allocate_traction = false);

  ScratchData(const ScratchData<dim>  &data);

  void assign_vector_options_local_cell(
    const std::string                                          &name,
    const FEValuesExtractors::Vector                           &velocity,
    const FEValuesExtractors::Scalar                           &pressure,
    const std::shared_ptr<const Utility::AngularVelocity<dim>> &angular_velocity_ptr = nullptr,
    const std::shared_ptr<const TensorFunction<1,dim>>         &body_force_ptr = nullptr,
    const std::shared_ptr<const TensorFunction<1,dim>>         &background_velocity_ptr = nullptr,
    const double                                                rossby_number = 0.0,
    const double                                                froude_number = 0.0);

  void assign_vector_options_local_boundary(
    const std::string                                          &name,
    const FEValuesExtractors::Vector                           &velocity,
    const FEValuesExtractors::Scalar                           &pressure,
    const double                                                nu,
    const std::shared_ptr<const TensorFunction<1,dim>>         &boundary_traction_ptr = nullptr,
    const std::shared_ptr<const TensorFunction<1,dim>>         &background_velocity_ptr = nullptr);

  void assign_scalar_options_local_cell(const unsigned int q_point_index);

  void assign_optional_shape_functions_local_cell(
    const FEValuesExtractors::Vector &velocity,
    const FEValuesExtractors::Scalar &pressure,
    const unsigned int                q_point_index);

  void assign_optional_shape_functions_local_boundary(
    const FEValuesExtractors::Vector &velocity,
    const unsigned int                q_point_index);

  void adjust_velocity_field_local_cell();

  void adjust_velocity_field_local_boundary();

  const StabilizationFlags  &stabilization_flags;

  VectorOptions<dim>  vector_options;
  ScalarOptions<dim>  scalar_options;

  // shape functions
  std::vector<Tensor<1, dim>> phi_velocity;
  std::vector<Tensor<2, dim>> grad_phi_velocity;
  std::vector<double>         div_phi_velocity;
  std::vector<double>         phi_pressure;

  // stabilization related shape functions
  std::vector<Tensor<1, dim>> grad_phi_pressure;

  // present solution values which cannot be referenced
  std::vector<Tensor<1, dim>> present_velocity_values;
  std::vector<Tensor<2, dim>> present_velocity_gradients;

  // stress tensor related shape functions
  std::vector<SymmetricTensor<2, dim>>  sym_grad_phi_velocity;

  // stabilization related shape functions
  std::vector<Tensor<1, dim>> laplace_phi_velocity;

  // stress tensor and stabilization related shape functions
  std::vector<Tensor<1, dim>> grad_div_phi_velocity;

  // stabilization related quantities
  std::vector<Tensor<1, dim>> present_strong_residuals;
};

}  // namespace Matrix



namespace RightHandSide
{

template <int dim>
class ScratchData : virtual public MeshWorker::ScratchData<dim>
{
public:

  ScratchData(
    const Mapping<dim>       &mapping,
    const FiniteElement<dim> &fe,
    const Quadrature<dim>    &quadrature,
    const UpdateFlags        &update_flags,
    const Quadrature<dim-1>  &face_quadrature = Quadrature<dim-1>(),
    const UpdateFlags        &face_update_flags = update_default,
    const StabilizationFlags &stabilization_flags = apply_none,
    const bool                use_stress_form = false,
    const bool                allocate_background_velocity = false,
    const bool                allocate_body_force = false,
    const bool                allocate_traction = false);

  ScratchData(
    const FiniteElement<dim> &fe,
    const Quadrature<dim>    &quadrature,
    const UpdateFlags        &update_flags,
    const Quadrature<dim-1>  &face_quadrature   = Quadrature<dim-1>(),
    const UpdateFlags        &face_update_flags = update_default,
    const StabilizationFlags &stabilization_flags = apply_none,
    const bool                use_stress_form = false,
    const bool                allocate_background_velocity = false,
    const bool                allocate_body_force = false,
    const bool                allocate_traction = false);

  ScratchData(const ScratchData<dim>  &data);

  void assign_vector_options_local_cell(
    const std::string                                          &name,
    const FEValuesExtractors::Vector                           &velocity,
    const FEValuesExtractors::Scalar                           &pressure,
    const std::shared_ptr<const Utility::AngularVelocity<dim>> &angular_velocity_ptr = nullptr,
    const std::shared_ptr<const TensorFunction<1,dim>>         &body_force_ptr = nullptr,
    const std::shared_ptr<const TensorFunction<1,dim>>         &background_velocity_ptr = nullptr,
    const double                                                rossby_number = 0.0,
    const double                                                froude_number = 0.0);

  void assign_vector_options_local_boundary(
    const std::string                                          &name,
    const FEValuesExtractors::Vector                           &velocity,
    const FEValuesExtractors::Scalar                           &pressure,
    const double                                                nu,
    const std::shared_ptr<const TensorFunction<1,dim>>         &boundary_traction_ptr = nullptr,
    const std::shared_ptr<const TensorFunction<1,dim>>         &background_velocity_ptr = nullptr);

  void assign_scalar_options_local_cell(const unsigned int q_point_index);

  void assign_optional_shape_functions_local_cell(
    const FEValuesExtractors::Vector &velocity,
    const FEValuesExtractors::Scalar &pressure,
    const unsigned int                q_point_index);

  void adjust_velocity_field_local_cell();

  void adjust_velocity_field_local_boundary();

  const StabilizationFlags  &stabilization_flags;

  VectorOptions<dim>  vector_options;
  ScalarOptions<dim>  scalar_options;

  // shape functions
  std::vector<Tensor<1, dim>> phi_velocity;
  std::vector<Tensor<2, dim>> grad_phi_velocity;
  std::vector<double>         div_phi_velocity;
  std::vector<double>         phi_pressure;

  // stabilization related shape functions
  std::vector<Tensor<1, dim>> grad_phi_pressure;

  // present solution values which cannot be referenced
  std::vector<Tensor<1, dim>> present_velocity_values;
  std::vector<Tensor<2, dim>> present_velocity_gradients;

  // stress tensor related shape functions
  std::vector<SymmetricTensor<2, dim>>  sym_grad_phi_velocity;

  // stabilization related quantities
  std::vector<Tensor<1, dim>> present_strong_residuals;
};

}  // namespace RightHandSide

}  // namespace AssemblyData

}  // namespace Hydrodynamic



#endif /* INCLUDE_HYDRODYNAMIC_ASSEMBLY_DATA_H_ */
