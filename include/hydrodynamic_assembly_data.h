/*
 * hydrodynamic_assembly_data.h
 *
 *  Created on: Apr 11, 2022
 *      Author: sg
 */

#ifndef INCLUDE_HYDRODYNAMIC_ASSEMBLY_DATA_H_
#define INCLUDE_HYDRODYNAMIC_ASSEMBLY_DATA_H_

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/scratch_data.h>

#include <hydrodynamic_options.h>
#include <stabilization_flags.h>

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

  const StabilizationFlags  &stabilization_flags;

  OptionalVectorArguments<dim>  vector_options;
  OptionalScalarArguments<dim>  scalar_options;

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

  const StabilizationFlags  &stabilization_flags;

  OptionalVectorArguments<dim>  vector_options;
  OptionalScalarArguments<dim>  scalar_options;

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
