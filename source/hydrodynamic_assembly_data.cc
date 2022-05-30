/*
 * hydrodynamic_assembly_data.cc
 *
 *  Created on: Sep 25, 2021
 *      Author: sg
 */

#include <hydrodynamic_assembly_data.h>

namespace Hydrodynamic {

namespace AssemblyData {

namespace Matrix {

template <int dim>
ScratchData<dim>::ScratchData
(const Mapping<dim>       &mapping,
 const FiniteElement<dim> &fe,
 const Quadrature<dim>    &quadrature,
 const UpdateFlags        &update_flags,
 const Quadrature<dim-1>  &face_quadrature,
 const UpdateFlags        &face_update_flags,
 const StabilizationFlags &stabilization_flags,
 const bool                use_stress_form,
 const bool                allocate_background_velocity,
 const bool                allocate_body_force,
 const bool                allocate_traction)
:
MeshWorker::ScratchData<dim>(mapping,
                             fe,
                             quadrature,
                             update_flags,
                             face_quadrature,
                             face_update_flags),
stabilization_flags(stabilization_flags),
vector_options(stabilization_flags,
               use_stress_form,
               allocate_background_velocity,
               allocate_body_force,
               allocate_traction,
               quadrature.size(),
               face_quadrature.size()),
scalar_options(use_stress_form),
phi_velocity(fe.n_dofs_per_cell()),
grad_phi_velocity(fe.n_dofs_per_cell()),
div_phi_velocity(fe.n_dofs_per_cell()),
phi_pressure(fe.n_dofs_per_cell()),
grad_phi_pressure(fe.n_dofs_per_cell()),
present_velocity_values(quadrature.size()),
present_velocity_gradients(quadrature.size()),
sym_grad_phi_velocity(),
laplace_phi_velocity(),
grad_div_phi_velocity(),
present_strong_residuals()
{
  if (use_stress_form)
    sym_grad_phi_velocity.resize(fe.n_dofs_per_cell());

  // stabilization related objects
  if (stabilization_flags & (apply_supg|apply_pspg))
  {
    laplace_phi_velocity.resize(fe.n_dofs_per_cell());
    if (use_stress_form)
      grad_div_phi_velocity.resize(fe.n_dofs_per_cell());

    present_strong_residuals.resize(quadrature.size());
  }
}



template <int dim>
ScratchData<dim>::ScratchData
(const FiniteElement<dim> &fe,
 const Quadrature<dim>    &quadrature,
 const UpdateFlags        &update_flags,
 const Quadrature<dim-1>  &face_quadrature,
 const UpdateFlags        &face_update_flags,
 const StabilizationFlags &stabilization_flags,
 const bool                use_stress_form,
 const bool                allocate_background_velocity,
 const bool                allocate_body_force,
 const bool                allocate_traction)
:
ScratchData<dim>(fe.reference_cell()
                 .template get_default_linear_mapping<dim>(),
                 fe,
                 quadrature,
                 update_flags,
                 face_quadrature,
                 face_update_flags,
                 stabilization_flags,
                 use_stress_form,
                 allocate_background_velocity,
                 allocate_body_force,
                 allocate_traction)
{}

template <int dim>
ScratchData<dim>::ScratchData(const ScratchData<dim>  &other)
:
MeshWorker::ScratchData<dim>(other),
stabilization_flags(other.stabilization_flags),
vector_options(other.vector_options),
scalar_options(other.scalar_options),
phi_velocity(other.phi_velocity),
grad_phi_velocity(other.grad_phi_velocity),
div_phi_velocity(other.div_phi_velocity),
phi_pressure(other.phi_pressure),
grad_phi_pressure(other.grad_phi_pressure),
present_velocity_values(other.present_velocity_values),
present_velocity_gradients(other.present_velocity_gradients),
sym_grad_phi_velocity(other.sym_grad_phi_velocity),
laplace_phi_velocity(other.laplace_phi_velocity),
grad_div_phi_velocity(other.grad_div_phi_velocity),
present_strong_residuals(other.present_strong_residuals)
{}

template class ScratchData<2>;
template class ScratchData<3>;

} // namespace Matrix

namespace RightHandSide
{

template <int dim>
ScratchData<dim>::ScratchData
(const Mapping<dim>        &mapping,
 const FiniteElement<dim>  &fe,
 const Quadrature<dim>     &quadrature,
 const UpdateFlags         &update_flags,
 const Quadrature<dim-1>   &face_quadrature,
 const UpdateFlags         &face_update_flags,
 const StabilizationFlags  &stabilization_flags,
 const bool                 use_stress_form,
 const bool                 allocate_background_velocity,
 const bool                 allocate_body_force,
 const bool                 allocate_traction)
:
MeshWorker::ScratchData<dim>(mapping,
                             fe,
                             quadrature,
                             update_flags,
                             face_quadrature,
                             face_update_flags),
stabilization_flags(stabilization_flags),
vector_options(stabilization_flags,
               use_stress_form,
               allocate_background_velocity,
               allocate_body_force,
               allocate_traction,
               quadrature.size(),
               face_quadrature.size()),
scalar_options(use_stress_form),
phi_velocity(fe.n_dofs_per_cell()),
grad_phi_velocity(fe.n_dofs_per_cell()),
div_phi_velocity(fe.n_dofs_per_cell()),
phi_pressure(fe.n_dofs_per_cell()),
grad_phi_pressure(fe.n_dofs_per_cell()),
present_velocity_values(quadrature.size()),
present_velocity_gradients(quadrature.size()),
sym_grad_phi_velocity(),
present_strong_residuals()
{
  if (use_stress_form)
    sym_grad_phi_velocity.resize(fe.n_dofs_per_cell());

  // stabilization related quantity
  if (stabilization_flags & (apply_supg|apply_pspg))
    present_strong_residuals.resize(quadrature.size());
}



template <int dim>
ScratchData<dim>::ScratchData
(const FiniteElement<dim>  &fe,
 const Quadrature<dim>     &quadrature,
 const UpdateFlags         &update_flags,
 const Quadrature<dim-1>   &face_quadrature,
 const UpdateFlags         &face_update_flags,
 const StabilizationFlags  &stabilization_flags,
 const bool                 use_stress_form,
 const bool                 allocate_background_velocity,
 const bool                 allocate_body_force,
 const bool                 allocate_traction)
:
ScratchData<dim>(fe.reference_cell()
                 .template get_default_linear_mapping<dim>(),
                 fe,
                 quadrature,
                 update_flags,
                 face_quadrature,
                 face_update_flags,
                 stabilization_flags,
                 use_stress_form,
                 allocate_background_velocity,
                 allocate_body_force,
                 allocate_traction)
{}

template <int dim>
ScratchData<dim>::ScratchData(const ScratchData<dim>  &other)
:
MeshWorker::ScratchData<dim>(other),
stabilization_flags(other.stabilization_flags),
vector_options(other.vector_options),
scalar_options(other.scalar_options),
phi_velocity(other.phi_velocity),
grad_phi_velocity(other.grad_phi_velocity),
div_phi_velocity(other.div_phi_velocity),
phi_pressure(other.phi_pressure),
grad_phi_pressure(other.grad_phi_pressure),
present_velocity_values(other.present_velocity_values),
present_velocity_gradients(other.present_velocity_gradients),
sym_grad_phi_velocity(other.sym_grad_phi_velocity),
present_strong_residuals(other.present_strong_residuals)
{}

template class ScratchData<2>;
template class ScratchData<3>;

} // namespace RightHandSide

} // namespace AssemblyData

} // namespace Hydrodynamic
