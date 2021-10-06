/*
 * assembly_data.cc
 *
 *  Created on: Sep 25, 2021
 *      Author: sg
 */

#include <hydrodynamic_solver.h>

namespace Hydrodynamic {

namespace AssemblyData {

namespace Matrix {

template <int dim>
Scratch<dim>::Scratch
(const Mapping<dim>       &mapping,
 const Quadrature<dim>    &quadrature_formula,
 const FiniteElement<dim> &fe,
 const UpdateFlags         update_flags,
 const Quadrature<dim-1>  &face_quadrature_formula,
 const UpdateFlags         face_update_flags,
 const StabilizationFlags  stabilization_flags,
 const bool                use_stress_form,
 const bool                allocate_background_velocity,
 const bool                allocate_body_force,
 const bool                allocate_traction)
:
AssemblyBaseData::Matrix::Scratch<dim>(mapping, quadrature_formula, fe, update_flags),
fe_face_values(mapping, fe, face_quadrature_formula, face_update_flags),
optional_arguments_strong_from(stabilization_flags,
                               use_stress_form,
                               allocate_background_velocity,
                               allocate_body_force,
                               this->n_q_points),
optional_arguments_weak_from(use_stress_form),
n_face_q_points(face_quadrature_formula.size()),
phi_velocity(this->dofs_per_cell),
grad_phi_velocity(this->dofs_per_cell),
div_phi_velocity(this->dofs_per_cell),
phi_pressure(this->dofs_per_cell),
sym_grad_phi_velocity(),
grad_phi_pressure(),
laplace_phi_velocity(),
grad_div_phi_velocity(),
present_velocity_values(this->n_q_points),
present_velocity_gradients(this->n_q_points),
present_pressure_values(this->n_q_points),
present_sym_velocity_gradients(),
present_velocity_laplaceans(),
present_pressure_gradients(),
present_strong_residuals(),
boundary_traction_values()
{
  if (use_stress_form)
  {
    sym_grad_phi_velocity.resize(this->dofs_per_cell);
    present_sym_velocity_gradients.resize(this->n_q_points);
  }

  // stabilization related shape functions
  if (stabilization_flags & (apply_supg|apply_pspg))
    grad_phi_pressure.resize(this->dofs_per_cell);
  if (stabilization_flags & (apply_supg|apply_pspg))
  {
    laplace_phi_velocity.resize(this->dofs_per_cell);
    if (use_stress_form)
      grad_div_phi_velocity.resize(this->dofs_per_cell);
  }

  // stabilization related solution values
  if (stabilization_flags & (apply_supg|apply_pspg))
  {
    present_velocity_laplaceans.resize(this->n_q_points);
    present_pressure_gradients.resize(this->n_q_points);

    present_strong_residuals.resize(this->n_q_points);
  }

  // source term face values
  if (allocate_traction)
    boundary_traction_values.resize(this->n_face_q_points);
}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim>  &data)
:
AssemblyBaseData::Matrix::Scratch<dim>(data),
fe_face_values(data.fe_face_values.get_mapping(),
               data.fe_face_values.get_fe(),
               data.fe_face_values.get_quadrature(),
               data.fe_face_values.get_update_flags()),
optional_arguments_strong_from(data.optional_arguments_strong_from),
optional_arguments_weak_from(data.optional_arguments_weak_from),
n_face_q_points(data.n_face_q_points),
phi_velocity(data.phi_velocity),
grad_phi_velocity(data.grad_phi_velocity),
div_phi_velocity(data.div_phi_velocity),
phi_pressure(data.phi_pressure),
sym_grad_phi_velocity(data.sym_grad_phi_velocity),
grad_phi_pressure(data.grad_phi_pressure),
laplace_phi_velocity(data.laplace_phi_velocity),
grad_div_phi_velocity(data.grad_div_phi_velocity),
present_velocity_values(data.present_velocity_values),
present_velocity_gradients(data.present_velocity_gradients),
present_pressure_values(data.present_pressure_values),
present_sym_velocity_gradients(data.present_sym_velocity_gradients),
present_velocity_laplaceans(data.present_velocity_laplaceans),
present_pressure_gradients(data.present_pressure_gradients),
present_strong_residuals(data.present_strong_residuals),
boundary_traction_values(data.boundary_traction_values)
{}

template struct Scratch<2>;
template struct Scratch<3>;

} // namespace Matrix

namespace RightHandSide
{

template <int dim>
Scratch<dim>::Scratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const FiniteElement<dim>  &fe,
 const UpdateFlags         update_flags,
 const Quadrature<dim-1>  &face_quadrature_formula,
 const UpdateFlags         face_update_flags,
 const StabilizationFlags  stabilization_flags,
 const bool                use_stress_form,
 const bool                allocate_background_velocity,
 const bool                allocate_body_force,
 const bool                allocate_traction)
:
AssemblyBaseData::RightHandSide::Scratch<dim>(mapping, quadrature_formula, fe, update_flags),
fe_face_values(mapping, fe, face_quadrature_formula, face_update_flags),
optional_arguments_strong_from(stabilization_flags,
                               use_stress_form,
                               allocate_background_velocity,
                               allocate_body_force,
                               this->n_q_points),
optional_arguments_weak_from(use_stress_form),
n_face_q_points(face_quadrature_formula.size()),
phi_velocity(this->dofs_per_cell),
grad_phi_velocity(this->dofs_per_cell),
div_phi_velocity(this->dofs_per_cell),
phi_pressure(this->dofs_per_cell),
sym_grad_phi_velocity(),
grad_phi_pressure(),
present_velocity_values(this->n_q_points),
present_velocity_gradients(this->n_q_points),
present_pressure_values(this->n_q_points),
present_sym_velocity_gradients(),
present_velocity_laplaceans(),
present_pressure_gradients(),
present_strong_residuals(),
boundary_traction_values()
{
  if (use_stress_form)
  {
    sym_grad_phi_velocity.resize(this->dofs_per_cell);
    present_sym_velocity_gradients.resize(this->n_q_points);
  }

  if (stabilization_flags & (apply_supg|apply_pspg))
  {
    // stabilization related shape functions
    grad_phi_pressure.resize(this->dofs_per_cell);

    // stabilization related solution values
    present_velocity_laplaceans.resize(this->n_q_points);
    present_pressure_gradients.resize(this->n_q_points);

    // stabilization related quantity
    present_strong_residuals.resize(this->n_q_points);
  }

  // source term face values
  if (allocate_traction)
    boundary_traction_values.resize(this->n_face_q_points);
}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim>  &data)
:
AssemblyBaseData::RightHandSide::Scratch<dim>(data),
fe_face_values(data.fe_face_values.get_mapping(),
               data.fe_face_values.get_fe(),
               data.fe_face_values.get_quadrature(),
               data.fe_face_values.get_update_flags()),
optional_arguments_strong_from(data.optional_arguments_strong_from),
optional_arguments_weak_from(data.optional_arguments_weak_from),
n_face_q_points(data.n_face_q_points),
phi_velocity(data.phi_velocity),
grad_phi_velocity(data.grad_phi_velocity),
div_phi_velocity(data.div_phi_velocity),
phi_pressure(data.phi_pressure),
sym_grad_phi_velocity(data.sym_grad_phi_velocity),
grad_phi_pressure(data.grad_phi_pressure),
present_velocity_values(data.present_velocity_values),
present_velocity_gradients(data.present_velocity_gradients),
present_pressure_values(data.present_pressure_values),
present_sym_velocity_gradients(data.present_sym_velocity_gradients),
present_velocity_laplaceans(data.present_velocity_laplaceans),
present_pressure_gradients(data.present_pressure_gradients),
present_strong_residuals(data.present_strong_residuals),
boundary_traction_values(data.boundary_traction_values)
{}

template struct Scratch<2>;
template struct Scratch<3>;

} // namespace RightHandSide

} // namespace AssemblyData

} // namespace Hydrodynamic
