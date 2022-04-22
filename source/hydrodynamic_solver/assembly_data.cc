/*
 * assembly_data.cc
 *
 *  Created on: Sep 25, 2021
 *      Author: sg
 */

#include <hydrodynamic_solver.h>

namespace Hydrodynamic {

namespace LegacyAssemblyData {

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
 const bool                allocate_face_normal,
 const bool                allocate_face_stresses,
 const bool                allocate_traction)
:
AssemblyBaseData::Matrix::Scratch<dim>(mapping, quadrature_formula, fe, update_flags),
fe_face_values(mapping, fe, face_quadrature_formula, face_update_flags),
hydrodynamic_strong_form_options(stabilization_flags,
                                 use_stress_form,
                                 allocate_background_velocity,
                                 allocate_body_force,
                                 this->n_q_points),
hydrodynamic_weak_form_options(use_stress_form),
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
face_normal_vectors(),
present_pressure_face_values(),
present_velocity_face_gradients(),
present_velocity_sym_face_gradients(),
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

  if (allocate_face_normal)
    face_normal_vectors.resize(this->n_face_q_points);

  // solution face values
  if (allocate_face_stresses)
  {
    present_pressure_face_values.resize(this->n_face_q_points);

    if (use_stress_form)
      present_velocity_sym_face_gradients.resize(this->n_face_q_points);
    else
      present_velocity_face_gradients.resize(this->n_face_q_points);

    if (!allocate_traction)
      boundary_traction_values.resize(this->n_face_q_points);
  }

  // source term face values
  if (allocate_traction)
    boundary_traction_values.resize(this->n_face_q_points);
}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim>  &other)
:
AssemblyBaseData::Matrix::Scratch<dim>(other),
fe_face_values(other.fe_face_values.get_mapping(),
               other.fe_face_values.get_fe(),
               other.fe_face_values.get_quadrature(),
               other.fe_face_values.get_update_flags()),
hydrodynamic_strong_form_options(other.hydrodynamic_strong_form_options),
hydrodynamic_weak_form_options(other.hydrodynamic_weak_form_options),
n_face_q_points(other.n_face_q_points),
phi_velocity(other.phi_velocity),
grad_phi_velocity(other.grad_phi_velocity),
div_phi_velocity(other.div_phi_velocity),
phi_pressure(other.phi_pressure),
sym_grad_phi_velocity(other.sym_grad_phi_velocity),
grad_phi_pressure(other.grad_phi_pressure),
laplace_phi_velocity(other.laplace_phi_velocity),
grad_div_phi_velocity(other.grad_div_phi_velocity),
present_velocity_values(other.present_velocity_values),
present_velocity_gradients(other.present_velocity_gradients),
present_pressure_values(other.present_pressure_values),
present_sym_velocity_gradients(other.present_sym_velocity_gradients),
present_velocity_laplaceans(other.present_velocity_laplaceans),
present_pressure_gradients(other.present_pressure_gradients),
present_strong_residuals(other.present_strong_residuals),
face_normal_vectors(other.face_normal_vectors),
present_pressure_face_values(other.present_pressure_face_values),
present_velocity_face_gradients(other.present_velocity_face_gradients),
present_velocity_sym_face_gradients(other.present_velocity_sym_face_gradients),
boundary_traction_values(other.boundary_traction_values)
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
 const bool                allocate_face_normal,
 const bool                allocate_face_stresses,
 const bool                allocate_traction)
:
AssemblyBaseData::RightHandSide::Scratch<dim>(mapping, quadrature_formula, fe, update_flags),
fe_face_values(mapping, fe, face_quadrature_formula, face_update_flags),
hydrodynamic_strong_form_options(stabilization_flags,
                                 use_stress_form,
                                 allocate_background_velocity,
                                 allocate_body_force,
                                 this->n_q_points),
hydrodynamic_weak_form_options(use_stress_form),
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
face_normal_vectors(),
present_pressure_face_values(),
present_velocity_face_gradients(),
present_velocity_sym_face_gradients(),
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

  if (allocate_face_normal)
    face_normal_vectors.resize(this->n_face_q_points);

  // solution face values
  if (allocate_face_stresses)
  {
    present_pressure_face_values.resize(this->n_face_q_points);

    if (use_stress_form)
      present_velocity_sym_face_gradients.resize(this->n_face_q_points);
    else
      present_velocity_face_gradients.resize(this->n_face_q_points);

    if (!allocate_traction)
      boundary_traction_values.resize(this->n_face_q_points);
  }

  // source term face values
  if (allocate_traction)
    boundary_traction_values.resize(this->n_face_q_points);
}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim>  &other)
:
AssemblyBaseData::RightHandSide::Scratch<dim>(other),
fe_face_values(other.fe_face_values.get_mapping(),
               other.fe_face_values.get_fe(),
               other.fe_face_values.get_quadrature(),
               other.fe_face_values.get_update_flags()),
hydrodynamic_strong_form_options(other.hydrodynamic_strong_form_options),
hydrodynamic_weak_form_options(other.hydrodynamic_weak_form_options),
n_face_q_points(other.n_face_q_points),
phi_velocity(other.phi_velocity),
grad_phi_velocity(other.grad_phi_velocity),
div_phi_velocity(other.div_phi_velocity),
phi_pressure(other.phi_pressure),
sym_grad_phi_velocity(other.sym_grad_phi_velocity),
grad_phi_pressure(other.grad_phi_pressure),
present_velocity_values(other.present_velocity_values),
present_velocity_gradients(other.present_velocity_gradients),
present_pressure_values(other.present_pressure_values),
present_sym_velocity_gradients(other.present_sym_velocity_gradients),
present_velocity_laplaceans(other.present_velocity_laplaceans),
present_pressure_gradients(other.present_pressure_gradients),
present_strong_residuals(other.present_strong_residuals),
face_normal_vectors(other.face_normal_vectors),
present_pressure_face_values(other.present_pressure_face_values),
present_velocity_face_gradients(other.present_velocity_face_gradients),
present_velocity_sym_face_gradients(other.present_velocity_sym_face_gradients),
boundary_traction_values(other.boundary_traction_values)
{}

template struct Scratch<2>;
template struct Scratch<3>;

} // namespace RightHandSide

} // namespace LegacyAssemblyData

} // namespace Hydrodynamic
