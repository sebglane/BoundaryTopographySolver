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
 const bool                allocate_body_force,
 const bool                allocate_traction,
 const bool                allocate_background_velocity)
:
AssemblyBaseData::Matrix::Scratch<dim>(mapping, quadrature_formula, fe, update_flags),
fe_face_values(mapping, fe, face_quadrature_formula, face_update_flags),
n_face_q_points(face_quadrature_formula.size()),
phi_velocity(this->dofs_per_cell),
grad_phi_velocity(this->dofs_per_cell),
div_phi_velocity(this->dofs_per_cell),
phi_pressure(this->dofs_per_cell),
present_velocity_values(this->n_q_points),
present_velocity_gradients(this->n_q_points),
present_pressure_values(this->n_q_points)
{
  // stabilization related shape functions
  if (stabilization_flags & (apply_supg|apply_pspg))
    grad_phi_pressure.resize(this->dofs_per_cell);
  if (stabilization_flags & (apply_supg|apply_pspg))
    laplace_phi_velocity.resize(this->dofs_per_cell);

  // stabilization related solution values
  if (stabilization_flags & (apply_supg|apply_pspg))
  {
    present_velocity_laplaceans.resize(this->n_q_points);
    present_pressure_gradients.resize(this->n_q_points);
  }

  // solution values
  if (allocate_background_velocity)
  {
    background_velocity_values.resize(this->n_q_points);
    background_velocity_gradients.resize(this->n_q_points);
  }

  // source term values
  if (allocate_body_force)
    body_force_values.resize(this->n_q_points);

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
n_face_q_points(data.n_face_q_points),
phi_velocity(data.phi_velocity),
grad_phi_velocity(data.grad_phi_velocity),
div_phi_velocity(data.div_phi_velocity),
phi_pressure(data.phi_pressure),
present_velocity_values(data.present_velocity_values),
present_velocity_gradients(data.present_velocity_gradients),
present_pressure_values(data.present_pressure_values)
{
  // stabilization related shape functions
  if (data.grad_phi_pressure.size() > 0)
    grad_phi_pressure.resize(data.grad_phi_pressure.size());
  if (data.laplace_phi_velocity.size() > 0)
    laplace_phi_velocity.resize(data.laplace_phi_velocity.size());

  // stabilization related solution values
  if (data.present_velocity_laplaceans.size() > 0)
    present_velocity_laplaceans.resize(data.present_velocity_laplaceans.size());
  if (data.present_pressure_gradients.size() > 0)
    present_pressure_gradients.resize(data.present_pressure_gradients.size());

  // solution values
  if (data.background_velocity_values.size() > 0)
    background_velocity_values.resize(data.background_velocity_values.size());
  if (data.background_velocity_gradients.size() > 0)
    background_velocity_gradients.resize(data.background_velocity_gradients.size());

  // source term values
  if (data.body_force_values.size() > 0)
    body_force_values.resize(data.body_force_values.size());

  // source term face values
  if (data.boundary_traction_values.size() > 0)
    boundary_traction_values.resize(data.boundary_traction_values.size());
}

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
 const bool                allocate_body_force,
 const bool                allocate_traction,
 const bool                allocate_background_velocity)
:
AssemblyBaseData::RightHandSide::Scratch<dim>(mapping, quadrature_formula, fe, update_flags),
fe_face_values(mapping, fe, face_quadrature_formula, face_update_flags),
n_face_q_points(face_quadrature_formula.size()),
phi_velocity(this->dofs_per_cell),
grad_phi_velocity(this->dofs_per_cell),
div_phi_velocity(this->dofs_per_cell),
phi_pressure(this->dofs_per_cell),
present_velocity_values(this->n_q_points),
present_velocity_gradients(this->n_q_points),
present_pressure_values(this->n_q_points)
{
  // stabilization related shape functions
  if (stabilization_flags & (apply_supg|apply_pspg))
    grad_phi_pressure.resize(this->dofs_per_cell);

  // stabilization related solution values
  if (stabilization_flags & (apply_supg|apply_pspg))
  {
    present_velocity_laplaceans.resize(this->n_q_points);
    present_pressure_gradients.resize(this->n_q_points);
  }

  // solution values
  if (allocate_background_velocity)
  {
    background_velocity_values.resize(this->n_q_points);
    background_velocity_gradients.resize(this->n_q_points);
  }

  // source term values
  if (allocate_body_force)
    body_force_values.resize(this->n_q_points);

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
n_face_q_points(data.n_face_q_points),
phi_velocity(data.phi_velocity),
grad_phi_velocity(data.grad_phi_velocity),
div_phi_velocity(data.div_phi_velocity),
phi_pressure(data.phi_pressure),
present_velocity_values(data.present_velocity_values),
present_velocity_gradients(data.present_velocity_gradients),
present_pressure_values(data.present_pressure_values)
{
  // stabilization related shape functions
  if (data.grad_phi_pressure.size() > 0)
    grad_phi_pressure.resize(data.grad_phi_pressure.size());

  // stabilization related solution values
  if (data.present_velocity_laplaceans.size() > 0)
    present_velocity_laplaceans.resize(data.present_velocity_laplaceans.size());
  if (data.present_pressure_gradients.size() > 0)
    present_pressure_gradients.resize(data.present_pressure_gradients.size());

  // solution values
  if (data.background_velocity_values.size() > 0)
    background_velocity_values.resize(data.background_velocity_values.size());
  if (data.background_velocity_gradients.size() > 0)
    background_velocity_gradients.resize(data.background_velocity_gradients.size());

  // source term values
  if (data.body_force_values.size() > 0)
    body_force_values.resize(data.body_force_values.size());

  // source term face values
  if (data.boundary_traction_values.size() > 0)
    boundary_traction_values.resize(data.boundary_traction_values.size());
}

template struct Scratch<2>;
template struct Scratch<3>;

} // namespace RightHandSide

} // namespace AssemblyData

} // namespace Hydrodynamic