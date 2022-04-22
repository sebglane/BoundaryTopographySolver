/*
 * assembly_data.cc
 *
 *  Created on: Sep 26, 2021
 *      Author: sg
 */

#include <buoyant_hydrodynamic_solver.h>

namespace BuoyantHydrodynamic {

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
 const bool                allocate_face_stresses,
 const bool                allocate_traction,
 const bool                allocate_gravity_field,
 const bool                allocate_reference_density,
 const bool                allocate_density_bc)
:
Hydrodynamic::LegacyAssemblyData::Matrix::Scratch<dim>(mapping,
                                                 quadrature_formula,
                                                 fe,
                                                 update_flags,
                                                 face_quadrature_formula,
                                                 face_update_flags,
                                                 stabilization_flags,
                                                 use_stress_form,
                                                 allocate_background_velocity,
                                                 allocate_body_force,
                                                 allocate_density_bc||allocate_face_stresses,
                                                 allocate_face_stresses,
                                                 allocate_traction),
strong_form_options(allocate_gravity_field,
                    allocate_reference_density,
                    this->n_q_points),
weak_form_options(),
phi_density(this->dofs_per_cell),
grad_phi_density(this->dofs_per_cell),
present_density_values(this->n_q_points),
present_density_gradients(this->n_q_points),
present_strong_density_residuals(this->n_q_points),
present_density_face_values(),
present_velocity_face_values(),
density_boundary_values()
{
  if (allocate_density_bc)
  {
    // solution face values
    present_density_face_values.resize(this->n_face_q_points);
    present_velocity_face_values.resize(this->n_face_q_points);

    // source term face values
    density_boundary_values.resize(this->n_face_q_points);
  }
}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim>  &data)
:
Hydrodynamic::LegacyAssemblyData::Matrix::Scratch<dim>(data),
strong_form_options(data.strong_form_options),
weak_form_options(data.weak_form_options),
phi_density(data.phi_density),
grad_phi_density(data.grad_phi_density),
present_density_values(data.present_density_values),
present_density_gradients(data.present_density_gradients),
present_strong_density_residuals(data.present_strong_density_residuals),
present_density_face_values(data.present_density_face_values),
present_velocity_face_values(data.present_velocity_face_values),
density_boundary_values(data.density_boundary_values)
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
 const bool                allocate_face_stresses,
 const bool                allocate_traction,
 const bool                allocate_gravity_field,
 const bool                allocate_reference_density,
 const bool                allocate_density_bc)
:
Hydrodynamic::LegacyAssemblyData::RightHandSide::Scratch<dim>(mapping,
                                                        quadrature_formula,
                                                        fe,
                                                        update_flags,
                                                        face_quadrature_formula,
                                                        face_update_flags,
                                                        stabilization_flags,
                                                        use_stress_form,
                                                        allocate_background_velocity,
                                                        allocate_body_force,
                                                        allocate_density_bc||allocate_face_stresses,
                                                        allocate_face_stresses,
                                                        allocate_traction),
strong_form_options(allocate_gravity_field,
                    allocate_reference_density,
                    this->n_q_points),
weak_form_options(),
phi_density(this->dofs_per_cell),
grad_phi_density(this->dofs_per_cell),
present_density_values(this->n_q_points),
present_density_gradients(this->n_q_points),
present_strong_density_residuals(this->n_q_points),
present_density_face_values(),
present_velocity_face_values(),
density_boundary_values()
{
  if (allocate_density_bc)
  {
    // solution face values
    present_density_face_values.resize(this->n_face_q_points);
    present_velocity_face_values.resize(this->n_face_q_points);

    // source term face values
    density_boundary_values.resize(this->n_face_q_points);
  }
}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim>  &data)
:
Hydrodynamic::LegacyAssemblyData::RightHandSide::Scratch<dim>(data),
strong_form_options(data.strong_form_options),
weak_form_options(data.weak_form_options),
phi_density(data.phi_density),
grad_phi_density(data.grad_phi_density),
present_density_values(data.present_density_values),
present_density_gradients(data.present_density_gradients),
present_strong_density_residuals(data.present_strong_density_residuals),
present_density_face_values(data.present_density_face_values),
present_velocity_face_values(data.present_velocity_face_values),
density_boundary_values(data.density_boundary_values)
{}

template struct Scratch<2>;
template struct Scratch<3>;

} // namespace RightHandSide

} // namespace AssemblyData

} // namespace BuoyantHydrodynamic
