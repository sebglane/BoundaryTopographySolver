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
 const bool                allocate_body_force,
 const bool                allocate_traction,
 const bool                allocate_density_bc)
:
Hydrodynamic::AssemblyData::Matrix::Scratch<dim>(mapping,
                                                 quadrature_formula,
                                                 fe,
                                                 update_flags,
                                                 face_quadrature_formula,
                                                 face_update_flags,
                                                 stabilization_flags,
                                                 allocate_body_force,
                                                 allocate_traction),
phi_density(this->dofs_per_cell),
grad_phi_density(this->dofs_per_cell),
present_density_values(this->n_q_points),
present_density_gradients(this->n_q_points),
reference_density_gradients(this->n_q_points),
gravity_field_values(this->n_q_points)
{
  if (allocate_density_bc)
  {
    // solution face values
    present_density_face_values.resize(this->n_face_q_points);
    face_normal_vectors.resize(this->n_face_q_points);
    present_velocity_face_values.resize(this->n_face_q_points);

    // source term face values
    density_boundary_values.resize(this->n_face_q_points);
  }
}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim>  &data)
:
Hydrodynamic::AssemblyData::Matrix::Scratch<dim>(data),
phi_density(data.phi_density),
grad_phi_density(data.grad_phi_density),
present_density_values(data.present_density_values),
present_density_gradients(data.present_density_gradients),
reference_density_gradients(data.reference_density_gradients),
gravity_field_values(data.gravity_field_values)
{
  // solution face values
  if (data.present_density_face_values.size() > 0)
    present_density_face_values.resize(data.present_density_face_values.size());
  if (data.face_normal_vectors.size() > 0)
    face_normal_vectors.resize(data.face_normal_vectors.size());
  if (data.present_velocity_face_values.size() > 0)
    present_velocity_face_values.resize(data.present_velocity_face_values.size());

  // source term face values
  if (data.density_boundary_values.size() > 0)
    density_boundary_values.resize(data.density_boundary_values.size());
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
 const bool                allocate_density_bc)
:
Hydrodynamic::AssemblyData::RightHandSide::Scratch<dim>(mapping,
                                                        quadrature_formula,
                                                        fe,
                                                        update_flags,
                                                        face_quadrature_formula,
                                                        face_update_flags,
                                                        stabilization_flags,
                                                        allocate_body_force,
                                                        allocate_traction),
phi_density(this->dofs_per_cell),
grad_phi_density(this->dofs_per_cell),
present_density_values(this->n_q_points),
present_density_gradients(this->n_q_points),
reference_density_gradients(this->n_q_points),
gravity_field_values(this->n_q_points)
{
  if (allocate_density_bc)
  {
    // solution face values
    present_density_face_values.resize(this->n_face_q_points);
    face_normal_vectors.resize(this->n_face_q_points);
    present_velocity_face_values.resize(this->n_face_q_points);

    // source term face values
    density_boundary_values.resize(this->n_face_q_points);
  }
}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim>  &data)
:
Hydrodynamic::AssemblyData::RightHandSide::Scratch<dim>(data),
phi_density(data.phi_density),
grad_phi_density(data.grad_phi_density),
present_density_values(data.present_density_values),
present_density_gradients(data.present_density_gradients),
reference_density_gradients(data.reference_density_gradients),
gravity_field_values(data.gravity_field_values)
{
  // solution face values
  if (data.present_density_face_values.size() > 0)
    present_density_face_values.resize(data.present_density_face_values.size());
  if (data.face_normal_vectors.size() > 0)
    face_normal_vectors.resize(data.face_normal_vectors.size());
  if (data.face_normal_vectors.size() > 0)
    present_velocity_face_values.resize(data.face_normal_vectors.size());

  // source term face values
  if (data.density_boundary_values.size() > 0)
    density_boundary_values.resize(data.density_boundary_values.size());
}

template struct Scratch<2>;
template struct Scratch<3>;

} // namespace RightHandSide

} // namespace AssemblyData

} // namespace BuoyantHydrodynamic
