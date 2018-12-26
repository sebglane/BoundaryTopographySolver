/*
 * assembly_data.cc
 *
 *  Created on: Dec 24, 2018
 *      Author: sg
 */

#include "assembly_data.h"

namespace Assembly {



template<int dim>
Scratch<dim>::Scratch(const FiniteElement<dim>  &finite_element,
                      const Quadrature<dim>     &quadrature,
                      const Quadrature<dim-1>   &face_quadrature,
                      const UpdateFlags          update_flags,
                      const UpdateFlags          face_update_flags)
:
fe_values(finite_element,
          quadrature,
          update_flags),
fe_face_values(finite_element,
               face_quadrature,
               face_update_flags),
phi_density(finite_element.dofs_per_cell),
grad_phi_density(finite_element.dofs_per_cell),
present_density_values(quadrature.size()),
present_face_density_values(face_quadrature.size()),
present_density_gradients(quadrature.size()),
div_phi_velocity(finite_element.dofs_per_cell),
phi_velocity(finite_element.dofs_per_cell),
grad_phi_velocity(finite_element.dofs_per_cell),
phi_pressure(finite_element.dofs_per_cell),
present_velocity_divergences(quadrature.size()),
present_pressure_values(quadrature.size()),
present_velocity_values(quadrature.size()),
present_face_velocity_values(face_quadrature.size()),
present_velocity_gradients(quadrature.size())
{}


template<int dim>
Scratch<dim>::Scratch(const Scratch<dim>  &scratch)
:
fe_values(scratch.fe_values.get_mapping(),
          scratch.fe_values.get_fe(),
          scratch.fe_values.get_quadrature(),
          scratch.fe_values.get_update_flags()),
fe_face_values(scratch.fe_face_values.get_mapping(),
               scratch.fe_face_values.get_fe(),
               scratch.fe_face_values.get_quadrature(),
               scratch.fe_face_values.get_update_flags()),
phi_density(scratch.phi_density),
grad_phi_density(scratch.grad_phi_density),
present_density_values(scratch.present_density_values),
present_face_density_values(scratch.present_face_density_values),
present_density_gradients(scratch.present_density_gradients),
div_phi_velocity(scratch.div_phi_velocity),
phi_velocity(scratch.phi_velocity),
grad_phi_velocity(scratch.grad_phi_velocity),
phi_pressure(scratch.phi_pressure),
present_velocity_divergences(scratch.present_velocity_divergences),
present_pressure_values(scratch.present_pressure_values),
present_velocity_values(scratch.present_velocity_values),
present_face_velocity_values(scratch.present_face_velocity_values),
present_velocity_gradients(scratch.present_velocity_gradients)
{}

template<int dim>
CopyData<dim>::CopyData(const FiniteElement<dim>    &finite_element)
:
local_matrix(finite_element.dofs_per_cell,
             finite_element.dofs_per_cell),
local_rhs(finite_element.dofs_per_cell),
local_dof_indices(finite_element.dofs_per_cell)
{}


template<int dim>
CopyData<dim>::CopyData(const CopyData<dim>   &data)
:
local_matrix(data.local_matrix),
local_rhs(data.local_rhs),
local_dof_indices(data.local_dof_indices)
{}

}  // namespace Assembly


// explicit instantiation
template struct Assembly::Scratch<2>;
template struct Assembly::Scratch<3>;

template struct Assembly::CopyData<2>;
template struct Assembly::CopyData<3>;
