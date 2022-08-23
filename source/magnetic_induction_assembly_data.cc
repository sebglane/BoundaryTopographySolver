/*
 * magnetic_induction_assembly_data.cc
 *
 *  Created on: Sep 25, 2021
 *      Author: sg
 */

#include <magnetic_induction_assembly_data.h>

namespace MagneticInduction {

namespace AssemblyData {

template <int dim>
ScratchData<dim>::ScratchData
(const Mapping<dim>       &mapping,
 const FiniteElement<dim> &fe,
 const Quadrature<dim>    &quadrature,
 const UpdateFlags        &update_flags,
 const bool                allocate_velocity_field,
 const bool                allocate_background_magnetic_field)
:
MeshWorker::ScratchData<dim>(mapping,
                             fe,
                             quadrature,
                             update_flags),
vector_options(quadrature.size(),
               allocate_velocity_field,
               allocate_background_magnetic_field),
phi_magnetic_field(fe.n_dofs_per_cell()),
curl_phi_magnetic_field(fe.n_dofs_per_cell()),
div_phi_magnetic_field(fe.n_dofs_per_cell()),
grad_phi_magnetic_pressure(fe.n_dofs_per_cell()),
present_magnetic_field_values(quadrature.size()),
present_magnetic_field_curls(quadrature.size()),
present_magnetic_field_divergences(quadrature.size()),
present_magnetic_pressure_gradients(quadrature.size())
{}



template <int dim>
ScratchData<dim>::ScratchData
(const FiniteElement<dim> &fe,
 const Quadrature<dim>    &quadrature,
 const UpdateFlags        &update_flags,
 const bool                allocate_velocity_field,
 const bool                allocate_background_magnetic_field)
:
ScratchData<dim>(fe.reference_cell()
                 .template get_default_linear_mapping<dim>(),
                 fe,
                 quadrature,
                 update_flags,
                 allocate_velocity_field,
                 allocate_background_magnetic_field)
{}

template <int dim>
ScratchData<dim>::ScratchData(const ScratchData<dim>  &other)
:
MeshWorker::ScratchData<dim>(other),
vector_options(other.vector_options),
phi_magnetic_field(other.phi_magnetic_field),
curl_phi_magnetic_field(other.curl_phi_magnetic_field),
div_phi_magnetic_field(other.div_phi_magnetic_field),
grad_phi_magnetic_pressure(other.grad_phi_magnetic_pressure),
present_magnetic_field_values(other.present_magnetic_field_values),
present_magnetic_field_curls(other.present_magnetic_field_curls),
present_magnetic_field_divergences(other.present_magnetic_field_divergences),
present_magnetic_pressure_gradients(other.present_magnetic_pressure_gradients)
{}



// explicit instantiations
template class ScratchData<2>;
template class ScratchData<3>;

} // namespace AssemblyData

} // namespace MagneticInduction
