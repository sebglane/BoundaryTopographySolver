/*
 * advection_assembly_data.cc
 *
 *  Created on: Sep 25, 2021
 *      Author: sg
 */

#include <advection_assembly_data.h>

namespace Advection {

namespace AssemblyData {

template <int dim>
ScratchData<dim>::ScratchData
(const Mapping<dim>       &mapping,
 const FiniteElement<dim> &fe,
 const Quadrature<dim>    &quadrature,
 const UpdateFlags        &update_flags,
 const Quadrature<dim-1>  &face_quadrature,
 const UpdateFlags        &face_update_flags,
 const bool                allocate_source_term,
 const bool                allocate_boundary_values,
 const bool                allocate_background_advection,
 const bool                allocate_reference_gradient)
:
MeshWorker::ScratchData<dim>(mapping,
                             fe,
                             quadrature,
                             update_flags,
                             face_quadrature,
                             face_update_flags),
scalar_options(),
vector_options(quadrature.size(),
               face_quadrature.size(),
               allocate_source_term,
               allocate_boundary_values,
               allocate_background_advection,
               allocate_reference_gradient),
phi(fe.n_dofs_per_cell()),
grad_phi(fe.n_dofs_per_cell()),
advection_field_values(quadrature.size())
{}



template <int dim>
ScratchData<dim>::ScratchData
(const FiniteElement<dim> &fe,
 const Quadrature<dim>    &quadrature,
 const UpdateFlags        &update_flags,
 const Quadrature<dim-1>  &face_quadrature,
 const UpdateFlags        &face_update_flags,
 const bool                allocate_source_term,
 const bool                allocate_boundary_values,
 const bool                allocate_background_advection,
 const bool                allocate_reference_gradient)
:
ScratchData<dim>(fe.reference_cell()
                 .template get_default_linear_mapping<dim>(),
                 fe,
                 quadrature,
                 update_flags,
                 face_quadrature,
                 face_update_flags,
                 allocate_source_term,
                 allocate_boundary_values,
                 allocate_background_advection,
                 allocate_reference_gradient)
{}

template <int dim>
ScratchData<dim>::ScratchData(const ScratchData<dim>  &other)
:
MeshWorker::ScratchData<dim>(other),
scalar_options(other.scalar_options),
vector_options(other.vector_options),
phi(other.phi),
grad_phi(other.grad_phi),
advection_field_values(other.advection_field_values)
{}

template class ScratchData<2>;
template class ScratchData<3>;

} // namespace AssemblyData

} // namespace Advection
