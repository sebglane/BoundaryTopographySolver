/*
 * buoyant_hydrodynamic_assembly_data.cc
 *
 *  Created on: Apr 12, 2022
 *      Author: sg
 */

#include <buoyant_hydrodynamic_assembly_data.h>

namespace BuoyantHydrodynamic {

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
 const bool                allocate_gravity_field,
 const bool                allocate_traction,
 const bool                allocate_density_source_term,
 const bool                allocate_density_boundary_values,
 const bool                allocate_reference_gradient)
:
dealii::MeshWorker::ScratchData<dim>(mapping,
                                     fe,
                                     quadrature,
                                     update_flags,
                                     face_quadrature,
                                     face_update_flags),
Hydrodynamic::AssemblyData::Matrix::ScratchData<dim>(mapping,
                                                     fe,
                                                     quadrature,
                                                     update_flags,
                                                     face_quadrature,
                                                     face_update_flags,
                                                     stabilization_flags,
                                                     use_stress_form,
                                                     allocate_background_velocity,
                                                     allocate_body_force,
                                                     allocate_traction),
Advection::AssemblyData::ScratchData<dim>(mapping,
                                          fe,
                                          quadrature,
                                          update_flags,
                                          face_quadrature,
                                          face_update_flags,
                                          allocate_density_source_term,
                                          allocate_density_boundary_values,
                                          false,
                                          allocate_reference_gradient),
vector_options(quadrature.size(),
               allocate_gravity_field)
{}



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
 const bool                allocate_gravity_field,
 const bool                allocate_traction,
 const bool                allocate_density_source_term,
 const bool                allocate_density_boundary_values,
 const bool                allocate_reference_gradient)
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
                 allocate_gravity_field,
                 allocate_traction,
                 allocate_density_source_term,
                 allocate_density_boundary_values,
                 allocate_reference_gradient)
{}




template <int dim>
ScratchData<dim>::ScratchData(const ScratchData<dim> &other)
:
dealii::MeshWorker::ScratchData<dim>(other),
Hydrodynamic::AssemblyData::Matrix::ScratchData<dim>(other),
Advection::AssemblyData::ScratchData<dim>(other),
vector_options(other.vector_options),
scalar_options(other.scalar_options)
{}


template class ScratchData<2>;
template class ScratchData<3>;

}  // namespace Matrix

namespace RightHandSide {

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
 const bool                allocate_gravity_field,
 const bool                allocate_traction,
 const bool                allocate_density_source_term,
 const bool                allocate_density_boundary_values,
 const bool                allocate_reference_gradient)
:
dealii::MeshWorker::ScratchData<dim>(mapping,
                                     fe,
                                     quadrature,
                                     update_flags,
                                     face_quadrature,
                                     face_update_flags),
Hydrodynamic::AssemblyData::RightHandSide::ScratchData<dim>(mapping,
                                                            fe,
                                                            quadrature,
                                                            update_flags,
                                                            face_quadrature,
                                                            face_update_flags,
                                                            stabilization_flags,
                                                            use_stress_form,
                                                            allocate_background_velocity,
                                                            allocate_body_force,
                                                            allocate_traction),
Advection::AssemblyData::ScratchData<dim>(mapping,
                                          fe,
                                          quadrature,
                                          update_flags,
                                          face_quadrature,
                                          face_update_flags,
                                          allocate_density_source_term,
                                          allocate_density_boundary_values,
                                          false,
                                          allocate_reference_gradient),
vector_options(quadrature.size(),
               allocate_gravity_field),
scalar_options()
{}



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
 const bool                allocate_gravity_field,
 const bool                allocate_traction,
 const bool                allocate_density_source_term,
 const bool                allocate_density_boundary_values,
 const bool                allocate_reference_gradient)
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
                 allocate_gravity_field,
                 allocate_traction,
                 allocate_density_source_term,
                 allocate_density_boundary_values,
                 allocate_reference_gradient)
{}



template <int dim>
ScratchData<dim>::ScratchData(const ScratchData<dim> &other)
:
dealii::MeshWorker::ScratchData<dim>(other),
Hydrodynamic::AssemblyData::RightHandSide::ScratchData<dim>(other),
Advection::AssemblyData::ScratchData<dim>(other),
vector_options(other.vector_options),
scalar_options(other.scalar_options)
{}

template class ScratchData<2>;
template class ScratchData<3>;

}  // namespace RightHandSide

}  // namespace AssemblyData

}  // namespace Buoyanthydrodynamic


