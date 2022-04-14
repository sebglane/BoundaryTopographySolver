/*
 * buoyant_hydrodynamic_assembly_data.h
 *
 *  Created on: Apr 12, 2022
 *      Author: sg
 */

#ifndef INCLUDE_BUOYANT_HYDRODYNAMIC_ASSEMBLY_DATA_H_
#define INCLUDE_BUOYANT_HYDRODYNAMIC_ASSEMBLY_DATA_H_

#include <advection_assembly_data.h>
#include <buoyant_hydrodynamic_options.h>
#include <hydrodynamic_assembly_data.h>

namespace BuoyantHydrodynamic {

using namespace dealii;

namespace AssemblyData {

namespace Matrix {

template <int dim>
class ScratchData : public Hydrodynamic::AssemblyData::Matrix::ScratchData<dim>,
                    public Advection::AssemblyData::ScratchData<dim>
{
  ScratchData(
    const Mapping<dim>       &mapping,
    const FiniteElement<dim> &fe,
    const Quadrature<dim>    &quadrature,
    const UpdateFlags        &update_flags,
    const Quadrature<dim-1>  &face_quadrature = Quadrature<dim-1>(),
    const UpdateFlags        &face_update_flags = update_default,
    const bool                use_stress_form = false,
    const bool                allocate_background_velocity = false,
    const bool                allocate_body_force = false,
    const bool                allocate_gravity_field = false,
    const bool                allocate_traction = false,
    const bool                allocate_source_term = false,
    const bool                allocate_density_boundary_values = false,
    const bool                allocate_reference_gradient = false);

  ScratchData(
    const FiniteElement<dim> &fe,
    const Quadrature<dim>    &quadrature,
    const UpdateFlags        &update_flags,
    const Quadrature<dim-1>  &face_quadrature   = Quadrature<dim-1>(),
    const UpdateFlags        &face_update_flags = update_default,
    const bool                use_stress_form = false,
    const bool                allocate_background_velocity = false,
    const bool                allocate_body_force = false,
    const bool                allocate_gravity_field = false,
    const bool                allocate_traction = false,
    const bool                allocate_source_term = false,
    const bool                allocate_density_boundary_values = false,
    const bool                allocate_reference_gradient = false);

  ScratchData(const ScratchData<dim>  &data);

  OptionalScalarArguments<dim>  scalar_options;

  OptionalVectorArguments<dim>  vector_options;

};



}  // namespace Matrix

namespace RightHandSide {

template <int dim>
class ScratchData : public Hydrodynamic::AssemblyData::RightHandSide::ScratchData<dim>,
                    public Advection::AssemblyData::ScratchData<dim>
{
  ScratchData(
    const Mapping<dim>       &mapping,
    const FiniteElement<dim> &fe,
    const Quadrature<dim>    &quadrature,
    const UpdateFlags        &update_flags,
    const Quadrature<dim-1>  &face_quadrature = Quadrature<dim-1>(),
    const UpdateFlags        &face_update_flags = update_default,
    const bool                use_stress_form = false,
    const bool                allocate_background_velocity = false,
    const bool                allocate_body_force = false,
    const bool                allocate_gravity_field = false,
    const bool                allocate_traction = false,
    const bool                allocate_density_source_term = false,
    const bool                allocate_density_boundary_values = false,
    const bool                allocate_reference_gradient = false);

  ScratchData(
    const FiniteElement<dim> &fe,
    const Quadrature<dim>    &quadrature,
    const UpdateFlags        &update_flags,
    const Quadrature<dim-1>  &face_quadrature   = Quadrature<dim-1>(),
    const UpdateFlags        &face_update_flags = update_default,
    const bool                use_stress_form = false,
    const bool                allocate_background_velocity = false,
    const bool                allocate_body_force = false,
    const bool                allocate_gravity_field = false,
    const bool                allocate_traction = false,
    const bool                allocate_density_source_term = false,
    const bool                allocate_density_boundary_values = false,
    const bool                allocate_reference_gradient = false);

  ScratchData(const ScratchData<dim>  &data);

  OptionalScalarArguments<dim>  scalar_options;

  OptionalVectorArguments<dim>  vector_options;

};

}  // namespace RightHandSide

}  // namespace AssemblyData

}  // namespace BuoyantHydrodynamic



#endif /* INCLUDE_BUOYANT_HYDRODYNAMIC_ASSEMBLY_DATA_H_ */
