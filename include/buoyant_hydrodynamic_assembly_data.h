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

/**
 * @todo Add documentation.
 */
template <int dim>
class ScratchData : public Hydrodynamic::AssemblyData::ScratchData<dim>,
                    public Advection::AssemblyData::ScratchData<dim>
{
public:
  ScratchData(
    const Mapping<dim>       &mapping,
    const FiniteElement<dim> &fe,
    const Quadrature<dim>    &quadrature,
    const UpdateFlags        &update_flags,
    const Quadrature<dim-1>  &face_quadrature = Quadrature<dim-1>(),
    const UpdateFlags        &face_update_flags = update_default,
    const StabilizationFlags &stabilization_flags = apply_none,
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
    const StabilizationFlags &stabilization_flags = apply_none,
    const bool                use_stress_form = false,
    const bool                allocate_background_velocity = false,
    const bool                allocate_body_force = false,
    const bool                allocate_gravity_field = false,
    const bool                allocate_traction = false,
    const bool                allocate_source_term = false,
    const bool                allocate_density_boundary_values = false,
    const bool                allocate_reference_gradient = false);

  ScratchData(const ScratchData<dim>  &data);

  void assign_vector_options_local_cell(
    const std::string                                          &name,
    const FEValuesExtractors::Vector                           &velocity,
    const FEValuesExtractors::Scalar                           &pressure,
    const std::shared_ptr<const Utility::AngularVelocity<dim>> &angular_velocity_ptr = nullptr,
    const std::shared_ptr<const TensorFunction<1,dim>>         &body_force_ptr = nullptr,
    const std::shared_ptr<const TensorFunction<1,dim>>         &gravity_field_ptr = nullptr,
    const std::shared_ptr<const TensorFunction<1,dim>>         &background_velocity_ptr = nullptr,
    const std::shared_ptr<const Function<dim>>                 &source_term_ptr = nullptr,
    const std::shared_ptr<const Function<dim>>                 &reference_field_ptr = nullptr,
    const double                                                rossby_number = 0.0,
    const double                                                froude_number = 0.0,
    const double                                                gradient_scaling_number = 0.0);

  VectorOptions<dim>  vector_options;

};



namespace Matrix {

template <int dim>
using ScratchData = ScratchData<dim>;

}  // namespace Matrix



namespace RightHandSide {

template <int dim>
using ScratchData = ScratchData<dim>;

}  // namespace RightHandSide

}  // namespace AssemblyData

}  // namespace BuoyantHydrodynamic



#endif /* INCLUDE_BUOYANT_HYDRODYNAMIC_ASSEMBLY_DATA_H_ */
