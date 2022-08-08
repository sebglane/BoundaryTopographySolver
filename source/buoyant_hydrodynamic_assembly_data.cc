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



template <int dim>
void ScratchData<dim>::
assign_vector_options_local_cell
(const std::string                                          &name,
 const FEValuesExtractors::Vector                           &velocity,
 const FEValuesExtractors::Scalar                           &pressure,
 const std::shared_ptr<const Utility::AngularVelocity<dim>> &angular_velocity_ptr,
 const std::shared_ptr<const TensorFunction<1,dim>>         &body_force_ptr,
 const std::shared_ptr<const TensorFunction<1,dim>>         &gravity_field_ptr,
 const std::shared_ptr<const TensorFunction<1,dim>>         &background_velocity_ptr,
 const std::shared_ptr<const Function<dim>>                 &reference_field_ptr,
 const double                                                rossby_number,
 const double                                                froude_number,
 const double                                                stratification_number)
{
  Hydrodynamic::AssemblyData::Matrix::
  ScratchData<dim>::assign_vector_options_local_cell(name,
                                                     velocity,
                                                     pressure,
                                                     angular_velocity_ptr,
                                                     body_force_ptr,
                                                     background_velocity_ptr,
                                                     rossby_number,
                                                     froude_number);

  Advection::AssemblyData::Matrix::
  ScratchData<dim>::assign_vector_options_local_cell(nullptr,
                                                     nullptr,
                                                     reference_field_ptr,
                                                     stratification_number);

  this->adjust_velocity_field_local_cell();
  this->advection_field_values = this->present_velocity_values;

  // gravity field
  if (gravity_field_ptr != nullptr)
  {
    Assert(froude_number > 0.0,
           ExcLowerRangeType<double>(froude_number, 0.0));

    Assert(this->vector_options.gravity_field_values,
           ExcMessage("Gravity field values are not allocated in options."));
    AssertDimension(this->vector_options.gravity_field_values->size(),
                    this->get_current_fe_values().n_quadrature_points);

    gravity_field_ptr->value_list(this->get_quadrature_points(),
                                  *vector_options.gravity_field_values);

    this->
    Hydrodynamic::AssemblyData::Matrix::
    ScratchData<dim>::vector_options.froude_number = froude_number;
  }
}



// explicit instantiations
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



template <int dim>
void ScratchData<dim>::
assign_vector_options_local_cell
(const std::string                                          &name,
 const FEValuesExtractors::Vector                           &velocity,
 const FEValuesExtractors::Scalar                           &pressure,
 const std::shared_ptr<const Utility::AngularVelocity<dim>> &angular_velocity_ptr,
 const std::shared_ptr<const TensorFunction<1,dim>>         &body_force_ptr,
 const std::shared_ptr<const TensorFunction<1,dim>>         &gravity_field_ptr,
 const std::shared_ptr<const TensorFunction<1,dim>>         &background_velocity_ptr,
 const std::shared_ptr<const Function<dim>>                 &reference_field_ptr,
 const double                                                rossby_number,
 const double                                                froude_number,
 const double                                                stratification_number)
{
  Hydrodynamic::AssemblyData::RightHandSide::
  ScratchData<dim>::assign_vector_options_local_cell(name,
                                                     velocity,
                                                     pressure,
                                                     angular_velocity_ptr,
                                                     body_force_ptr,
                                                     background_velocity_ptr,
                                                     rossby_number,
                                                     froude_number);

  Advection::AssemblyData::RightHandSide::
  ScratchData<dim>::assign_vector_options_local_cell(nullptr,
                                                     nullptr,
                                                     reference_field_ptr,
                                                     stratification_number);

  this->adjust_velocity_field_local_cell();
  this->advection_field_values = this->present_velocity_values;

  // gravity field
  if (gravity_field_ptr != nullptr)
  {
    Assert(froude_number > 0.0,
           ExcLowerRangeType<double>(froude_number, 0.0));

    Assert(this->vector_options.gravity_field_values,
           ExcMessage("Gravity field values are not allocated in options."));
    AssertDimension(this->vector_options.gravity_field_values->size(),
                    this->get_current_fe_values().n_quadrature_points);


    gravity_field_ptr->value_list(this->get_quadrature_points(),
                                  *vector_options.gravity_field_values);

    this->
    Hydrodynamic::AssemblyData::RightHandSide::
    ScratchData<dim>::vector_options.froude_number
    = froude_number;
  }
}



// explicit instantiations
template class ScratchData<2>;
template class ScratchData<3>;

}  // namespace RightHandSide

}  // namespace AssemblyData

}  // namespace Buoyanthydrodynamic


