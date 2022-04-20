/*
 * hydrodynamic_assembly_data.h
 *
 *  Created on: Apr 11, 2022
 *      Author: sg
 */

#ifndef INCLUDE_HYDRODYNAMIC_ASSEMBLY_DATA_H_
#define INCLUDE_HYDRODYNAMIC_ASSEMBLY_DATA_H_

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/meshworker/scratch_data.h>

#include <hydrodynamic_options.h>
#include <stabilization_flags.h>

#include <memory>

namespace Hydrodynamic {

using namespace dealii;

namespace AssemblyData {

namespace Matrix {

template <int dim>
class ScratchData : virtual public MeshWorker::ScratchData<dim>
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
    const bool                allocate_traction = false);

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
    const bool                allocate_traction = false);

  ScratchData(const ScratchData<dim>  &data);

  void assign_vector_options_local_cell(
    const std::string                                          &name,
    const FEValuesExtractors::Vector                           &velocity,
    const FEValuesExtractors::Scalar                           &pressure,
    const std::shared_ptr<const Utility::AngularVelocity<dim>> &angular_velocity_ptr = nullptr,
    const std::shared_ptr<const TensorFunction<1,dim>>         &body_force_ptr = nullptr,
    const std::shared_ptr<const TensorFunction<1,dim>>         &background_velocity_ptr = nullptr,
    const double                                                rossby_number = 0.0,
    const double                                                froude_number = 0.0);

  void assign_scalar_options_local_cell(const unsigned int q_point_index);

  const StabilizationFlags  &stabilization_flags;

  OptionalVectorArguments<dim>  vector_options;
  OptionalScalarArguments<dim>  scalar_options;

  // shape functions
  std::vector<Tensor<1, dim>> phi_velocity;
  std::vector<Tensor<2, dim>> grad_phi_velocity;
  std::vector<double>         div_phi_velocity;
  std::vector<double>         phi_pressure;

  // stabilization related shape functions
  std::vector<Tensor<1, dim>> grad_phi_pressure;

  // present solution values which cannot be referenced
  std::vector<Tensor<1, dim>> present_velocity_values;
  std::vector<Tensor<2, dim>> present_velocity_gradients;

  // stress tensor related shape functions
  std::vector<SymmetricTensor<2, dim>>  sym_grad_phi_velocity;

  // stabilization related shape functions
  std::vector<Tensor<1, dim>> laplace_phi_velocity;

  // stress tensor and stabilization related shape functions
  std::vector<Tensor<1, dim>> grad_div_phi_velocity;

  // stabilization related quantities
  std::vector<Tensor<1, dim>> present_strong_residuals;
};



template <int dim>
inline void ScratchData<dim>::
assign_vector_options_local_cell
(const std::string                                          &name,
 const FEValuesExtractors::Vector                           &velocity,
 const FEValuesExtractors::Scalar                           &pressure,
 const std::shared_ptr<const Utility::AngularVelocity<dim>> &angular_velocity_ptr,
 const std::shared_ptr<const TensorFunction<1,dim>>         &body_force_ptr,
 const std::shared_ptr<const TensorFunction<1,dim>>         &background_velocity_ptr,
 const double                                                rossby_number,
 const double                                                froude_number)
{
  const unsigned int n_q_points{this->get_current_fe_values().n_quadrature_points};

  // stress form
  if (vector_options.use_stress_form)
  {
    Assert(vector_options.present_sym_velocity_gradients,
           ExcMessage("Symmetric velocity gradients are not allocated in options."));

    AssertDimension(vector_options.present_sym_velocity_gradients->size(),
                    n_q_points);

    vector_options.present_sym_velocity_gradients
      = this->get_symmetric_gradients(name,
                                      velocity);
  }

  // stabilization related solution values
  if (stabilization_flags & (apply_supg|apply_pspg))
  {
    Assert(vector_options.present_velocity_laplaceans,
           ExcMessage("Velocity laplacean are not allocated in options."));
    AssertDimension(vector_options.present_velocity_laplaceans->size(),
                    n_q_points);
    vector_options.present_velocity_laplaceans
      = this->get_laplacians(name, velocity);

    Assert(vector_options.present_pressure_gradients,
           ExcMessage("Present pressure gradients are not allocated in options."));
    AssertDimension(vector_options.present_pressure_gradients->size(),
                    n_q_points);
    vector_options.present_pressure_gradients
      = this->get_gradients(name, pressure);

    if (vector_options.use_stress_form)
    {
      Assert(vector_options.present_velocity_grad_divergences,
             ExcMessage("Gradients of velocity divergence are not allocated in options."));
      AssertDimension(vector_options.present_velocity_grad_divergences->size(),
                      n_q_points);

      const auto &present_hessians = this->get_hessians(name,
                                                        velocity);

      std::vector<Tensor<1, dim>> &present_velocity_grad_divergences =
          vector_options.present_velocity_grad_divergences.value();
      for (std::size_t q=0; q<present_hessians.size(); ++q)
      {
        present_velocity_grad_divergences[q] = 0;
        for (unsigned int d=0; d<dim; ++d)
          present_velocity_grad_divergences[q] += present_hessians[q][d][d];
      }
    }
  }

  // Coriolis term
  if (angular_velocity_ptr != nullptr)
  {
    Assert(rossby_number > 0.0,
           ExcLowerRangeType<double>(rossby_number, 0.0));

    vector_options.angular_velocity = angular_velocity_ptr->value();
    vector_options.rossby_number = rossby_number;

    scalar_options.angular_velocity = angular_velocity_ptr->value();
    scalar_options.rossby_number = rossby_number;
  }

  // body force
  if (body_force_ptr != nullptr)
  {
    Assert(froude_number > 0.0,
           ExcLowerRangeType<double>(froude_number, 0.0));

    Assert(vector_options.body_force_values,
           ExcMessage("Body force values are not allocated in options."));
    AssertDimension(vector_options.body_force_values->size(),
                    n_q_points);

    body_force_ptr->value_list(this->get_quadrature_points(),
                               *vector_options.body_force_values);
    vector_options.froude_number = froude_number;
    scalar_options.froude_number = froude_number;
  }

  // background field
  if (background_velocity_ptr != nullptr)
  {
    Assert(vector_options.background_velocity_values,
           ExcMessage("Background velocity values are not allocated in options."));
    AssertDimension(vector_options.background_velocity_values->size(),
                    n_q_points);
    background_velocity_ptr->value_list(this->get_quadrature_points(),
                                        *vector_options.background_velocity_values);

    Assert(vector_options.background_velocity_gradients,
           ExcMessage("Background velocity gradients are not allocated in options."));
    AssertDimension(vector_options.background_velocity_gradients->size(),
                    n_q_points);
    background_velocity_ptr->gradient_list(this->get_quadrature_points(),
                                           *vector_options.background_velocity_gradients);
  }
}




template <int dim>
inline void ScratchData<dim>::
assign_scalar_options_local_cell
(const unsigned int q)
{
  // stress form
  if (vector_options.use_stress_form)
  {
    Assert(vector_options.present_sym_velocity_gradients,
           ExcMessage("Symmetric velocity gradients are not allocated in options."));
    scalar_options.present_symmetric_velocity_gradient =
          vector_options.present_sym_velocity_gradients->at(q);
  }
  // body force
  if (vector_options.body_force_values)
    scalar_options.body_force_value =
        vector_options.body_force_values->at(q);
}




}  // namespace Matrix

namespace RightHandSide
{

template <int dim>
class ScratchData : virtual public MeshWorker::ScratchData<dim>
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
    const bool                allocate_traction = false);

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
    const bool                allocate_traction = false);

  ScratchData(const ScratchData<dim>  &data);

  void assign_vector_options_local_cell(
      const std::string                                          &name,
      const FEValuesExtractors::Vector                           &velocity,
      const FEValuesExtractors::Scalar                           &pressure,
      const std::shared_ptr<const Utility::AngularVelocity<dim>> &angular_velocity_ptr = nullptr,
      const std::shared_ptr<const TensorFunction<1,dim>>         &body_force_ptr = nullptr,
      const std::shared_ptr<const TensorFunction<1,dim>>         &background_velocity_ptr = nullptr,
      const double                                                rossby_number = 0.0,
      const double                                                froude_number = 0.0);


    void assign_scalar_options_local_cell(const unsigned int q_point_index);

  const StabilizationFlags  &stabilization_flags;

  OptionalVectorArguments<dim>  vector_options;
  OptionalScalarArguments<dim>  scalar_options;

  // shape functions
  std::vector<Tensor<1, dim>> phi_velocity;
  std::vector<Tensor<2, dim>> grad_phi_velocity;
  std::vector<double>         div_phi_velocity;
  std::vector<double>         phi_pressure;

  // stabilization related shape functions
  std::vector<Tensor<1, dim>> grad_phi_pressure;

  // present solution values which cannot be referenced
  std::vector<Tensor<1, dim>> present_velocity_values;
  std::vector<Tensor<2, dim>> present_velocity_gradients;

  // stress tensor related shape functions
  std::vector<SymmetricTensor<2, dim>>  sym_grad_phi_velocity;

  // stabilization related quantities
  std::vector<Tensor<1, dim>> present_strong_residuals;
};





template <int dim>
inline void ScratchData<dim>::
assign_vector_options_local_cell
(const std::string                                          &name,
 const FEValuesExtractors::Vector                           &velocity,
 const FEValuesExtractors::Scalar                           &pressure,
 const std::shared_ptr<const Utility::AngularVelocity<dim>> &angular_velocity_ptr,
 const std::shared_ptr<const TensorFunction<1,dim>>         &body_force_ptr,
 const std::shared_ptr<const TensorFunction<1,dim>>         &background_velocity_ptr,
 const double                                                rossby_number,
 const double                                                froude_number)
{
  const unsigned int n_q_points{this->get_current_fe_values().n_quadrature_points};

  // stress form
  if (vector_options.use_stress_form)
  {
    Assert(vector_options.present_sym_velocity_gradients,
           ExcMessage("Symmetric velocity gradients are not allocated in options."));
    AssertDimension(vector_options.present_sym_velocity_gradients->size(),
                    n_q_points);
    vector_options.present_sym_velocity_gradients
      = this->get_symmetric_gradients(name,
                                      velocity);
  }

  // stabilization related solution values
  if (stabilization_flags & (apply_supg|apply_pspg))
  {
    Assert(vector_options.present_velocity_laplaceans,
           ExcMessage("Velocity laplaceans are not allocated in options."));
    AssertDimension(vector_options.present_velocity_laplaceans->size(),
                    n_q_points);
    vector_options.present_velocity_laplaceans
      = this->get_laplacians(name, velocity);

    Assert(vector_options.present_pressure_gradients,
           ExcMessage("Pressure gradients are not allocated in options."));
    AssertDimension(vector_options.present_pressure_gradients->size(),
                        n_q_points);
    vector_options.present_pressure_gradients
      = this->get_gradients(name, pressure);

    if (vector_options.use_stress_form)
    {
      Assert(vector_options.present_velocity_grad_divergences,
             ExcMessage("Gradients of velocity divergence are not allocated in options."));
      AssertDimension(vector_options.present_velocity_grad_divergences->size(),
                      n_q_points);

      const auto &present_hessians = this->get_hessians(name,
                                                        velocity);

      std::vector<Tensor<1, dim>> &present_velocity_grad_divergences =
          vector_options.present_velocity_grad_divergences.value();
      for (std::size_t q=0; q<present_hessians.size(); ++q)
      {
        present_velocity_grad_divergences[q] = 0;
        for (unsigned int d=0; d<dim; ++d)
          present_velocity_grad_divergences[q] += present_hessians[q][d][d];
      }
    }
  }

  // Coriolis term
  if (angular_velocity_ptr != nullptr)
  {
    Assert(rossby_number > 0.0,
           ExcLowerRangeType<double>(rossby_number, 0.0));

    vector_options.angular_velocity = angular_velocity_ptr->value();
    vector_options.rossby_number = rossby_number;

    scalar_options.angular_velocity = angular_velocity_ptr->value();
    scalar_options.rossby_number = rossby_number;
  }

  // body force
  if (body_force_ptr != nullptr)
  {
    Assert(froude_number > 0.0,
           ExcLowerRangeType<double>(froude_number, 0.0));

    Assert(vector_options.body_force_values,
           ExcMessage("Body force values are not allocated in options."));
    AssertDimension(vector_options.body_force_values->size(),
                    n_q_points);

    body_force_ptr->value_list(this->get_quadrature_points(),
                               *vector_options.body_force_values);
    vector_options.froude_number = froude_number;
    scalar_options.froude_number = froude_number;
  }

  // background field
  if (background_velocity_ptr != nullptr)
  {
    Assert(vector_options.background_velocity_values,
           ExcMessage("Background velocity values are not allocated in options."));
    AssertDimension(vector_options.background_velocity_values->size(),
                    n_q_points);
    background_velocity_ptr->value_list(this->get_quadrature_points(),
                                        *vector_options.background_velocity_values);

    Assert(vector_options.background_velocity_gradients,
           ExcMessage("Background velocity gradients are not allocated in options."));
    AssertDimension(vector_options.background_velocity_gradients->size(),
                    n_q_points);
    background_velocity_ptr->gradient_list(this->get_quadrature_points(),
                                           *vector_options.background_velocity_gradients);
  }
}



template <int dim>
inline void ScratchData<dim>::
assign_scalar_options_local_cell
(const unsigned int q)
{
  // stress form
  if (vector_options.use_stress_form)
  {
    Assert(vector_options.present_sym_velocity_gradients,
           ExcMessage("Symmetric velocity gradients are not allocated in options."));
    scalar_options.present_symmetric_velocity_gradient =
          vector_options.present_sym_velocity_gradients->at(q);
  }
  // body force
  if (vector_options.body_force_values)
    scalar_options.body_force_value =
        vector_options.body_force_values->at(q);
}





}  // namespace RightHandSide

}  // namespace AssemblyData

}  // namespace Hydrodynamic



#endif /* INCLUDE_HYDRODYNAMIC_ASSEMBLY_DATA_H_ */
