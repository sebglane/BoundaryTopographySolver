/*
 * advection_assembly_data.h
 *
 *  Created on: Apr 11, 2022
 *      Author: sg
 */

#ifndef INCLUDE_ADVECTION_ASSEMBLY_DATA_H_
#define INCLUDE_ADVECTION_ASSEMBLY_DATA_H_

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/meshworker/scratch_data.h>

#include <advection_options.h>

#include <memory>

namespace Advection {

using namespace dealii;

namespace AssemblyData {

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
    const bool                allocate_source_term = false,
    const bool                allocate_boundary_values = false,
    const bool                allocate_background_velocity = false,
    const bool                allocate_reference_gradient = false);

  ScratchData(
    const FiniteElement<dim> &fe,
    const Quadrature<dim>    &quadrature,
    const UpdateFlags        &update_flags,
    const Quadrature<dim-1>  &face_quadrature   = Quadrature<dim-1>(),
    const UpdateFlags        &face_update_flags = update_default,
    const bool                allocate_source_term = false,
    const bool                allocate_boundary_values = false,
    const bool                allocate_background_velocity = false,
    const bool                allocate_reference_gradient = false);

  ScratchData(const ScratchData<dim>  &data);

  void assign_vector_options_local_cell(
    const std::shared_ptr<const Function<dim>>         &source_term_ptr = nullptr,
    const std::shared_ptr<const TensorFunction<1,dim>> &background_advection_ptr = nullptr,
    const std::shared_ptr<const Function<dim>>         &reference_field_ptr = nullptr,
    const double                                        gradient_scaling = 0.0);

  void assign_vector_options_local_boundary(
    const std::shared_ptr<const Function<dim>>         &boundary_function_ptr,
    const std::shared_ptr<const TensorFunction<1,dim>> &background_advection_ptr = nullptr);

  void assign_scalar_options_local_cell(const unsigned int q_point_index);

  void adjust_advection_field_local_cell();

  void adjust_advection_field_local_boundary();

  // optional parameters
  OptionalScalarArguments<dim>  scalar_options;
  OptionalVectorArguments<dim>  vector_options;

  // shape functions
  std::vector<double>         phi;
  std::vector<Tensor<1, dim>> grad_phi;

  // advection field
  std::vector<Tensor<1,dim>>  advection_field_values;

};



template <int dim>
inline void ScratchData<dim>::
assign_vector_options_local_cell
(const std::shared_ptr<const Function<dim>>          &source_term_ptr,
 const std::shared_ptr<const TensorFunction<1,dim>>  &background_advection_ptr,
 const std::shared_ptr<const Function<dim>>          &reference_field_ptr,
 const double                                         gradient_scaling)
{
  if (source_term_ptr != nullptr)
    source_term_ptr->value_list(this->get_quadrature_points(),
                                *vector_options.source_term_values);

  if (background_advection_ptr != nullptr)
    background_advection_ptr->value_list(this->get_quadrature_points(),
                                         *vector_options.background_advection_values);

  if (reference_field_ptr != nullptr)
  {
    Assert(gradient_scaling > 0.0,
           ExcLowerRangeType<double>(gradient_scaling, 0.0));

    reference_field_ptr->gradient_list(this->get_quadrature_points(),
                                       *vector_options.reference_gradients);

    vector_options.gradient_scaling = gradient_scaling;
    scalar_options.gradient_scaling = gradient_scaling;
  }
}



template <int dim>
inline void ScratchData<dim>::
assign_vector_options_local_boundary
(const std::shared_ptr<const Function<dim>>         &boundary_function_ptr,
 const std::shared_ptr<const TensorFunction<1,dim>> &background_advection_ptr)
{
  Assert(boundary_function_ptr != nullptr, ExcInternalError());
  if (boundary_function_ptr != nullptr)
    boundary_function_ptr->value_list(this->get_quadrature_points(),
                                      vector_options.boundary_values);

  if (background_advection_ptr != nullptr)
  {
    const unsigned int n_q_points{(unsigned int)vector_options.advection_field_face_values.size()};

    if (vector_options.background_advection_values->size() != n_q_points)
      vector_options.background_advection_values->resize(n_q_points);

    background_advection_ptr->value_list(this->get_quadrature_points(),
                                         *vector_options.background_advection_values);
  }
}



template <int dim>
inline void ScratchData<dim>::
assign_scalar_options_local_cell
(const unsigned int q)
{
  if (vector_options.source_term_values)
    scalar_options.source_term_value = vector_options.source_term_values->at(q);

  if (vector_options.reference_gradients)
    scalar_options.reference_gradient = vector_options.reference_gradients->at(q);
}



template <int dim>
inline void ScratchData<dim>::
adjust_advection_field_local_cell()
{
  if (vector_options.background_advection_values)
  {
    AssertDimension(vector_options.background_advection_values->size(),
                    advection_field_values.size());

    for (unsigned int i=0; i<advection_field_values.size(); ++i)
      advection_field_values[i] += vector_options.background_advection_values->at(i);
  }
}



template <int dim>
inline void ScratchData<dim>::
adjust_advection_field_local_boundary()
{
  if (vector_options.background_advection_values)
  {
    AssertDimension(vector_options.background_advection_values->size(),
                    vector_options.advection_field_face_values.size());

    for (unsigned int i=0; i<advection_field_values.size(); ++i)
      vector_options.advection_field_face_values[i] += vector_options.background_advection_values->at(i);
  }
}

namespace Matrix {

template <int dim>
using ScratchData = ScratchData<dim>;

}  // namespace Matrix

namespace RightHandSide {

template <int dim>
using ScratchData = ScratchData<dim>;

}  // namespace RightHandSide

}  // namespace AssemblyData

}  // namespace Advection



#endif /* INCLUDE_ADVECTION_ASSEMBLY_DATA_H_ */
