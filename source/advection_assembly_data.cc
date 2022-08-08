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
advection_field_values(quadrature.size()),
present_values(quadrature.size()),
present_gradients(quadrature.size()),
present_strong_residuals(quadrature.size())
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
advection_field_values(other.advection_field_values),
present_values(other.present_values),
present_gradients(other.present_gradients),
present_strong_residuals(other.present_strong_residuals)
{}



template <int dim>
void ScratchData<dim>::
assign_vector_options_local_cell
(const std::shared_ptr<const Function<dim>>          &source_term_ptr,
 const std::shared_ptr<const TensorFunction<1,dim>>  &background_advection_ptr,
 const std::shared_ptr<const Function<dim>>          &reference_field_ptr,
 const double                                         gradient_scaling)
{
  const unsigned int n_q_points{this->get_current_fe_values().n_quadrature_points};

  if (source_term_ptr != nullptr)
  {
    Assert(vector_options.source_term_values,
           ExcMessage("Source term values are not allocated in options."));
    AssertDimension(vector_options.source_term_values->size(),
                    n_q_points);

    source_term_ptr->value_list(this->get_quadrature_points(),
                                *vector_options.source_term_values);
  }

  if (background_advection_ptr != nullptr)
  {
    Assert(vector_options.background_advection_values,
           ExcMessage("Source term values are not allocated in options."));
    AssertDimension(vector_options.background_advection_values->size(),
                    n_q_points);

    background_advection_ptr->value_list(this->get_quadrature_points(),
                                         *vector_options.background_advection_values);
  }

  if (reference_field_ptr != nullptr)
  {
    Assert(gradient_scaling > 0.0,
           ExcLowerRangeType<double>(gradient_scaling, 0.0));

    Assert(vector_options.reference_gradients,
           ExcMessage("Reference field gradients are not allocated in options."));
    AssertDimension(vector_options.reference_gradients->size(),
                    n_q_points);

    reference_field_ptr->gradient_list(this->get_quadrature_points(),
                                       *vector_options.reference_gradients);
    vector_options.gradient_scaling = gradient_scaling;
    scalar_options.gradient_scaling = gradient_scaling;
  }
}



template <int dim>
void ScratchData<dim>::
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
void ScratchData<dim>::
assign_scalar_options_local_cell
(const unsigned int q)
{
  if (vector_options.source_term_values)
    scalar_options.source_term_value = vector_options.source_term_values->at(q);

  if (vector_options.reference_gradients)
    scalar_options.reference_gradient = vector_options.reference_gradients->at(q);
}



template <int dim>
void ScratchData<dim>::
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
void ScratchData<dim>::
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

// explicit instantiations
template class ScratchData<2>;
template class ScratchData<3>;

} // namespace AssemblyData

} // namespace Advection
