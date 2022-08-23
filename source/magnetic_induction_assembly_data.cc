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


template <int dim>
void ScratchData<dim>::assign_vector_options
(const std::shared_ptr<const TensorFunction<1,dim>> &velocity_field_ptr,
 const std::shared_ptr<const TensorFunction<1,dim>> &background_magnetic_field_ptr)
{
  const unsigned int n_q_points{this->get_current_fe_values().n_quadrature_points};

  // background magnetic field
  if (background_magnetic_field_ptr != nullptr)
  {
    Assert(vector_options.background_magnetic_field_values,
           ExcMessage("Background magnetic field values are not allocated in options."));
    AssertDimension(vector_options.background_magnetic_field_values->size(),
                    n_q_points);

    background_magnetic_field_ptr->value_list(this->get_quadrature_points(),
                                              *vector_options.background_magnetic_field_values);

    Assert(vector_options.background_magnetic_field_curls,
           ExcMessage("Background magnetic field curls are not allocated in options."));
    AssertDimension(vector_options.background_magnetic_field_curls->size(),
                    n_q_points);
    Assert(vector_options.background_magnetic_field_divergences,
           ExcMessage("Background magnetic field curls are not allocated in options."));
    AssertDimension(vector_options.background_magnetic_field_divergences->size(),
                    n_q_points);

    std::vector<Tensor<2,dim>> background_magnetic_field_gradients(n_q_points);

    background_magnetic_field_ptr->gradient_list(this->get_quadrature_points(),
                                                 background_magnetic_field_gradients);

    if constexpr(dim == 2)
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        vector_options.background_magnetic_field_divergences->at(q) = trace(background_magnetic_field_gradients[q]);
        vector_options.background_magnetic_field_curls->at(q)[0] = background_magnetic_field_gradients[q][1][0] -
                                                                   background_magnetic_field_gradients[q][0][1];
      }
    else if constexpr(dim == 3)
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        vector_options.background_magnetic_field_divergences->at(q) = trace(background_magnetic_field_gradients[q]);
        vector_options.background_magnetic_field_curls->at(q)[0] = background_magnetic_field_gradients[q][2][1] -
                                                                   background_magnetic_field_gradients[q][1][2];
        vector_options.background_magnetic_field_curls->at(q)[1] = background_magnetic_field_gradients[q][0][2] -
                                                                   background_magnetic_field_gradients[q][2][0];
        vector_options.background_magnetic_field_curls->at(q)[2] = background_magnetic_field_gradients[q][1][0] -
                                                                   background_magnetic_field_gradients[q][0][1];
      }
  }

  // velocity field
  if (velocity_field_ptr != nullptr)
  {
    Assert(vector_options.velocity_field_values,
           ExcMessage("Velocity field values are not allocated in options."));
    AssertDimension(vector_options.velocity_field_values->size(),
                    n_q_points);

    velocity_field_ptr->value_list(this->get_quadrature_points(),
                                   *vector_options.velocity_field_values);
  }
}



template <int dim>
void ScratchData<dim>::adjust_magnetic_field_local_cell()
{
  if (vector_options.background_magnetic_field_values)
  {
    Assert(vector_options.background_magnetic_field_curls,
           ExcMessage("Background magnetic field curls are not assigned in options."));
    Assert(vector_options.background_magnetic_field_divergences,
           ExcMessage("Background magnetic field divergences are not assigned in options."));

    AssertDimension(vector_options.background_magnetic_field_values->size(),
                    present_magnetic_field_values.size());
    AssertDimension(vector_options.background_magnetic_field_curls->size(),
                    present_magnetic_field_curls.size());
    AssertDimension(vector_options.background_magnetic_field_divergences->size(),
                    present_magnetic_field_divergences.size());

    for (unsigned int q=0; q<present_magnetic_field_values.size(); ++q)
    {
      present_magnetic_field_values[q] += vector_options.background_magnetic_field_values->at(q);
      present_magnetic_field_curls[q] += vector_options.background_magnetic_field_curls->at(q);
      present_magnetic_field_divergences[q] += vector_options.background_magnetic_field_divergences->at(q);
    }
  }
}



// explicit instantiations
template class ScratchData<2>;
template class ScratchData<3>;

} // namespace AssemblyData

} // namespace MagneticInduction
