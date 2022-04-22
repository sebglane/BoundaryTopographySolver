/*
 * advection_assembly_data.h
 *
 *  Created on: Apr 11, 2022
 *      Author: sg
 */

#ifndef INCLUDE_ADVECTION_ASSEMBLY_DATA_H_
#define INCLUDE_ADVECTION_ASSEMBLY_DATA_H_

#include <deal.II/meshworker/scratch_data.h>

#include <advection_options.h>

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

  // optional parameters
  OptionalScalarArguments<dim>  scalar_options;
  OptionalVectorArguments<dim>  vector_options;

  // shape functions
  std::vector<double>         phi;
  std::vector<Tensor<1, dim>> grad_phi;

  // advection field
  std::vector<Tensor<1,dim>>  advection_field_values;

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

}  // namespace Advection



#endif /* INCLUDE_ADVECTION_ASSEMBLY_DATA_H_ */
