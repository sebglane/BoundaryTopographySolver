/*
 * advection_assembly_data.h
 *
 *  Created on: Apr 11, 2022
 *      Author: sg
 */

#ifndef INCLUDE_ADVECTION_ASSEMBLY_DATA_H_
#define INCLUDE_ADVECTION_ASSEMBLY_DATA_H_

#include <deal.II/meshworker/scratch_data.h>

namespace Advection {

using namespace dealii;

namespace AssemblyData {

template <int dim>
class ScratchData : public MeshWorker::ScratchData<dim>
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
    const bool                allocate_boundary_values = false);

  ScratchData(
    const FiniteElement<dim> &fe,
    const Quadrature<dim>    &quadrature,
    const UpdateFlags        &update_flags,
    const Quadrature<dim-1>  &face_quadrature   = Quadrature<dim-1>(),
    const UpdateFlags        &face_update_flags = update_default,
    const bool                allocate_source_term = false,
    const bool                allocate_boundary_values = false);

  ScratchData(const ScratchData<dim>  &data);

  // shape functions
  std::vector<double>         phi;
  std::vector<Tensor<1, dim>> grad_phi;

  // advection field
  std::vector<Tensor<1,dim>>  advection_field_values;
  std::vector<Tensor<1,dim>>  advection_field_face_values;

  // source term
  std::vector<double> source_term_values;

  // boundary term
  std::vector<double> boundary_values;

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
