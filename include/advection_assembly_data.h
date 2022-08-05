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
  ScalarOptions<dim>  scalar_options;
  VectorOptions<dim>  vector_options;

  // shape functions
  std::vector<double>         phi;
  std::vector<Tensor<1, dim>> grad_phi;

  // advection field
  std::vector<Tensor<1,dim>>  advection_field_values;

  // present solution values
  std::vector<double>         present_values;
  std::vector<Tensor<1, dim>> present_gradients;


  // stabilization related quantities
  std::vector<double>         present_strong_residuals;

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
