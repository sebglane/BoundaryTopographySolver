/*
 * assemble_system_local.cc
 *
 *  Created on: Apr 11, 2022
 *      Author: sg
 */

#include <advection_solver.h>
#include <assembly_functions.h>

namespace Advection {

template<int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
assemble_system_local_cell
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 AssemblyData::Matrix::ScratchData<dim>         &scratch,
 MeshWorker::CopyData<1,1,1>                           &data) const
{
  cell->get_dof_indices(data.local_dof_indices[0]);

  const auto &fe_values = scratch.reinit(cell);
  scratch.extract_local_dof_values("evaluation_point",
                                   this->evaluation_point);
  const auto &JxW = scratch.get_JxW_values();

  // stabilization parameter
  const double delta{c * std::pow(cell->diameter(), 2)};

  // solution values
  scratch.present_gradients = scratch.get_gradients("evaluation_point",
                                                    FEValuesExtractors::Scalar(FEValuesExtractors::Scalar(scalar_fe_index)));

  // advection field
  advection_field_ptr->value_list(scratch.get_quadrature_points(),
                                  scratch.advection_field_values);

  // assign vector options
  scratch.assign_vector_options_local_cell(source_term_ptr,
                                           nullptr,
                                           reference_field_ptr,
                                           gradient_scaling_number);
  scratch.adjust_advection_field_local_cell();

  // stabilization
  compute_strong_residual(scratch);

  // loop over cell quadrature points
  for (const auto q: fe_values.quadrature_point_indices())
  {
    for (const auto i: fe_values.dof_indices())
    {
      scratch.phi[i] = fe_values.shape_value(i, q);
      scratch.grad_phi[i] = fe_values.shape_grad(i, q);
    }

    for (const auto i: fe_values.dof_indices())
    {
      for (const auto j: fe_values.dof_indices())
        data.matrices[0](i, j) += compute_matrix(scratch, i, j, q, delta) * JxW[q];

      data.vectors[0](i) += compute_rhs(scratch, i, q, delta) * JxW[q];
    }
  } // loop over cell quadrature points
}



// explicit instantiation
template
void
Solver<2>::
assemble_system_local_cell
(const typename DoFHandler<2>::active_cell_iterator  &cell,
 AssemblyData::RightHandSide::ScratchData<2>         &scratch,
 MeshWorker::CopyData<1,1,1>                         &data) const;
template
void
Solver<3>::
assemble_system_local_cell
(const typename DoFHandler<3>::active_cell_iterator  &cell,
 AssemblyData::RightHandSide::ScratchData<3>         &scratch,
 MeshWorker::CopyData<1,1,1>                         &data) const;

}  // namespace Advection
