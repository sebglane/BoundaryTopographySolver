/*
 * assemble_rhs_local.cc
 *
 *  Created on: Apr 11, 2022
 *      Author: sg
 */

#include <assembly_functions.h>
#include <advection_solver.h>

namespace Advection {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
assemble_rhs_local_cell
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 AssemblyData::RightHandSide::ScratchData<dim>         &scratch,
 MeshWorker::CopyData<0,1,1>                           &data) const
{
  data.vectors[0] = 0;
  cell->get_dof_indices(data.local_dof_indices[0]);

  const auto &fe_values = scratch.reinit(cell);
  scratch.extract_local_dof_values("evaluation_point",
                                   this->evaluation_point);
  const auto &JxW = scratch.get_JxW_values();

  // stabilization parameter
  const double delta{c * std::pow(cell->diameter(), 2)};

  // solution values
  const auto &present_gradients = scratch.get_gradients("evaluation_point",
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
  compute_strong_residual(present_gradients,
                          scratch);

  // loop over cell quadrature points
  for (const auto q: fe_values.quadrature_point_indices())
  {
    scratch.assign_scalar_options_local_cell(q);

    for (const auto i: fe_values.dof_indices())
    {
      scratch.phi[i] = fe_values.shape_value(i, q);
      scratch.grad_phi[i] = fe_values.shape_grad(i, q);
    }

    for (const auto i: fe_values.dof_indices())
    {
      const double rhs{compute_rhs(scratch,
                                   present_gradients[q],
                                   i,
                                   q,
                                   delta)};

      data.vectors[0](i) += rhs * JxW[q];
    }
  } // loop over cell quadrature points
}



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
assemble_rhs_local_boundary
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 const unsigned int                                     face_number,
 AssemblyData::RightHandSide::ScratchData<dim>         &scratch,
 MeshWorker::CopyData<0,1,1>                           &data) const
{
  const typename ScalarBoundaryConditions<dim>::BCMapping
  &dirichlet_bcs = scalar_boundary_conditions.dirichlet_bcs;
  const types::boundary_id  boundary_id{cell->face(face_number)->boundary_id()};

  // Dirichlet boundary condition
  if (!dirichlet_bcs.empty())
    if (dirichlet_bcs.find(boundary_id) != dirichlet_bcs.end())
    {
      const auto &fe_face_values = scratch.reinit(cell, face_number);
      const auto &JxW = scratch.get_JxW_values();

      // evaluate solution
      scratch.extract_local_dof_values("evaluation_point",
                                       this->evaluation_point);
      const auto &present_values  = scratch.get_values("evaluation_point",
                                                       FEValuesExtractors::Scalar(scalar_fe_index));

      // advection field
      advection_field_ptr->value_list(fe_face_values.get_quadrature_points(),
                                      scratch.vector_options.advection_field_face_values);
      const auto &advection_field_values{scratch.vector_options.advection_field_face_values};

      // options
      scratch.assign_vector_options_local_boundary(dirichlet_bcs.at(boundary_id),
                                                   nullptr);
      scratch.adjust_advection_field_local_boundary();

      // boundary values
      const auto &boundary_values{scratch.vector_options.boundary_values};

      // normal vectors
      const auto &normal_vectors = fe_face_values.get_normal_vectors();

      // loop over face quadrature points
      for (const auto q: fe_face_values.quadrature_point_indices())
        if (normal_vectors[q] * advection_field_values[q] < 0.)
        {
          // extract the test function's values at the face quadrature points
          for (const auto i: fe_face_values.dof_indices())
            scratch.phi[i] = fe_face_values.shape_value(i, q);

          // loop over the degrees of freedom
          for (const auto i: fe_face_values.dof_indices())
            data.vectors[0](i) += advection_field_values[q] *
                                  normal_vectors[q] *
                                  scratch.phi[i] *
                                  (present_values[q] - boundary_values[q]) *
                                  JxW[q];
        } // loop over face quadrature points
    }
}



// explicit instantiation
template
void
Solver<2>::
assemble_rhs_local_cell
(const typename DoFHandler<2>::active_cell_iterator  &cell,
 AssemblyData::RightHandSide::ScratchData<2>         &scratch,
 MeshWorker::CopyData<0,1,1>                         &data) const;
template
void
Solver<3>::
assemble_rhs_local_cell
(const typename DoFHandler<3>::active_cell_iterator  &cell,
 AssemblyData::RightHandSide::ScratchData<3>         &scratch,
 MeshWorker::CopyData<0,1,1>                         &data) const;

template
void
Solver<2>::
assemble_rhs_local_boundary
(const typename DoFHandler<2>::active_cell_iterator  &cell,
 const unsigned int                                   face_number,
 AssemblyData::RightHandSide::ScratchData<2>         &scratch,
 MeshWorker::CopyData<0,1,1>                         &data) const;
template
void
Solver<3>::
assemble_rhs_local_boundary
(const typename DoFHandler<3>::active_cell_iterator  &cell,
 const unsigned int                                   face_number,
 AssemblyData::RightHandSide::ScratchData<3>         &scratch,
 MeshWorker::CopyData<0,1,1>                         &data) const;

}  // namespace Advection
