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

  // solution values
  const auto &present_gradients = scratch.get_gradients("evaluation_point",
                                                        FEValuesExtractors::Scalar(0));

  // source term
  if (source_term_ptr != nullptr)
    source_term_ptr->value_list(fe_values.get_quadrature_points(),
                                *scratch.vector_options.source_term_values);

  // advection field
  advection_field_ptr->value_list(fe_values.get_quadrature_points(),
                                  scratch.advection_field_values);

  // stabilization parameter
  //  const double delta{c * std::pow(cell->diameter(), 2)};
  double delta;
  {
    double max_velocity = 0;

    for (std::size_t q=0; q<scratch.advection_field_values.size(); ++q)
      max_velocity = std::max(scratch.advection_field_values[q].norm(), max_velocity);

    delta = 0.5 * cell->diameter() / max_velocity;
  }
  Assert(delta > 0.0, ExcLowerRangeType<double>(0.0, delta));

  // loop over cell quadrature points
  for (const auto q: fe_values.quadrature_point_indices())
  {
    // source term
    if (scratch.vector_options.source_term_values)
      scratch.scalar_options.source_term_value = scratch.vector_options.source_term_values->at(q);

    for (const auto i: fe_values.dof_indices())
    {
      scratch.phi[i] = fe_values.shape_value(i, q);
      scratch.grad_phi[i] = fe_values.shape_grad(i, q);
    }

    for (const auto i: fe_values.dof_indices())
    {
      const double test_function_value{scratch.phi[i] +
                                       delta *
                                       scratch.advection_field_values[q] *
                                       scratch.grad_phi[i]};

      for (const auto j: fe_values.dof_indices())
      {
        const double matrix{compute_matrix(scratch.grad_phi[j],
                                           scratch.advection_field_values[q],
                                           test_function_value,
                                           scratch.scalar_options)};
        data.matrices[0](i, j) += matrix * JxW[q];
      }

      const double rhs{compute_rhs(present_gradients[q],
                                   scratch.advection_field_values[q],
                                   test_function_value,
                                   scratch.scalar_options)};

      data.vectors[0](i) += rhs * JxW[q];
    }
  } // loop over cell quadrature points
}



template<int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
assemble_system_local_boundary
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 const unsigned int                                     face_number,
 AssemblyData::Matrix::ScratchData<dim>                &scratch,
 MeshWorker::CopyData<1,1,1>                           &data) const
{
  const typename ScalarBoundaryConditions<dim>::BCMapping
  &dirichlet_bcs = boundary_conditions.dirichlet_bcs;
  const types::boundary_id  boundary_id{cell->face(face_number)->boundary_id()};

  if (!dirichlet_bcs.empty())
    if (dirichlet_bcs.find(boundary_id) != dirichlet_bcs.end())
    {
      const auto &fe_face_values = scratch.reinit(cell, face_number);
      const auto &JxW = scratch.get_JxW_values();

      // evaluate solution
      scratch.extract_local_dof_values("evaluation_point",
                                       this->evaluation_point);
      const auto &present_values  = scratch.get_values("evaluation_point",
                                                       FEValuesExtractors::Scalar(0));

      // boundary values
      dirichlet_bcs.at(boundary_id)->value_list(fe_face_values.get_quadrature_points(),
                                                scratch.vector_options.boundary_values);
      const auto &boundary_values{scratch.vector_options.boundary_values};

      // advection field
      advection_field_ptr->value_list(fe_face_values.get_quadrature_points(),
                                      scratch.vector_options.advection_field_face_values);
      const auto &advection_field_values{scratch.vector_options.advection_field_face_values};

      // normal vectors
      const auto &normal_vectors = fe_face_values.get_normal_vectors();

      // loop over face quadrature points
      for (const auto q: fe_face_values.quadrature_point_indices())
        if (normal_vectors[q] * advection_field_values[q] < 0.)
        {
          // extract the test function's values at the face quadrature points
          for (const auto i: fe_face_values.dof_indices())
            scratch.phi[i] = fe_face_values.shape_value(i,q);

          // loop over the degrees of freedom
          for (const auto i: fe_face_values.dof_indices())
          {
            for (const auto j: fe_face_values.dof_indices())
              data.matrices[0](i, j) -= normal_vectors[q] *
                                        advection_field_values[q] *
                                        scratch.phi[i] *
                                        scratch.phi[j] *
                                        JxW[q];
            data.vectors[0](i) += advection_field_values[q] *
                                  normal_vectors[q] *
                                  scratch.phi[i] *
                                  (present_values[q] - boundary_values[q]) *
                                  JxW[q];
          }
        } // loop over face quadrature points
    }
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

template
void
Solver<2>::
assemble_system_local_boundary
(const typename DoFHandler<2>::active_cell_iterator  &cell,
 const unsigned int                                   face_number,
 AssemblyData::RightHandSide::ScratchData<2>         &scratch,
 MeshWorker::CopyData<1,1,1>                         &data) const;
template
void
Solver<3>::
assemble_system_local_boundary
(const typename DoFHandler<3>::active_cell_iterator  &cell,
 const unsigned int                                   face_number,
 AssemblyData::RightHandSide::ScratchData<3>         &scratch,
 MeshWorker::CopyData<1,1,1>                         &data) const;


}  // namespace Advection
