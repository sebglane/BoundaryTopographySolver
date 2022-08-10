/*
 * assemble_local_boundary.cc
 *
 *  Created on: Aug 10, 2022
 *      Author: sg
 */
#include <assembly_functions.h>
#include <hydrodynamic_solver.h>

#include <functional>
#include <optional>


namespace Hydrodynamic {

template <int dim, typename TriangulationType>
template <int n_matrices, int n_vectors, int n_dof_indices>
void Solver<dim, TriangulationType>::
assemble_local_boundary
(const typename DoFHandler<dim>::active_cell_iterator      &cell,
 const unsigned int                                         face_number,
 AssemblyData::ScratchData<dim>                            &scratch,
 MeshWorker::CopyData<n_matrices,n_vectors,n_dof_indices>  &data) const
{
  const FEValuesExtractors::Vector  velocity(velocity_fe_index);
  const FEValuesExtractors::Scalar  pressure(pressure_fe_index);

  const types::boundary_id  boundary_id{cell->face(face_number)->boundary_id()};

  const typename VectorBoundaryConditions<dim>::NeumannBCMapping
  &neumann_bcs = velocity_boundary_conditions.neumann_bcs;

  // Traction boundary conditions
  if (!neumann_bcs.empty())
    if (neumann_bcs.find(boundary_id) != neumann_bcs.end())
    {
      const auto &fe_face_values = scratch.reinit(cell, face_number);
      const auto &JxW = scratch.get_JxW_values();

      // assign vector options
      scratch.assign_vector_options_boundary("",
                                             velocity,
                                             pressure,
                                             0.0,
                                             neumann_bcs.at(boundary_id),
                                             background_velocity_ptr);

      // boundary traction
      const auto &boundary_tractions{scratch.vector_options.boundary_traction_values};

      // loop over face quadrature points
      for (const auto q: fe_face_values.quadrature_point_indices())
      {
        // extract the test function's values at the face quadrature points
        for (const auto i: fe_face_values.dof_indices())
          scratch.phi_velocity[i] = fe_face_values[velocity].value(i, q);

        // loop over the degrees of freedom
        for (const auto i: fe_face_values.dof_indices())
          data.vectors[0](i) += scratch.phi_velocity[i] *
                                boundary_tractions[q] *
                                JxW[q];
      } // loop over face quadrature points
    }

  // Traction-free boundary conditions
  if (include_boundary_stress_terms)
    if (std::find(boundary_stress_ids.begin(),
                  boundary_stress_ids.end(),
                  boundary_id) != boundary_stress_ids.end())
    {
      const double nu{1.0 / reynolds_number};

      const auto &fe_face_values = scratch.reinit(cell, face_number);
      const auto &JxW = scratch.get_JxW_values();

      // evaluate solution
      scratch.extract_local_dof_values("evaluation_point",
                                       this->evaluation_point);

      // normal vectors
      const auto &face_normal_vectors = scratch.get_normal_vectors();

      // assign vector options
      scratch.assign_vector_options_boundary("evaluation_point",
                                             velocity,
                                             pressure,
                                             nu,
                                             nullptr,
                                             background_velocity_ptr);

      // boundary traction
      const auto &boundary_tractions{scratch.vector_options.boundary_traction_values};

      // loop over face quadrature points
      for (const auto q: fe_face_values.quadrature_point_indices())
      {
        // extract the test function's values at the face quadrature points
        if (data.matrices.size() > 0)
        {
          for (const auto i: fe_face_values.dof_indices())
          {
            scratch.phi_velocity[i] = fe_face_values[velocity].value(i, q);
            scratch.phi_pressure[i] = fe_face_values[pressure].value(i, q);
          }

          // assign optional shape functions
          scratch.assign_optional_shape_functions_system_local_boundary(velocity, q);

        }
        else
          for (const auto i: fe_face_values.dof_indices())
            scratch.phi_velocity[i] = fe_face_values[velocity].value(i, q);

        // loop over the degrees of freedom
        if (data.matrices.size() > 0)
        {
          if (scratch.vector_options.use_stress_form)
            for (const auto i: fe_face_values.dof_indices())
              for (const auto j: fe_face_values.dof_indices())
                data.matrices[0](i, j) -=
                    (-scratch.phi_pressure[j] * face_normal_vectors[q] +
                     2.0 * nu * scratch.sym_grad_phi_velocity[j] * face_normal_vectors[q]) *
                     scratch.phi_velocity[i] * JxW[q];
          else
            for (const auto i: fe_face_values.dof_indices())
              for (const auto j: fe_face_values.dof_indices())
                data.matrices[0](i, j) -=
                    (-scratch.phi_pressure[j] * face_normal_vectors[q] +
                     nu * scratch.grad_phi_velocity[j] * face_normal_vectors[q]) *
                     scratch.phi_velocity[i] * JxW[q];
        }

        for (const auto i: fe_face_values.dof_indices())
          data.vectors[0](i) += scratch.phi_velocity[i] *
                                boundary_tractions[q] *
                                JxW[q];
      } // loop over face quadrature points
    }
}



// explicit instantiation
template
void
Solver<2>::
assemble_local_boundary<0,1,1>
(const typename DoFHandler<2>::active_cell_iterator &,
 const unsigned int                                  ,
 AssemblyData::Matrix::ScratchData<2>               &,
 MeshWorker::CopyData<0,1,1>                        &) const;
template
void
Solver<3>::
assemble_local_boundary<0,1,1>
(const typename DoFHandler<3>::active_cell_iterator &,
 const unsigned int                                  ,
 AssemblyData::Matrix::ScratchData<3>               &,
 MeshWorker::CopyData<0,1,1>                        &) const;

template
void
Solver<2>::
assemble_local_boundary<1,1,1>
(const typename DoFHandler<2>::active_cell_iterator &,
 const unsigned int                                  ,
 AssemblyData::Matrix::ScratchData<2>               &,
 MeshWorker::CopyData<1,1,1>                        &) const;
template
void
Solver<3>::
assemble_local_boundary<1,1,1>
(const typename DoFHandler<3>::active_cell_iterator &,
 const unsigned int                                  ,
 AssemblyData::Matrix::ScratchData<3>               &,
 MeshWorker::CopyData<1,1,1>                        &) const;



}  // namespace Hydrodynamic
