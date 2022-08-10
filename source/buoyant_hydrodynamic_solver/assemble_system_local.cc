/*
 * assemble_system_local.cc
 *
 *  Created on: Apr 14, 2022
 *      Author: sg
 */

#include <assembly_functions.h>
#include <buoyant_hydrodynamic_solver.h>

#include <optional>

namespace BuoyantHydrodynamic {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
assemble_system_local_cell
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 AssemblyData::Matrix::ScratchData<dim>                &scratch,
 MeshWorker::CopyData<1,1,1>                           &data,
 const bool                                             use_newton_linearization) const
{
  data.matrices[0] = 0;
  data.vectors[0] = 0;
  cell->get_dof_indices(data.local_dof_indices[0]);

  const auto &fe_values = scratch.reinit(cell);
  scratch.extract_local_dof_values("evaluation_point",
                                   this->evaluation_point);
  const auto &JxW = scratch.get_JxW_values();

  const FEValuesExtractors::Vector  velocity(this->velocity_fe_index);
  const FEValuesExtractors::Scalar  pressure(this->pressure_fe_index);
  const FEValuesExtractors::Scalar  density(this->scalar_fe_index);

  const double nu{1.0 / this->reynolds_number};
  const double delta{this->Hydrodynamic::Solver<dim, TriangulationType>::c *
                     std::pow(cell->diameter(), 2)};
  const double delta_density{this->Advection::Solver<dim, TriangulationType>::c *
                             std::pow(cell->diameter(), 2)};
  const double nu_density{this->Advection::Solver<dim, TriangulationType>::nu};

  // solution values
  scratch.present_velocity_values = scratch.get_values("evaluation_point",
                                                       velocity);
  scratch.present_velocity_gradients = scratch.get_gradients("evaluation_point",
                                                             velocity);
  scratch.present_pressure_values = scratch.get_values("evaluation_point",
                                                       pressure);
  scratch.present_values = scratch.get_values("evaluation_point",
                                              density);
  scratch.present_gradients = scratch.get_gradients("evaluation_point",
                                                    density);

  scratch.assign_vector_options_local_cell("evaluation_point",
                                           velocity,
                                           pressure,
                                           this->angular_velocity_ptr,
                                           this->body_force_ptr,
                                           this->gravity_field_ptr,
                                           this->background_velocity_ptr,
                                           this->source_term_ptr,
                                           this->reference_field_ptr,
                                           this->rossby_number,
                                           this->froude_number,
                                           this->gradient_scaling_number);

  // stabilization
  compute_strong_residuals(scratch, nu);

  for (const auto q: fe_values.quadrature_point_indices())
  {
    for (const auto i: fe_values.dof_indices())
    {
      scratch.phi_velocity[i] = fe_values[velocity].value(i, q);
      scratch.grad_phi_velocity[i] = fe_values[velocity].gradient(i, q);
      scratch.div_phi_velocity[i] = fe_values[velocity].divergence(i, q);
      scratch.phi_pressure[i] = fe_values[pressure].value(i, q);
      scratch.grad_phi_pressure[i] = fe_values[pressure].gradient(i, q);
      scratch.phi[i] = fe_values[density].value(i, q);
      scratch.grad_phi[i] = fe_values[density].gradient(i, q);
    }

    // assign optional shape functions
    scratch.assign_optional_shape_functions_system_local(velocity, pressure, q);

    for (const auto i: fe_values.dof_indices())
    {
      for (const auto j: fe_values.dof_indices())
      {
        const double matrix{compute_matrix(scratch,
                                           i,
                                           j,
                                           q,
                                           nu,
                                           delta,
                                           this->mu,
                                           delta_density,
                                           nu_density,
                                           use_newton_linearization)};

        data.matrices[0](i, j) += matrix * JxW[q];
      }

      const double rhs{compute_rhs(scratch,
                                   i,
                                   q,
                                   nu,
                                   this->mu,
                                   delta,
                                   delta_density)};

      data.vectors[0](i) += rhs * JxW[q];
    }

  } // end loop over cell quadrature points

}



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
assemble_system_local_boundary
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 const unsigned int                                     face_number,
 AssemblyData::Matrix::ScratchData<dim>                &scratch,
 MeshWorker::CopyData<1,1,1>                           &data) const
{
  this->Hydrodynamic::Solver<dim, TriangulationType>::
  assemble_local_boundary(cell,
                          face_number,
                          scratch,
                          data);

  // Dirichlet boundary conditions
  const typename ScalarBoundaryConditions<dim>::BCMapping
  &dirichlet_bcs = this->scalar_boundary_conditions.dirichlet_bcs;
  const types::boundary_id  boundary_id{cell->face(face_number)->boundary_id()};

  if (dirichlet_bcs.find(boundary_id) != dirichlet_bcs.end())
  {
    Advection::AssemblyData::Matrix::
    ScratchData<dim> &advection_scratch
      = static_cast<Advection::AssemblyData::Matrix::ScratchData<dim> &>(scratch);

    const FEValuesExtractors::Vector  velocity(this->velocity_fe_index);
    const FEValuesExtractors::Scalar  density(this->scalar_fe_index);

    const auto &fe_face_values = scratch.reinit(cell, face_number);
    const auto &JxW = scratch.get_JxW_values();

    // evaluate solution
    scratch.extract_local_dof_values("evaluation_point",
                                     this->evaluation_point);
    const auto &present_values = scratch.get_values("evaluation_point",
                                                    density);
    const auto &present_velocity_values = scratch.get_values("evaluation_point",
                                                             velocity);

    // boundary values
    dirichlet_bcs.at(boundary_id)->value_list(fe_face_values.get_quadrature_points(),
                                              advection_scratch.vector_options.boundary_values);
    const auto &boundary_values{advection_scratch.vector_options.boundary_values};

    // normal vectors
    const auto &normal_vectors = fe_face_values.get_normal_vectors();

    // loop over face quadrature points
    for (const auto q: fe_face_values.quadrature_point_indices())
      if (normal_vectors[q] * present_velocity_values[q] < 0.)
      {
        // extract the test function's values at the face quadrature points
        for (const auto i: fe_face_values.dof_indices())
          advection_scratch.phi[i] = fe_face_values[density].value(i,q);

        // loop over the degrees of freedom
        for (const auto i: fe_face_values.dof_indices())
        {
          for (const auto j: fe_face_values.dof_indices())
            data.matrices[0](i, j) -= normal_vectors[q] *
                                      present_velocity_values[q] *
                                      advection_scratch.phi[i] *
                                      advection_scratch.phi[j] *
                                      JxW[q];
          data.vectors[0](i) += present_velocity_values[q] *
                                normal_vectors[q] *
                                advection_scratch.phi[i] *
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
(const typename DoFHandler<2>::active_cell_iterator &,
 AssemblyData::Matrix::ScratchData<2>               &,
 MeshWorker::CopyData<1,1,1>                        &,
 const bool                                           ) const;
template
void
Solver<3>::
assemble_system_local_cell
(const typename DoFHandler<3>::active_cell_iterator &,
 AssemblyData::Matrix::ScratchData<3>               &,
 MeshWorker::CopyData<1,1,1>                        &,
 const bool                                           ) const;

template
void
Solver<2>::
assemble_system_local_boundary
(const typename DoFHandler<2>::active_cell_iterator &,
 const unsigned int                                  ,
 AssemblyData::Matrix::ScratchData<2>               &,
 MeshWorker::CopyData<1,1,1>                        &) const;
template
void
Solver<3>::
assemble_system_local_boundary
(const typename DoFHandler<3>::active_cell_iterator &,
 const unsigned int                                  ,
 AssemblyData::Matrix::ScratchData<3>               &,
 MeshWorker::CopyData<1,1,1>                        &) const;



}  // namespace BuoyantHydrodynamic
