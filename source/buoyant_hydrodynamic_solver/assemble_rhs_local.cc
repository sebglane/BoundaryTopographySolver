/*
 * assemble_rhs_local.cc
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

  const FEValuesExtractors::Vector  velocity(this->velocity_fe_index);
  const FEValuesExtractors::Scalar  pressure(this->pressure_fe_index);
  const FEValuesExtractors::Scalar  density(this->scalar_fe_index);

  const double nu{1.0 / this->reynolds_number};
  const double delta{this->Hydrodynamic::Solver<dim, TriangulationType>::c *
                     std::pow(cell->diameter(), 2)};
  const double delta_density{this->Advection::Solver<dim, TriangulationType>::c *
                             std::pow(cell->diameter(), 2)};

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
  compute_strong_residuals(scratch,
                           nu);

  for (const auto q: fe_values.quadrature_point_indices())
  {
    for (const auto i: fe_values.dof_indices())
    {
      scratch.phi_velocity[i] = fe_values[velocity].value(i, q);
      scratch.grad_phi_velocity[i] = fe_values[velocity].gradient(i, q);
      scratch.div_phi_velocity[i] = fe_values[velocity].divergence(i, q);
      scratch.phi_pressure[i] = fe_values[pressure].value(i, q);
      scratch.phi[i] = fe_values[density].value(i, q);
      scratch.grad_phi[i] = fe_values[density].gradient(i, q);
    }

    // assign optional shape functions
    scratch.assign_optional_shape_functions_rhs_local(velocity, pressure, q);

    for (const auto i: fe_values.dof_indices())
    {
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
assemble_rhs_local_boundary
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 const unsigned int                                     face_number,
 AssemblyData::RightHandSide::ScratchData<dim>         &scratch,
 MeshWorker::CopyData<0,1,1>                           &data) const
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
    Advection::AssemblyData::RightHandSide::
    ScratchData<dim> &advection_scratch
      = static_cast<Advection::AssemblyData::RightHandSide::ScratchData<dim> &>(scratch);
    const FEValuesExtractors::Vector  velocity(this->velocity_fe_index);
    const FEValuesExtractors::Scalar  density(this->scalar_fe_index);

    auto &fe_face_values = scratch.reinit(cell, face_number);
    const auto &JxW = scratch.get_JxW_values();

    // evaluate solution
    scratch.extract_local_dof_values("evaluation_point",
                                     this->evaluation_point);
    const auto &present_values  = scratch.get_values("evaluation_point",
                                                     density);

    // advection field
    const auto &present_velocity_values = scratch.get_values("evaluation_point",
                                                             velocity);
    auto &advection_field_values{advection_scratch.vector_options.advection_field_face_values};
    if (this->background_velocity_ptr != nullptr)
    {
      this->background_velocity_ptr->value_list(fe_face_values.get_quadrature_points(),
                                                advection_field_values);
      AssertDimension(present_velocity_values.size(),
                      advection_field_values.size());

      for (unsigned int q=0; q<advection_field_values.size(); ++q)
        advection_field_values[q] += present_velocity_values[q];
    }
    else
      advection_field_values = present_velocity_values;

    // options
    advection_scratch.assign_vector_options_local_boundary(dirichlet_bcs.at(boundary_id),
                                                           nullptr);

    // boundary values
    const auto &boundary_values{advection_scratch.vector_options.boundary_values};

    // normal vectors
    const auto &normal_vectors = fe_face_values.get_normal_vectors();

    // loop over face quadrature points
    for (const auto q: fe_face_values.quadrature_point_indices())
      if (normal_vectors[q] * advection_field_values[q] < 0.)
      {
        // extract the test function's values at the face quadrature points
        for (const auto i: fe_face_values.dof_indices())
          advection_scratch.phi[i] = fe_face_values[density].value(i,q);

        // loop over the degrees of freedom
        for (const auto i: fe_face_values.dof_indices())
          data.vectors[0](i) += advection_field_values[q] *
                                normal_vectors[q] *
                                advection_scratch.phi[i] *
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
(const typename DoFHandler<2>::active_cell_iterator &,
 AssemblyData::RightHandSide::ScratchData<2>        &,
 MeshWorker::CopyData<0,1,1>                        &) const;
template
void Solver<3>::
assemble_rhs_local_cell
(const typename DoFHandler<3>::active_cell_iterator &,
 AssemblyData::RightHandSide::ScratchData<3>        &,
 MeshWorker::CopyData<0,1,1>                        &) const;

template
void Solver<2>::
assemble_rhs_local_boundary
(const typename DoFHandler<2>::active_cell_iterator &,
 const unsigned int                                  ,
 AssemblyData::RightHandSide::ScratchData<2>        &,
 MeshWorker::CopyData<0,1,1>                        &) const;
template
void Solver<3>::
assemble_rhs_local_boundary
(const typename DoFHandler<3>::active_cell_iterator &,
 const unsigned int                                  ,
 AssemblyData::RightHandSide::ScratchData<3>        &,
 MeshWorker::CopyData<0,1,1>                        &) const;

}  // namespace BuoyantHydrodynamic

