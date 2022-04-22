/*
 * assemble_rhs.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <hydrodynamic_solver.h>

namespace Hydrodynamic {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::assemble_rhs(const bool use_homogeneous_constraints)
{
  if (this->verbose)
    this->pcout << "    Assemble rhs..." << std::endl;

  if (angular_velocity_ptr != nullptr)
    AssertThrow(rossby_number > 0.0,
                ExcMessage("Non-vanishing Rossby number is required if the angular "
                           "velocity vector is specified."));
  if (body_force_ptr != nullptr)
    AssertThrow(froude_number > 0.0,
                ExcMessage("Non-vanishing Froude number is required if the body "
                           "force is specified."));
  AssertThrow(reynolds_number != 0.0,
              ExcMessage("The Reynolds must not vanish (stabilization is not "
                         "implemented yet)."));

  TimerOutput::Scope timer_section(this->computing_timer, "Assemble rhs");

  const bool use_stress_form{viscous_term_weak_form == ViscousTermWeakForm::stress};

  this->system_rhs = 0;

  // initialize quadrature formula
  const QGauss<dim>   quadrature_formula(velocity_fe_degree + 1);

  // initialize face quadrature formula
  const QGauss<dim-1>   face_quadrature_formula(velocity_fe_degree + 1);

  // setup lambda functions for the local assembly operation
  using ScratchData = AssemblyData::RightHandSide::ScratchData<dim>;
  using CopyData = MeshWorker::CopyData<0, 1, 1>;
  using CellIteratorType = typename DoFHandler<dim>::active_cell_iterator;

  // setup the lambda function for the cell assembly
  auto cell_worker = [this, use_stress_form](
      const CellIteratorType &cell,
      ScratchData            &scratch,
      CopyData               &data)
      {
        this->assemble_rhs_local_cell(cell,
                                      scratch,
                                      data,
                                      use_stress_form);
      };

  // setup the lambda function for the boundary assembly
  auto boundary_worker = [this, use_stress_form](
      const CellIteratorType &cell,
      const unsigned int      face_number,
      ScratchData            &scratch,
      CopyData               &data)
      {
        this->assemble_rhs_local_boundary(cell,
                                          face_number,
                                          scratch,
                                          data,
                                          use_stress_form);
      };


  // setup the lambda function for the copy local to global operation
  auto copier = [this, use_homogeneous_constraints](
      const CopyData  &data)
      {
        this->copy_local_to_global_rhs(data, use_homogeneous_constraints);
      };

  // setup update flags
  UpdateFlags update_flags{update_values|
                           update_gradients|
                           update_JxW_values};
  if (body_force_ptr != nullptr || background_velocity_ptr != nullptr)
    update_flags |= update_quadrature_points;
  if (stabilization & (apply_supg|apply_pspg))
    update_flags |= update_hessians;

  UpdateFlags face_update_flags{update_values|
                                update_quadrature_points|
                                update_JxW_values};
  if (include_boundary_stress_terms)
    face_update_flags |= update_gradients|
                         update_normal_vectors;

  // initialize scratch and copy object
  ScratchData scratch(this->mapping,
                      *this->fe_system,
                      quadrature_formula,
                      update_flags,
                      face_quadrature_formula,
                      face_update_flags,
                      stabilization,
                      use_stress_form,
                      background_velocity_ptr != nullptr,
                      body_force_ptr != nullptr,
                      !velocity_boundary_conditions.neumann_bcs.empty() ||
                      include_boundary_stress_terms);
  CopyData  copy(this->fe_system->n_dofs_per_cell());

  // mesh worker
  MeshWorker::mesh_loop(this->dof_handler.active_cell_iterators(),
                        cell_worker,
                        copier,
                        scratch,
                        copy,
                        MeshWorker::AssembleFlags::assemble_own_cells|
                        MeshWorker::AssembleFlags::assemble_boundary_faces,
                        boundary_worker);

  this->system_rhs.compress(VectorOperation::add);
}




template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::legacy_copy_local_to_global_rhs
(const AssemblyBaseData::RightHandSide::Copy  &data,
 const bool use_homogeneous_constraints)
{
  const AffineConstraints<double> &constraints =
      (use_homogeneous_constraints ? this->zero_constraints: this->nonzero_constraints);

  constraints.distribute_local_to_global(data.local_rhs,
                                         data.local_dof_indices,
                                         this->system_rhs);
}


// explicit instantiation
template
void
Solver<2>::
legacy_copy_local_to_global_rhs
(const AssemblyBaseData::RightHandSide::Copy  &data,
 const bool use_homogeneous_constraints);
template
void
Solver<3>::
legacy_copy_local_to_global_rhs
(const AssemblyBaseData::RightHandSide::Copy  &data,
 const bool use_homogeneous_constraints);

template void Solver<2>::assemble_rhs(const bool);
template void Solver<3>::assemble_rhs(const bool);



}  // namespace Hydrodynamic

