/*
 * assemble_system.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <buoyant_hydrodynamic_solver.h>

namespace BuoyantHydrodynamic {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::assemble_system
(const bool use_homogeneous_constraints,
 const bool use_newton_linearization)
{
  if (this->verbose)
    this->pcout << "    Assemble linear system..." << std::endl;

  if (this->angular_velocity_ptr != nullptr)
    AssertThrow(this->rossby_number > 0.0,
                ExcMessage("Non-vanishing Rossby number is required if the angular "
                           "velocity vector is specified."));
  if (this->body_force_ptr != nullptr)
    AssertThrow(this->froude_number > 0.0,
                ExcMessage("Non-vanishing Froude number is required if the body "
                           "force is specified."));
  if (this->reference_field_ptr != nullptr)
    AssertThrow(this->gradient_scaling_number > 0.0,
                ExcMessage("Non-vanishing gradient scaling is required if the "
                           "reference density field is specified."));

  AssertThrow(gravity_field_ptr != nullptr,
              ExcMessage("For a buoyant fluid, the gravity field must be specified."));
  AssertThrow(this->froude_number > 0.0,
              ExcMessage("For a buoyant fluid, the Froude number must be specified."));
  AssertThrow(this->reynolds_number > 0.0,
              ExcMessage("The Reynolds number must not vanish."));

  TimerOutput::Scope timer_section(this->computing_timer, "Assemble system");

  const bool use_stress_form{this->viscous_term_weak_form
    == Hydrodynamic::ViscousTermWeakForm::stress};

  this->system_matrix = 0;
  this->system_rhs = 0;

  // initialize quadrature formula
  const unsigned int fe_degree{this->fe_system->degree};
  const QGauss<dim>   quadrature_formula(fe_degree + 1);

  // initialize face quadrature formula
  const QGauss<dim-1>   face_quadrature_formula(fe_degree + 1);

  // setup lambda functions for the local assembly operation
  using ScratchData = AssemblyData::Matrix::ScratchData<dim>;
  using CopyData = MeshWorker::CopyData<1, 1, 1>;
  using CellIteratorType = typename DoFHandler<dim>::active_cell_iterator;

  // setup the lambda function for the cell assembly
  auto cell_worker = [this, use_newton_linearization](
      const CellIteratorType &cell,
      ScratchData            &scratch,
      CopyData               &data)
      {
        this->assemble_system_local_cell(cell,
                                         scratch,
                                         data,
                                         use_newton_linearization);
      };

  // setup the lambda function for the boundary assembly
  auto boundary_worker = [this](
      const CellIteratorType &cell,
      const unsigned int      face_number,
      ScratchData            &scratch,
      CopyData               &data)
      {
        this->assemble_system_local_boundary(cell,
                                             face_number,
                                             scratch,
                                             data);
      };

  // setup the lambda function for the copy local to global operation
  auto copier = [this, use_homogeneous_constraints](
      const CopyData  &data)
      {
        this->copy_local_to_global_system(data, use_homogeneous_constraints);
      };

  // setup update flags
  UpdateFlags update_flags{update_values|
                           update_gradients|
                           update_JxW_values};
  if (this->body_force_ptr != nullptr ||
      this->background_velocity_ptr != nullptr ||
      gravity_field_ptr != nullptr)
    update_flags |= update_quadrature_points;
  if (this->stabilization & (apply_supg|apply_pspg))
    update_flags |= update_hessians;

  UpdateFlags face_update_flags = update_values|
                                  update_quadrature_points|
                                  update_JxW_values;
  if (this->include_boundary_stress_terms)
    face_update_flags |= update_gradients|
                         update_normal_vectors;
  if (!this->scalar_boundary_conditions.dirichlet_bcs.empty())
    face_update_flags |= update_normal_vectors;

  // initialize scratch and copy object
  ScratchData scratch(this->mapping,
                      *this->fe_system,
                      quadrature_formula,
                      update_flags,
                      face_quadrature_formula,
                      face_update_flags,
                      this->stabilization,
                      use_stress_form,
                      this->background_velocity_ptr != nullptr,
                      this->body_force_ptr != nullptr,
                      this->gravity_field_ptr != nullptr,
                      !this->velocity_boundary_conditions.neumann_bcs.empty()||
                      this->include_boundary_stress_terms,
                      this->source_term_ptr != nullptr,
                      !this->scalar_boundary_conditions.dirichlet_bcs.empty(),
                      this->reference_field_ptr != nullptr);
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

  this->system_matrix.compress(VectorOperation::add);
  this->system_rhs.compress(VectorOperation::add);
}


// explicit instantiation
template void Solver<2>::assemble_system(const bool, const bool);
template void Solver<3>::assemble_system(const bool, const bool);

}  // namespace BuoyantHydrodynamic

