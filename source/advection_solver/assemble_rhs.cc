/*
 * setup.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <advection_solver.h>

namespace Advection {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::assemble_rhs(const bool use_homogeneous_constraints)
{
  if (this->verbose)
    this->pcout << "    Assemble rhs..." << std::endl;

  AssertThrow(advection_field_ptr != nullptr,
              ExcMessage("The advection field must be specified."));

  if (reference_field_ptr != nullptr)
    AssertThrow(gradient_scaling_number > 0.0,
                ExcMessage("Non-vanishing gradient scaling number is required if"
                           " the reference field is specified."));

  TimerOutput::Scope timer_section(this->computing_timer, "Assemble rhs");

  this->system_rhs = 0;

  // initialize quadrature formula
  const QGauss<dim>   quadrature_formula(scalar_fe_degree + 1);

  // initialize face quadrature formula
  const QGauss<dim-1>   face_quadrature_formula(scalar_fe_degree + 1);

  // setup lambda functions for the local assembly operation
  using ScratchData = AssemblyData::RightHandSide::ScratchData<dim>;
  using CopyData = MeshWorker::CopyData<0, 1, 1>;
  using CellIteratorType = typename DoFHandler<dim>::active_cell_iterator;

  // setup the lambda function for the cell assembly
  auto cell_worker = [this](
      const CellIteratorType &cell,
      ScratchData            &scratch,
      CopyData               &data)
      {
        this->assemble_rhs_local_cell(cell,
                                      scratch,
                                      data);
      };

  // setup the lambda function for the boundary assembly
  auto boundary_worker = [this](
      const CellIteratorType &cell,
      const unsigned int      face_number,
      ScratchData            &scratch,
      CopyData               &data)
      {
        this->assemble_rhs_local_boundary(cell,
                                          face_number,
                                          scratch,
                                          data);
      };


  // setup the lambda function for the copy local to global operation
  auto copier = [this, use_homogeneous_constraints](
      const CopyData  &data)
      {
        this->copy_local_to_global_rhs(data, use_homogeneous_constraints);
      };

  // setup update flags
  const UpdateFlags update_flags{update_values|
                                 update_gradients|
                                 update_quadrature_points|
                                 update_JxW_values};
  const UpdateFlags face_update_flags{update_values|
                                      update_normal_vectors|
                                      update_quadrature_points|
                                      update_JxW_values};

  // initialize scratch and copy object
  ScratchData scratch(this->mapping,
                      *this->fe_system,
                      quadrature_formula,
                      update_flags,
                      face_quadrature_formula,
                      face_update_flags,
                      source_term_ptr != nullptr,
                      !scalar_boundary_conditions.dirichlet_bcs.empty(),
                      false,
                      reference_field_ptr != nullptr);
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

// explicit instantiation
template void Solver<2>::assemble_rhs(const bool);
template void Solver<3>::assemble_rhs(const bool);

}  // namespace Advection

