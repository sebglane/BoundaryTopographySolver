/*
 * assemble_system.cc
 *
 *  Created on: Aug 23, 2022
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <magnetic_induction_solver.h>

namespace MagneticInduction {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::assemble_system
(const bool use_homogeneous_constraints,
 const bool use_newton_linearization)
{
  if (this->verbose)
    this->pcout << "    Assemble linear system..." << std::endl;

  AssertThrow(magnetic_reynolds_number > 0.0,
              ExcMessage("The magnetic Reynolds number must not vanish."));

  TimerOutput::Scope timer_section(this->computing_timer, "Assemble system");

  this->system_matrix = 0;
  this->system_rhs = 0;

  // initialize quadrature formula
  const QGauss<dim>   quadrature_formula(magnetic_fe_degree + 1);

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
  if (velocity_field_ptr != nullptr || background_magnetic_field_ptr != nullptr)
    update_flags |= update_quadrature_points;

  // initialize scratch and copy object
  ScratchData scratch(this->mapping,
                      *this->fe_system,
                      quadrature_formula,
                      update_flags,
                      velocity_field_ptr != nullptr,
                      background_magnetic_field_ptr != nullptr);
  CopyData  copy(this->fe_system->n_dofs_per_cell());

  // mesh worker
  MeshWorker::mesh_loop(this->dof_handler.active_cell_iterators(),
                        cell_worker,
                        copier,
                        scratch,
                        copy,
                        MeshWorker::AssembleFlags::assemble_own_cells);

  this->system_matrix.compress(VectorOperation::add);
  this->system_rhs.compress(VectorOperation::add);
}



// explicit instantiation
template
void
Solver<2>::
assemble_system
(const bool, const bool);
template
void
Solver<3>::
assemble_system(const bool, const bool);



}  // namespace MagneticInduction

