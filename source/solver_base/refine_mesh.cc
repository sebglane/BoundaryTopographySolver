/*
 * refine_mesh.cc
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <solver_base.h>

#include <vector>

namespace SolverBase {

using TrilinosContainer = LinearAlgebraContainer<TrilinosWrappers::MPI::Vector,
                                                 TrilinosWrappers::SparseMatrix,
                                                 TrilinosWrappers::SparsityPattern>;



template <int dim>
using ParallelTriangulation =  parallel::distributed::Triangulation<dim>;



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void Solver<dim, TriangulationType, LinearAlgebraContainer>::refine_mesh()
{
  pcout << "Mesh refinement..." << std::endl;

  using VectorType = typename LinearAlgebraContainer::vector_type;

  TimerOutput::Scope timer_section(computing_timer, "Refine mesh");

  if (refinement_parameters.adaptive_mesh_refinement)
  {
    // error estimation
    Vector<float>   estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(mapping,
                                       dof_handler,
                                       QGauss<dim-1>(fe_system->degree + 1),
                                       std::map<types::boundary_id, const Function<dim> *>(),
                                       container.present_solution,
                                       estimated_error_per_cell);
    // set refinement flags
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      refinement_parameters.cell_fraction_to_refine,
                                                      refinement_parameters.cell_fraction_to_coarsen);

    // Clear refinement flags if refinement level exceeds maximum
    if (triangulation.n_levels() > refinement_parameters.n_maximum_levels)
      for (const auto &cell: triangulation.active_cell_iterators_on_level(refinement_parameters.n_maximum_levels))
        if (cell->is_locally_owned())
          cell->clear_refine_flag();

    // Clear coarsen flags if level decreases minimum
    for (const auto &cell: triangulation.active_cell_iterators_on_level(refinement_parameters.n_minimum_levels))
      if (cell->is_locally_owned())
        cell->clear_coarsen_flag();

    // Count number of cells to be refined and coarsened
    unsigned int cell_counts[2] = {0, 0};
    for (const auto &cell: triangulation.active_cell_iterators())
        if (cell->is_locally_owned() && cell->refine_flag_set())
          cell_counts[0] += 1;
        else if (cell->is_locally_owned() && cell->coarsen_flag_set())
          cell_counts[1] += 1;

    pcout << "    Number of cells set for refinement: " << cell_counts[0] << std::endl
          << "    Number of cells set for coarsening: " << cell_counts[1] << std::endl;

  }
  else
    triangulation.set_all_refine_flags();

  // preparing temperature solution transfer
  std::vector<VectorType> x_solution(1);
  x_solution[0] = container.present_solution;
  SolutionTransfer<dim, VectorType> solution_transfer(dof_handler);

  // preparing triangulation refinement
  triangulation.prepare_coarsening_and_refinement();
  solution_transfer.prepare_for_coarsening_and_refinement(x_solution);

  // refine triangulation
  triangulation.execute_coarsening_and_refinement();

  // setup dofs and constraints on refined mesh
  this->setup_dofs();

  // transfer of solution
  {
    std::vector<VectorType> tmp_solution(1);
    tmp_solution[0].reinit(container.present_solution);
    solution_transfer.interpolate(x_solution, tmp_solution);

    container.present_solution = tmp_solution[0];

    container.distribute_constraints(container.present_solution,
                                     nonzero_constraints);
  }
}

// explicit instantiations
template
void
Solver<2>::
refine_mesh();
template
void
Solver<3>::
refine_mesh();

template
void
Solver<2, ParallelTriangulation<2>, TrilinosContainer>::
refine_mesh();
template
void
Solver<3, ParallelTriangulation<3>, TrilinosContainer>::
refine_mesh();

}  // namespace SolverBase


