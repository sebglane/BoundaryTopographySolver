/*
 * refine_mesh.cc
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

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

namespace internal
{

template <int dim>
void refine_and_coarsen
(Triangulation<dim>  &tria,
 const Vector<float> &criteria,
 const double         top_fraction,
 const double         bottom_fraction)
{
  GridRefinement::refine_and_coarsen_fixed_fraction(tria,
                                                    criteria,
                                                    top_fraction,
                                                    bottom_fraction);
}



template <int dim>
void refine_and_coarsen
(parallel::distributed::Triangulation<dim>   &tria,
 const Vector<float> &criteria,
 const double         top_fraction,
 const double         bottom_fraction)
{
  parallel::distributed::
  GridRefinement::refine_and_coarsen_fixed_fraction(tria,
                                                    criteria,
                                                    top_fraction,
                                                    bottom_fraction);
}

}  // namespace internal



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
                                       estimated_error_per_cell,
                                       ComponentMask(),
                                       nullptr,
                                       0,
                                       triangulation.locally_owned_subdomain());
    // set refinement flags
    internal::refine_and_coarsen(triangulation,
                                 estimated_error_per_cell,
                                 refinement_parameters.cell_fraction_to_refine,
                                 refinement_parameters.cell_fraction_to_coarsen);


    // Clear refinement flags if refinement level exceeds maximum
    if (triangulation.n_levels() > refinement_parameters.n_maximum_levels)
      for (const auto &cell: triangulation.active_cell_iterators_on_level(refinement_parameters.n_maximum_levels))
        cell->clear_refine_flag();

    // Clear coarsen flags if level decreases minimum
    for (const auto &cell: triangulation.active_cell_iterators_on_level(refinement_parameters.n_minimum_levels))
      cell->clear_coarsen_flag();

    // Count number of cells to be refined and coarsened
    unsigned int cell_counts[2] = {0, 0};
    for (const auto &cell: triangulation.active_cell_iterators())
        if (cell->is_locally_owned() && cell->refine_flag_set())
          cell_counts[0] += 1;
        else if (cell->is_locally_owned() && cell->coarsen_flag_set())
          cell_counts[1] += 1;

    unsigned int global_cell_counts[2];
    Utilities::MPI::sum(cell_counts,
                        triangulation.get_communicator(),
                        global_cell_counts);

    pcout << "    Number of cells set for refinement: " << global_cell_counts[0] << std::endl
          << "    Number of cells set for coarsening: " << global_cell_counts[1] << std::endl;

  }
  else
    triangulation.set_all_refine_flags();

  execute_mesh_refinement();

}



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void Solver<dim, TriangulationType, LinearAlgebraContainer>::execute_mesh_refinement()
{
  // preparing temperature solution transfer
  using VectorType = typename LinearAlgebraContainer::vector_type;
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

    container.distribute_constraints(tmp_solution[0],
                                     nonzero_constraints);
    container.present_solution = tmp_solution[0];
  }
}



template <>
void Solver<2, parallel::distributed::Triangulation<2>, TrilinosContainer>::
execute_mesh_refinement()
{
  constexpr int dim{2};
  // preparing solution transfer
  using VectorType = TrilinosContainer::vector_type;
  typename parallel::distributed::SolutionTransfer<dim, VectorType>
  solution_transfer(dof_handler);

  std::vector<const VectorType *> x_solution(1);
  x_solution[0] = &container.present_solution;

  // preparing triangulation refinement
  triangulation.prepare_coarsening_and_refinement();
  solution_transfer.prepare_for_coarsening_and_refinement(x_solution);

  // refine triangulation
  triangulation.execute_coarsening_and_refinement();

  // setup dofs and constraints on refined mesh
  this->setup_dofs();

  // transfer of solution
  {
    VectorType tmp_solution;
    tmp_solution.reinit(container.system_rhs);

    std::vector<VectorType *> tmp(1);
    tmp[0] = &tmp_solution;
    solution_transfer.interpolate(tmp_solution);

    container.distribute_constraints(tmp_solution, nonzero_constraints);

    container.present_solution = tmp_solution;

    container.distribute_constraints(container.present_solution,
                                     nonzero_constraints);
  }
}



template <>
void Solver<3, parallel::distributed::Triangulation<3>, TrilinosContainer>::
execute_mesh_refinement()
{
  constexpr int dim{3};
  // preparing solution transfer
  using VectorType = TrilinosContainer::vector_type;
  typename parallel::distributed::SolutionTransfer<dim, VectorType>
  solution_transfer(dof_handler);

  std::vector<const VectorType *> x_solution(1);
  x_solution[0] = &container.present_solution;

  // preparing triangulation refinement
  triangulation.prepare_coarsening_and_refinement();
  solution_transfer.prepare_for_coarsening_and_refinement(x_solution);

  // refine triangulation
  triangulation.execute_coarsening_and_refinement();

  // setup dofs and constraints on refined mesh
  this->setup_dofs();

  pcout << container.system_matrix.m() << ", " << container.system_matrix.n() << std::endl;

  // transfer of solution
  {
    VectorType tmp_solution;
    tmp_solution.reinit(container.system_rhs);

    std::vector<VectorType *> tmp(1);
    tmp[0] = &tmp_solution;
    solution_transfer.interpolate(tmp_solution);

    container.distribute_constraints(tmp_solution, nonzero_constraints);

    container.present_solution = tmp_solution;

    container.distribute_constraints(container.present_solution,
                                     nonzero_constraints);
  }
}


// explicit instantiations
template
void
Solver<2>::
execute_mesh_refinement();
template
void
Solver<3>::
execute_mesh_refinement();

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


