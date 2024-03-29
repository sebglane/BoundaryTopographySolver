/*
 * refine_mesh.cc
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */
#include <base.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <vector>

namespace Base {

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



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::refine_mesh()
{
  pcout << "Mesh refinement..." << std::endl;

  TimerOutput::Scope timer_section(computing_timer, "Refine mesh");

  if (refinement_parameters.adaptive_mesh_refinement)
  {
    // error estimation
    Vector<float>   estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(mapping,
                                       dof_handler,
                                       QGauss<dim-1>(fe_system->degree + 1),
                                       std::map<types::boundary_id, const Function<dim> *>(),
                                       this->present_solution,
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



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::execute_mesh_refinement()
{
  // preparing temperature solution transfer
  std::vector<VectorType> x_solution(1);
  x_solution[0] = this->present_solution;
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
    tmp_solution[0].reinit(this->present_solution);
    solution_transfer.interpolate(x_solution, tmp_solution);

    nonzero_constraints.distribute(tmp_solution[0]);

    this->present_solution = tmp_solution[0];
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


}  // namespace Base


