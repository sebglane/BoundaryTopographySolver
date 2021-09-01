/*
 * apply_constraints.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <solver_base.h>

namespace TopographyProblem {

template <int dim>
void SolverBase<dim>::apply_hanging_node_constraints()
{
  DoFTools::make_hanging_node_constraints(dof_handler,
                                          nonzero_constraints);

  DoFTools::make_hanging_node_constraints(dof_handler,
                                          zero_constraints);
}



template <int dim>
void SolverBase<dim>::apply_periodicity_constraints
(std::vector<PeriodicBoundaryData<dim>> &periodic_bcs)
{
  std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
  periodicity_vector;

  for (auto const &periodic_bc : periodic_bcs)
    GridTools::collect_periodic_faces(dof_handler,
                                      periodic_bc.boundary_pair.first,
                                      periodic_bc.boundary_pair.second,
                                      periodic_bc.direction,
                                      periodicity_vector);

  DoFTools::make_periodicity_constraints<DoFHandler<dim>>(periodicity_vector,
                                                          nonzero_constraints);

  DoFTools::make_periodicity_constraints<DoFHandler<dim>>(periodicity_vector,
                                                          zero_constraints);

}



template <int dim>
void SolverBase<dim>::apply_dirichlet_constraints
(const typename BoundaryConditionsBase<dim>::BCMapping &dirichlet_bcs,
 const ComponentMask                                   &mask)
{
  std::map<types::boundary_id, const Function<dim> *> function_map;

  const unsigned int n_components{mask.n_selected_components()};

  for (const auto &[boundary_id, function]: dirichlet_bcs)
  {
    AssertDimension(function->n_components, n_components);
    function_map[boundary_id] = function.get();
  }

  VectorTools::interpolate_boundary_values(*mapping_ptr,
                                           dof_handler,
                                           function_map,
                                           nonzero_constraints,
                                           mask);

  function_map.clear();
  const Functions::ZeroFunction<dim>  zero_function(n_components);
  for (const auto &[boundary_id, function]: dirichlet_bcs)
  {
    function_map[boundary_id] = &zero_function;
  }
  VectorTools::interpolate_boundary_values(*mapping_ptr,
                                           dof_handler,
                                           function_map,
                                           zero_constraints,
                                           mask);
}



template <int dim>
void SolverBase<dim>::apply_normal_flux_constraints
(const typename BoundaryConditionsBase<dim>::BCMapping &normal_flux_bcs,
 const ComponentMask                                   &mask)
{
  std::map<types::boundary_id, const Function<dim> *> function_map;
  std::set<types::boundary_id>  boundary_id_set;

  const unsigned int n_components{mask.n_selected_components()};
  AssertDimension(n_components, dim);

  const unsigned int first_vector_component{mask.first_selected_component()};

  for (auto const &[boundary_id, function] : normal_flux_bcs)
  {
    function_map[boundary_id] = function.get();
    boundary_id_set.insert(boundary_id);
  }

  VectorTools::compute_nonzero_normal_flux_constraints(dof_handler,
                                                       first_vector_component,
                                                       boundary_id_set,
                                                       function_map,
                                                       nonzero_constraints,
                                                       *mapping_ptr);
  VectorTools::compute_no_normal_flux_constraints(dof_handler,
                                                  first_vector_component,
                                                  boundary_id_set,
                                                  zero_constraints,
                                                  *mapping_ptr);
}

// explicit instantiations
template void SolverBase<2>::apply_hanging_node_constraints();
template void SolverBase<3>::apply_hanging_node_constraints();

template void SolverBase<2>::apply_periodicity_constraints
(std::vector<PeriodicBoundaryData<2>> &);
template void SolverBase<3>::apply_periodicity_constraints
(std::vector<PeriodicBoundaryData<3>> &);

template void SolverBase<2>::apply_dirichlet_constraints
(const typename BoundaryConditionsBase<2>::BCMapping &, const ComponentMask &);
template void SolverBase<3>::apply_dirichlet_constraints
(const typename BoundaryConditionsBase<3>::BCMapping &, const ComponentMask &);

template void SolverBase<2>::apply_normal_flux_constraints
(const typename BoundaryConditionsBase<2>::BCMapping &, const ComponentMask &);
template void SolverBase<3>::apply_normal_flux_constraints
(const typename BoundaryConditionsBase<3>::BCMapping &, const ComponentMask &);

}  // namespace TopographyProblem
