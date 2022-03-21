/*
 * apply_constraints.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <solver_base.h>

#include <functional>

namespace SolverBase {

using TrilinosContainer = LinearAlgebraContainer<TrilinosWrappers::MPI::Vector,
                                                 TrilinosWrappers::SparseMatrix,
                                                 TrilinosWrappers::SparsityPattern>;



template <int dim>
using ParallelTriangulation =  parallel::distributed::Triangulation<dim>;



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void Solver<dim, TriangulationType, LinearAlgebraContainer>::
apply_hanging_node_constraints()
{
  DoFTools::make_hanging_node_constraints(dof_handler,
                                          nonzero_constraints);

  DoFTools::make_hanging_node_constraints(dof_handler,
                                          zero_constraints);
}



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void Solver<dim, TriangulationType, LinearAlgebraContainer>::
apply_periodicity_constraints
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

  DoFTools::make_periodicity_constraints<dim, dim>(periodicity_vector,
                                                   nonzero_constraints);

  DoFTools::make_periodicity_constraints<dim, dim>(periodicity_vector,
                                                   zero_constraints);

}



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void Solver<dim, TriangulationType, LinearAlgebraContainer>::
apply_dirichlet_constraints
(const typename BoundaryConditionsBase<dim>::BCMapping &dirichlet_bcs,
 const ComponentMask                                   &mask)
{
  std::map<types::boundary_id, const Function<dim> *> function_map;

  const unsigned int n_components{mask.n_selected_components()};
  const unsigned int first_selected_component{mask.first_selected_component()};

  Assert(n_components == 1 || n_components==dim,
         ExcMessage("Expected either one or dim selected components."));

  if (n_components==dim)
    for (unsigned d=0; d<dim; ++d)
      AssertThrow(mask[first_selected_component+ d] == true,
                  ExcMessage("Expected a sequence of dim selected components in "
                             "component mask"));

  if (n_components != fe_system->n_components())
  {
    std::map<types::boundary_id, FunctionFromFunctionObjects<dim>> aux_function_map;

    Functions::ZeroFunction<dim>  zero_function;
    auto zero_function_value = [&](const Point<dim> &point){ return zero_function.value(point); };
    for (const auto &[boundary_id, function]: dirichlet_bcs)
    {
      AssertDimension(function->n_components, n_components);

      std::vector<std::function<double(const Point<dim> &)>> function_values;

      for (std::size_t i=0; i<fe_system->n_components(); )
      {
        if (mask[i] == false)
        {
          function_values.push_back(zero_function_value);
          ++i;
        }
        else
        {
          if (n_components == 1)
          {
            auto fun = [&](const Point<dim> &point){ return function->value(point); };
            function_values.push_back(fun);
            ++i;
          }
          else
            for (unsigned int d=0; d<dim; ++d, ++i)
            {
              auto fun = [&, d](const Point<dim> &point){ return function->value(point, d); };
              function_values.push_back(fun);
            }
        }
      }
      AssertDimension(function_values.size(), fe_system->n_components());

      FunctionFromFunctionObjects<dim>  auxiliary_function(fe_system->n_components());
      auxiliary_function.set_function_values(function_values);

      aux_function_map.insert
      (std::pair<types::boundary_id, FunctionFromFunctionObjects<dim>>(boundary_id, auxiliary_function));
    }

    for (const auto &bc: dirichlet_bcs)
      function_map[bc.first] = &aux_function_map[bc.first];

    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             function_map,
                                             nonzero_constraints,
                                             mask);
  }
  else
  {
    for (const auto &[boundary_id, function]: dirichlet_bcs)
    {
      AssertDimension(function->n_components, n_components);

      function_map[boundary_id] = function.get();
    }
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             function_map,
                                             nonzero_constraints,
                                             mask);
  }

  function_map.clear();
  {
    const Functions::ZeroFunction<dim>  zero_function(fe_system->n_components());
    for (const auto &[boundary_id, function]: dirichlet_bcs)
    {
      function_map[boundary_id] = &zero_function;
    }
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             function_map,
                                             zero_constraints,
                                             mask);
  }
}



template <int dim, typename TriangulationType, typename LinearAlgebraContainer>
void Solver<dim, TriangulationType, LinearAlgebraContainer>::
apply_normal_flux_constraints
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
                                                       mapping);
  VectorTools::compute_no_normal_flux_constraints(dof_handler,
                                                  first_vector_component,
                                                  boundary_id_set,
                                                  zero_constraints,
                                                  mapping);
}

// explicit instantiations
template
void
Solver<2>::
apply_hanging_node_constraints();
template
void
Solver<3>::
apply_hanging_node_constraints();

template
void
Solver<2, ParallelTriangulation<2>, TrilinosContainer>::
apply_hanging_node_constraints();
template
void
Solver<3, ParallelTriangulation<3>, TrilinosContainer>::
apply_hanging_node_constraints();


template void Solver<2>::apply_periodicity_constraints
(std::vector<PeriodicBoundaryData<2>> &);
template void Solver<3>::apply_periodicity_constraints
(std::vector<PeriodicBoundaryData<3>> &);

template
void
Solver<2, ParallelTriangulation<2>, TrilinosContainer>::
apply_periodicity_constraints
(std::vector<PeriodicBoundaryData<2>> &);
template
void
Solver<3, ParallelTriangulation<3>, TrilinosContainer>::
apply_periodicity_constraints
(std::vector<PeriodicBoundaryData<3>> &);

template void Solver<2>::apply_dirichlet_constraints
(const typename BoundaryConditionsBase<2>::BCMapping &, const ComponentMask &);
template void Solver<3>::apply_dirichlet_constraints
(const typename BoundaryConditionsBase<3>::BCMapping &, const ComponentMask &);

template
void
Solver<2, ParallelTriangulation<2>, TrilinosContainer>::
apply_dirichlet_constraints
(const typename BoundaryConditionsBase<2>::BCMapping &, const ComponentMask &);
template
void
Solver<3, ParallelTriangulation<3>, TrilinosContainer>::
apply_dirichlet_constraints
(const typename BoundaryConditionsBase<3>::BCMapping &, const ComponentMask &);


template
void
Solver<2>::
apply_normal_flux_constraints
(const typename BoundaryConditionsBase<2>::BCMapping &, const ComponentMask &);
template
void
Solver<3>::
apply_normal_flux_constraints
(const typename BoundaryConditionsBase<3>::BCMapping &, const ComponentMask &);

template
void
Solver<2, ParallelTriangulation<2>, TrilinosContainer>::
apply_normal_flux_constraints
(const typename BoundaryConditionsBase<2>::BCMapping &, const ComponentMask &);
template
void
Solver<3, ParallelTriangulation<3>, TrilinosContainer>::
apply_normal_flux_constraints
(const typename BoundaryConditionsBase<3>::BCMapping &, const ComponentMask &);

}  // namespace SolverBase
