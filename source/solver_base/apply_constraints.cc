/*
 * apply_constraints.cc
 *
 *  Created on: Aug 31, 2021
 *      Author: sg
 */
#include <base.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <functional>

namespace Base {


template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
apply_hanging_node_constraints()
{
  DoFTools::make_hanging_node_constraints(dof_handler,
                                          nonzero_constraints);

  DoFTools::make_hanging_node_constraints(dof_handler,
                                          zero_constraints);
}



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
apply_periodicity_constraints
(std::vector<PeriodicBoundaryData<dim>> &periodic_bcs,
 const ComponentMask                    &mask)
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
                                                   nonzero_constraints,
                                                   mask);

  DoFTools::make_periodicity_constraints<dim, dim>(periodicity_vector,
                                                   zero_constraints,
                                                   mask);

}



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
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
      AssertThrow(mask[first_selected_component + d] == true,
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



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
apply_normal_flux_constraints
(const typename BoundaryConditionsBase<dim>::BCMapping &normal_flux_bcs,
 const ComponentMask                                   &mask)
{
  std::map<types::boundary_id, const Function<dim> *> function_map;
  std::set<types::boundary_id>  boundary_id_set;

  AssertDimension(mask.n_selected_components(), dim);

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



template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
apply_tangential_flux_constraints
(const typename BoundaryConditionsBase<dim>::BCMapping &tangential_flux_bcs,
 const ComponentMask                                   &mask)
{
  std::map<types::boundary_id, const Function<dim> *> function_map;
  std::set<types::boundary_id>  boundary_id_set;

  AssertDimension(mask.n_selected_components(), dim);

  const unsigned int first_vector_component{mask.first_selected_component()};

  for (auto const &[boundary_id, function] : tangential_flux_bcs)
  {
    function_map[boundary_id] = function.get();
    boundary_id_set.insert(boundary_id);
  }

  VectorTools::compute_nonzero_tangential_flux_constraints(dof_handler,
                                                           first_vector_component,
                                                           boundary_id_set,
                                                           function_map,
                                                           nonzero_constraints,
                                                           mapping);
  function_map.clear();
  boundary_id_set.clear();
  const Functions::ZeroFunction<dim>  zero_function(fe_system->n_components());

  for (const auto &[boundary_id, function]: tangential_flux_bcs)
  {
    function_map[boundary_id] = &zero_function;
    boundary_id_set.insert(boundary_id);
  }

  VectorTools::compute_nonzero_tangential_flux_constraints(dof_handler,
                                                           first_vector_component,
                                                           boundary_id_set,
                                                           function_map,
                                                           zero_constraints,
                                                           mapping);
}




template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
apply_mean_value_constraint
(const ComponentMask &mask,
 const double         mean_value)
{
  this->pcout << "    Apply mean value constraint..." << std::endl;
  Assert(mask.size() == this->fe_system->n_components(),
         ExcDimensionMismatch(mask.size(), this->fe_system->n_components()));

  Assert(mask.n_selected_components() == 1,
         ExcMessage("Only one component of the solution can be selected "
                    "for constraining the mean value."));

  const unsigned int selected_component{mask.first_selected_component()};
  AssertThrow(!component_mean_values.contains(selected_component),
              ExcMessage("The mean value of the selected component was already "
                         "specified."));

  const IndexSet  boundary_dofs{DoFTools::extract_boundary_dofs(this->dof_handler,
                                                                mask)};

  // Look for an admissible local degree of freedom to constrain
  types::global_dof_index local_idx = numbers::invalid_dof_index;
  IndexSet::ElementIterator idx = boundary_dofs.begin();
  IndexSet::ElementIterator endidx = boundary_dofs.end();
  for(; idx != endidx; ++idx)
    if ((this->zero_constraints.can_store_line(*idx) &&
         !this->zero_constraints.is_constrained(*idx)) &&
        (this->nonzero_constraints.can_store_line(*idx) &&
                     !this->nonzero_constraints.is_constrained(*idx)))
    {
      local_idx = *idx;
      break;
    }

  // choose the degree of freedom with the smallest index. If no
  // admissible degree of freedom was found in a given processor, its
  // value is set the number of degree of freedom
  types::global_dof_index global_idx{numbers::invalid_dof_index};

  // ensure that at least one processor found things
  if (const parallel::TriangulationBase<dim> *tria_ptr =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&this->triangulation);
      tria_ptr != nullptr)
  {
    global_idx = Utilities::MPI::min((local_idx != numbers::invalid_dof_index)? local_idx: this->dof_handler.n_dofs(),
                                     tria_ptr->get_communicator());
  }
  else
    global_idx = local_idx;
  AssertThrow(global_idx != numbers::invalid_dof_index,
              ExcMessage("Invalid DoF index when setting mean value constraint "
                         "on the pressure component."));

  // check that an admissible degree of freedom was found
  AssertThrow(global_idx < dof_handler.n_dofs(),
              ExcMessage("Error, couldn't find a DoF to constrain."));

  // set the degree of freedom to zero
  if (zero_constraints.can_store_line(global_idx))
  {
    AssertThrow(!zero_constraints.is_constrained(global_idx),
                ExcInternalError());
    zero_constraints.add_line(global_idx);
  }
  if (nonzero_constraints.can_store_line(global_idx))
  {
    AssertThrow(!nonzero_constraints.is_constrained(global_idx),
                ExcInternalError());
    nonzero_constraints.add_line(global_idx);
  }

  // add mean value
  component_mean_values[selected_component] = mean_value;

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

template void Solver<2>::apply_periodicity_constraints
(std::vector<PeriodicBoundaryData<2>> &,
 const ComponentMask                  &);
template void Solver<3>::apply_periodicity_constraints
(std::vector<PeriodicBoundaryData<3>> &,
 const ComponentMask                  &);

template void Solver<2>::apply_dirichlet_constraints
(const typename BoundaryConditionsBase<2>::BCMapping &, const ComponentMask &);
template void Solver<3>::apply_dirichlet_constraints
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
Solver<2>::
apply_tangential_flux_constraints
(const typename BoundaryConditionsBase<2>::BCMapping &, const ComponentMask &);
template
void
Solver<3>::
apply_tangential_flux_constraints
(const typename BoundaryConditionsBase<3>::BCMapping &, const ComponentMask &);

template
void
Solver<2>::
apply_mean_value_constraint
(const ComponentMask &, const double);
template
void
Solver<3>::
apply_mean_value_constraint
(const ComponentMask &, const double);

}  // namespace Base
