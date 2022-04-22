/*
 * assembly_base_data.cc
 *
 *  Created on: Sep 25, 2021
 *      Author: sg
 */
#include <legacy_assembly_base_data.h>

namespace AssemblyBaseData
{

namespace Matrix
{

Copy::Copy(const unsigned int dofs_per_cell)
:
local_matrix(dofs_per_cell, dofs_per_cell),
local_rhs(dofs_per_cell),
local_dof_indices(dofs_per_cell),
dofs_per_cell(dofs_per_cell)
{}



Copy::Copy(const Copy &data)
:
local_matrix(data.local_matrix),
local_rhs(data.local_rhs),
local_dof_indices(data.local_dof_indices),
dofs_per_cell(data.dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const FiniteElement<dim>  &fe,
 const UpdateFlags         update_flags)
:
fe_values(mapping,
          fe,
          quadrature_formula,
          update_flags),
dofs_per_cell(fe.n_dofs_per_cell()),
n_q_points(quadrature_formula.size())
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
fe_values(data.fe_values.get_mapping(),
          data.fe_values.get_fe(),
          data.fe_values.get_quadrature(),
          data.fe_values.get_update_flags()),
dofs_per_cell(data.dofs_per_cell),
n_q_points(data.n_q_points)
{}

} // namespace Matrix

namespace RightHandSide
{

Copy::Copy(const unsigned int dofs_per_cell)
:
local_rhs(dofs_per_cell),
local_dof_indices(dofs_per_cell),
dofs_per_cell(dofs_per_cell)
{}



Copy::Copy(const Copy &data)
:
local_rhs(data.local_rhs),
local_dof_indices(data.local_dof_indices),
dofs_per_cell(data.dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const FiniteElement<dim>  &fe,
 const UpdateFlags         update_flags)
:
fe_values(mapping,
          fe,
          quadrature_formula,
          update_flags),
dofs_per_cell(fe.n_dofs_per_cell()),
n_q_points(quadrature_formula.size())
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
fe_values(data.fe_values.get_mapping(),
          data.fe_values.get_fe(),
          data.fe_values.get_quadrature(),
          data.fe_values.get_update_flags()),
dofs_per_cell(data.dofs_per_cell),
n_q_points(data.n_q_points)
{}

} // namespace RightHandSide

} // namespace AssemblyData

template struct AssemblyBaseData::Matrix::Scratch<2>;
template struct AssemblyBaseData::Matrix::Scratch<3>;

template struct AssemblyBaseData::RightHandSide::Scratch<2>;
template struct AssemblyBaseData::RightHandSide::Scratch<3>;
