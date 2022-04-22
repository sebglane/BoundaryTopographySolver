#ifndef INCLUDE_LEGACY_ASSEMBLY_BASE_DATA_H_
#define INCLUDE_LEGACY_ASSEMBLY_BASE_DATA_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

namespace AssemblyBaseData
{

using namespace dealii;

namespace Matrix
{

struct Copy
{
  Copy(const unsigned int dofs_per_cell);

  Copy(const Copy &data);

  FullMatrix<double>                    local_matrix;

  Vector<double>                        local_rhs;

  std::vector<types::global_dof_index> local_dof_indices;

  types::global_dof_index dofs_per_cell;

};



template <int dim>
struct Scratch
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);

  Scratch(const Scratch<dim>        &data);

  FEValues<dim>       fe_values;

  types::global_dof_index dofs_per_cell;

  const unsigned int  n_q_points;

};

} // namespace Matrix

namespace RightHandSide
{

struct Copy
{
  Copy(const unsigned int dofs_per_cell);

  Copy(const Copy &data);

  Vector<double>                        local_rhs;

  std::vector<types::global_dof_index> local_dof_indices;

  types::global_dof_index dofs_per_cell;
};



template <int dim>
struct Scratch
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);

  Scratch(const Scratch<dim>        &data);

  FEValues<dim>       fe_values;

  types::global_dof_index dofs_per_cell;

  const unsigned int  n_q_points;

};

} // namespace RightHandSide

} // namespace AssemblyBaseData

#endif /* INCLUDE_LEGACY_ASSEMBLY_BASE_DATA_H_ */
