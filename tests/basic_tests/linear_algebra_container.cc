/*
 * linear_algebra_container.cc
 *
 *  Created on: Oct 19, 2021
 *      Author: sg
 */
#include <deal.II/base/function_lib.h>
#include <deal.II/base/table.h>
#include <deal.II/base/logstream.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>

namespace TestSpace
{

using namespace dealii;

template <typename VectorType>
void
set_block
(VectorType         &vector,
 const std::vector<types::global_dof_index> &dofs_per_block,
 const std::vector<IndexSet>                &locally_owned_dofs_per_block,
 const unsigned int  block,
 const double        value);



template <>
void
set_block
(Vector<double>      &vector,
 const std::vector<types::global_dof_index> &dofs_per_block,
 const std::vector<IndexSet>                &locally_owned_dofs_per_block,
 const unsigned int  block,
 const double        value)
{
  const std::size_t n_blocks{dofs_per_block.size()};
  AssertThrow(block < n_blocks, ExcInternalError());

  const types::global_dof_index n_dofs
    = std::accumulate(dofs_per_block.begin(),
                      dofs_per_block.end(),
                      types::global_dof_index(0));
  AssertDimension(vector.size(), n_dofs);

  std::vector<types::global_dof_index> dof_index_shifts(n_blocks, 0);
  {
    auto it = dof_index_shifts.begin();
    std::advance(it, 1);

    std::partial_sum(dofs_per_block.begin(),
                     dofs_per_block.end(),
                     it);
  }

  IndexSet  index_set(n_dofs);
  index_set.add_indices(locally_owned_dofs_per_block[block],
                        dof_index_shifts[block]);

  for (const auto idx: index_set)
  {
    Assert(idx < vector.size(), ExcInternalError());
    vector[idx] = value;
  }
}




template <int dim>
void
test_dealii_vector
(const unsigned int n_blocks,
 const unsigned int degree = 1)
{
  Triangulation<dim>            tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(3);

  FESystem<dim> fe_system(FE_Q<dim>(degree), n_blocks);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_system);
  DoFRenumbering::block_wise(dof_handler);

  std::cout << "    Number of active cells: "
            << tria.n_active_cells()
            << std::endl
            << "    Number of total degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  const std::vector<types::global_dof_index> dofs_per_block
    = DoFTools::count_dofs_per_fe_block(dof_handler);

  std::cout << "    Number of degrees of freedom per block: ";
  for (const auto n: dofs_per_block)
    std::cout << n << ", ";
  std::cout << std::endl;

  IndexSet              locally_owned_dofs, locally_relevant_dofs;
  std::vector<IndexSet> locally_owned_dofs_per_block, locally_relevant_dofs_per_block;
  {
    std::vector<types::global_dof_index> accumulated_dofs_per_block(dofs_per_block.size() + 1, 0);
    {
      auto it = accumulated_dofs_per_block.begin();
      std::advance(it, 1);
      std::partial_sum(dofs_per_block.begin(),
                       dofs_per_block.end(),
                       it);
    }

    locally_owned_dofs = dof_handler.locally_owned_dofs();

    locally_owned_dofs_per_block.clear();
    for (unsigned int i=0; i<dofs_per_block.size(); ++i)
      locally_owned_dofs_per_block.push_back(locally_owned_dofs.get_view(accumulated_dofs_per_block[i],
                                                                         accumulated_dofs_per_block[i+1]));

    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    locally_relevant_dofs_per_block.clear();
    for (unsigned int i=0; i<dofs_per_block.size(); ++i)
      locally_relevant_dofs_per_block.push_back(locally_relevant_dofs.
                                                get_view(accumulated_dofs_per_block[i],
                                                         accumulated_dofs_per_block[i+1]));
  }
  Vector<double>  vector;
  {
    types::global_dof_index n_dofs{0};
    for (const auto n: dofs_per_block)
      n_dofs += n;

    vector.reinit(n_dofs);
  }

  const double value{1.0 / std::sqrt(double(dofs_per_block[n_blocks-1]))};
  set_block(vector,
            dofs_per_block,
            locally_owned_dofs_per_block,
            n_blocks-1,
            value);
}



}  // namespace TestSpace



int main(void)
{
  try
  {
    dealii::deallog.depth_console(0);

    for (std::size_t i=1; i<5; ++i)
      TestSpace::test_dealii_vector<2>(i);
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}



