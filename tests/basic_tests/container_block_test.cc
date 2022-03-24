/*
 * linear_algebra_container_01.cc
 *
 *  Created on: Oct 19, 2021
 *      Author: sg
 */
#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/table.h>
#include <deal.II/base/logstream.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <linear_algebra_container.h>

namespace TestSpace
{

using namespace dealii;

template <int dim, typename Container>
void
test_container
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

  const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler);

  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  {
    Functions::ZeroFunction<dim>  zero_function(n_blocks);
    std::map<types::boundary_id, const Function<dim>* > function_map;
    function_map[0] = &zero_function;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             function_map,
                                             constraints);
  }

  Table<2, DoFTools::Coupling>  coupling_table;
  coupling_table.reinit(n_blocks, n_blocks);
  coupling_table.fill(DoFTools::Coupling::always);

  Container container(tria.get_communicator());
  container.setup(dof_handler,
                  constraints,
                  coupling_table,
                  n_blocks);

  const double value{1.0 / std::sqrt(double(dofs_per_block[n_blocks - 1]))};
  container.set_block(n_blocks - 1, value, container.system_rhs);

  for (const auto residual: container.get_residual_components())
    std::cout << residual << ", ";
  std::cout << std::endl;

  typename Container::vector_type  evaluation_point;
  container.setup_vector(evaluation_point);
  for (std::size_t i=0; i<n_blocks; ++i)
    container.set_block(i, std::pow(double(i), 2), evaluation_point);

  typename Container::vector_type  present_solution;
  container.setup_vector(present_solution);
  typename Container::vector_type  solution_update;
  container.setup_vector(solution_update);

  container.set_vector(evaluation_point, present_solution);
  container.set_vector(evaluation_point, solution_update);
  container.add(present_solution, solution_update);
  container.add(0.1, evaluation_point, present_solution);

}

}  // namespace TestSpace



int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
    deallog.depth_console(0);

    using Container = typename SolverBase::
        LinearAlgebraContainer<Vector<double>, SparseMatrix<double>, SparsityPattern>;

    using BlockContainer = typename SolverBase::
        LinearAlgebraContainer<BlockVector<double>, BlockSparseMatrix<double>, BlockSparsityPattern>;

    for (std::size_t i=1; i<9; ++i)
      TestSpace::test_container<2, Container>(i);

    for (std::size_t i=1; i<9; ++i)
      TestSpace::test_container<3, Container>(i);

    for (std::size_t i=1; i<9; ++i)
      TestSpace::test_container<2, BlockContainer>(i);

    for (std::size_t i=1; i<9; ++i)
      TestSpace::test_container<3, BlockContainer>(i);

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



