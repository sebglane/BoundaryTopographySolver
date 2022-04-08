/*
 * container_assembly_test.cc
 *
 *  Created on: Mar 22, 2021
 *      Author: sg
 */
#include <deal.II/base/function_lib.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/table.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>

#include <block_schur_preconditioner.h>

namespace TestSpace
{

using namespace dealii;

template <int dim>
void
test_assembly_serial
(const unsigned int n_refinements,
 const bool add_mean_value_constraint)
{
  // setup triangulation
  std::cout << "Make grid" << std::endl;
  Triangulation<dim>  tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  if (n_refinements > 0)
    tria.refine_global(n_refinements);

  const types::boundary_id  left_bndry_id{0};
  const types::boundary_id  right_bndry_id{1};
  const types::boundary_id  bottom_bndry_id{2};
  const types::boundary_id  top_bndry_id{3};
  const types::boundary_id  back_bndry_id{4};
  const types::boundary_id  front_bndry_id{5};

  // setup finite element
  const unsigned int degree{1};
  FESystem<dim> fe_system(FESystem<dim>(FE_Q<dim>(degree + 1), dim), 1,
                          FE_Q<dim>(degree), 1);
  const unsigned int velocity_fe_degree{degree + 1};

  // setup dofs
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_system);
  DoFRenumbering::block_wise(dof_handler);

  const std::vector<types::global_dof_index>
  dofs_per_block{DoFTools::count_dofs_per_fe_block(dof_handler)};

  std::cout << "    Number of active cells: "
            << tria.n_global_active_cells()
            << std::endl
            << "    Number of total degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  // setup boundary conditions
  AffineConstraints<double>   constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  {
    // velocity boundary conditions
    FEValuesExtractors::Vector  velocity(0);

    Functions::ZeroFunction<dim>  zero_function(dim + 1);
    std::map<types::boundary_id, const Function<dim>* > function_map;
    function_map[left_bndry_id] = &zero_function;
    function_map[right_bndry_id] = &zero_function;
    function_map[bottom_bndry_id] = &zero_function;
    if (dim == 3)
    {
      function_map[front_bndry_id] = &zero_function;
      function_map[back_bndry_id] = &zero_function;
    }

    // inhomogeneous boundary conditions
    std::vector<double> nonzero_value(dim + 1, 0.0);
    nonzero_value[0] = 1.0;
    Functions::ConstantFunction<dim>  nonzero_velocity_function(nonzero_value);
    function_map[top_bndry_id] = &nonzero_velocity_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             function_map,
                                             constraints,
                                             fe_system.component_mask(velocity));

    // pressure boundary conditions
    if (add_mean_value_constraint)
    {

      FEValuesExtractors::Scalar  pressure(dim);
      IndexSet    boundary_dofs;
      DoFTools::extract_boundary_dofs(dof_handler,
                                      fe_system.component_mask(pressure),
                                      boundary_dofs);

      // look for an admissible local degree of freedom to constrain
      types::global_dof_index bndry_idx = numbers::invalid_dof_index;
      IndexSet::ElementIterator idx = boundary_dofs.begin();
      IndexSet::ElementIterator endidx = boundary_dofs.end();
      for(; idx != endidx; ++idx)
        if (constraints.can_store_line(*idx) &&
            !constraints.is_constrained(*idx))
        {
          bndry_idx = *idx;
          break;
        }

      // check that an admissable degree of freedom was found
      AssertThrow(bndry_idx < dof_handler.n_dofs(),
                  ExcMessage("Error, couldn't find a DoF to constrain."));

      // sets the degree of freedom to zero
      constraints.add_line(bndry_idx);
    }

  }
  constraints.close();

  // velocity-pressure coupling
  Table<2, DoFTools::Coupling>  coupling_table;
  coupling_table.reinit(fe_system.n_components(),
                        fe_system.n_components());
  for (unsigned int c=0; c<fe_system.n_components(); ++c)
    for (unsigned int d=0; d<fe_system.n_components(); ++d)
      if (c==d)
        coupling_table[c][d] = DoFTools::always;
      else if ((c==dim && d<dim) || (c<dim && d==dim))
        coupling_table[c][d] = DoFTools::always;
      else
        coupling_table[c][d] = DoFTools::none;

  // setup system matrix and system rhs
  BlockSparsityPattern  sparsity_pattern;
  LA::BlockSparseMatrix system_matrix;
  LA::SparseMatrix      pressure_mass_matrix;
  {
    system_matrix.clear();
    pressure_mass_matrix.clear();

    BlockDynamicSparsityPattern dsp(dofs_per_block,
                                    dofs_per_block);

    DoFTools::make_sparsity_pattern(dof_handler,
                                    coupling_table,
                                    dsp,
                                    constraints);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
  }
  LA::BlockVector system_rhs;
  system_rhs.reinit(dofs_per_block);

  // setup solution
  LA::BlockVector solution;
  solution.reinit(dofs_per_block);

  // assemble system matrix
  {
    system_matrix = 0;
    system_rhs = 0;

    const UpdateFlags update_flags = update_values|
                                     update_gradients|
                                     update_JxW_values;
    const MappingQGeneric<dim>  mapping(1);
    const QGauss<dim>   quadrature_formula(velocity_fe_degree + 1);

    FEValues<dim> fe_values(mapping,
                            fe_system,
                            quadrature_formula,
                            update_flags);

    // shape functions
    std::vector<Tensor<2, dim>> grad_phi_velocity(fe_system.n_dofs_per_cell());
    std::vector<double>         div_phi_velocity(fe_system.n_dofs_per_cell());
    std::vector<double>         phi_pressure(fe_system.n_dofs_per_cell());

    FullMatrix<double>  local_matrix(fe_system.n_dofs_per_cell(),
                                     fe_system.n_dofs_per_cell());
    Vector<double>      local_rhs(fe_system.n_dofs_per_cell());

    std::vector<types::global_dof_index> local_dof_indices(fe_system.n_dofs_per_cell());

    const FEValuesExtractors::Vector  velocity(0);
    const FEValuesExtractors::Scalar  pressure(dim);

    for (const auto &cell: dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      local_matrix = 0;
      local_rhs = 0;

      cell->get_dof_indices(local_dof_indices);

      for (const auto q: fe_values.quadrature_point_indices())
      {
        for (const auto i: fe_values.dof_indices())
        {
          grad_phi_velocity[i] = fe_values[velocity].gradient(i, q);
          div_phi_velocity[i] = fe_values[velocity].divergence(i, q);
          phi_pressure[i] = fe_values[pressure].value(i, q);
        }

        const double JxW{fe_values.JxW(q)};

        for (const auto i: fe_values.dof_indices())
        {
          const Tensor<2, dim> &velocity_test_function_gradient = grad_phi_velocity[i];
          const double          pressure_test_function = phi_pressure[i];

          for (const auto j: fe_values.dof_indices())
          {
            double matrix{-(div_phi_velocity[j] * pressure_test_function +
                            phi_pressure[j] * div_phi_velocity[i])};
            matrix += scalar_product(grad_phi_velocity[j],
                                     velocity_test_function_gradient);
            matrix += phi_pressure[j] * phi_pressure[i];

            local_matrix(i, j) +=  matrix * JxW;
          }
        }
      } // end loop over cell quadrature points

      constraints.distribute_local_to_global(local_matrix,
                                             local_rhs,
                                             local_dof_indices,
                                             system_matrix,
                                             system_rhs);
    }
  }
  pressure_mass_matrix.reinit(sparsity_pattern.block(1, 1));
  pressure_mass_matrix.copy_from(system_matrix.block(1, 1));
  system_matrix.block(1, 1) = 0;

  SparseILU<double> preconditioner_A;
  preconditioner_A.initialize(system_matrix.block(0, 0));

  SparseILU<double> preconditioner_S;
  preconditioner_S.initialize(pressure_mass_matrix);

  const Preconditioning::BlockSchurPreconditioner<SparseILU<double>, SparseILU<double>>
  preconditioner(system_matrix,
                 pressure_mass_matrix,
                 preconditioner_A,
                 preconditioner_S,
                 true,
                 true,
                 true);

  // solve linear system
  SolverControl solver_control(1000, 1e-10 * system_rhs.l2_norm());
  SolverFGMRES<LA::BlockVector> solver(solver_control);
  try
  {
    solver.solve(system_matrix,
                 solution,
                 system_rhs,
                 preconditioner);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception in solving the linear system:" << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception in solving the linear system!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }
  constraints.distribute(solution);

  // correct mean value
  {
    FEValuesExtractors::Scalar  pressure(dim);
    const double mean_value{VectorTools::compute_mean_value(dof_handler,
                                                            QGauss<dim>(velocity_fe_degree + 1),
                                                            solution,
                                                            dim)};
    const IndexSet  pressure_dofs = DoFTools::extract_dofs(dof_handler,
                                                           fe_system.component_mask(pressure));

    IndexSet::ElementIterator idx = pressure_dofs.begin();
    IndexSet::ElementIterator endidx = pressure_dofs.end();
    for(; idx != endidx; ++idx)
      solution[*idx] -= mean_value;
  }

  std::cout << "Solved the linear system for " << std::endl
            << "\t- pressure mean value constraint: "
            << std::boolalpha << add_mean_value_constraint << std::noboolalpha << std::endl;
}

}  // namespace TestSpace



int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
    deallog.depth_console(0);

    for (unsigned int i=1; i<5; ++i)
    {
      TestSpace::test_assembly_serial<2>(i, true);
      TestSpace::test_assembly_serial<2>(i, false);
    }
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



