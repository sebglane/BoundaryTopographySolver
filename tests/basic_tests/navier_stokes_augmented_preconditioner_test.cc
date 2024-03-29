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
 const double reynolds_number = 100.0,
 const bool add_mean_value_constraint = false)
{
  // setup parameters
  const double gamma{1.0};

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
  AffineConstraints<double>   inhomogeneous_constraints;
  AffineConstraints<double>   homogeneous_constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, inhomogeneous_constraints);
  DoFTools::make_hanging_node_constraints(dof_handler, homogeneous_constraints);
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
                                             inhomogeneous_constraints,
                                             fe_system.component_mask(velocity));


    // homogeneous boundary conditions
    function_map[top_bndry_id] = &zero_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             function_map,
                                             homogeneous_constraints,
                                             fe_system.component_mask(velocity));

    // pressure boundary conditions
    if (add_mean_value_constraint)
    {

      FEValuesExtractors::Scalar  pressure(dim);
      const IndexSet boundary_dofs =
          DoFTools::extract_boundary_dofs(dof_handler,
                                          fe_system.component_mask(pressure));

      // look for an admissible local degree of freedom to constrain
      types::global_dof_index bndry_idx = numbers::invalid_dof_index;
      IndexSet::ElementIterator idx = boundary_dofs.begin();
      IndexSet::ElementIterator endidx = boundary_dofs.end();
      for(; idx != endidx; ++idx)
        if ((homogeneous_constraints.can_store_line(*idx) &&
             !homogeneous_constraints.is_constrained(*idx)) &&
            (inhomogeneous_constraints.can_store_line(*idx) &&
             !inhomogeneous_constraints.is_constrained(*idx)))
        {
          bndry_idx = *idx;
          break;
        }

      // check that an admissable degree of freedom was found
      AssertThrow(bndry_idx < dof_handler.n_dofs(),
                  ExcMessage("Error, couldn't find a DoF to constrain."));

      // sets the degree of freedom to zero
      homogeneous_constraints.add_line(bndry_idx);
      inhomogeneous_constraints.add_line(bndry_idx);
    }

  }
  inhomogeneous_constraints.close();
  homogeneous_constraints.close();

  // setup system matrix and system rhs
  BlockSparsityPattern  sparsity_pattern;
  LA::BlockSparseMatrix system_matrix;
  LA::SparseMatrix      scaled_pressure_mass_matrix;
  {
    system_matrix.clear();
    scaled_pressure_mass_matrix.clear();

    BlockDynamicSparsityPattern dsp(dofs_per_block,
                                    dofs_per_block);

    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    homogeneous_constraints);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
  }
  LA::BlockVector system_rhs;
  system_rhs.reinit(dofs_per_block);

  // setup solution
  LA::BlockVector evaluation_point;
  LA::BlockVector present_solution;
  LA::BlockVector solution_update;
  evaluation_point.reinit(dofs_per_block);
  present_solution.reinit(dofs_per_block);
  solution_update.reinit(dofs_per_block);

  inhomogeneous_constraints.distribute(evaluation_point);

  // assemble system matrix
  {
    evaluation_point = present_solution;

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
    std::vector<Tensor<1, dim>> phi_velocity(fe_system.n_dofs_per_cell());
    std::vector<Tensor<2, dim>> grad_phi_velocity(fe_system.n_dofs_per_cell());
    std::vector<double>         div_phi_velocity(fe_system.n_dofs_per_cell());
    std::vector<double>         phi_pressure(fe_system.n_dofs_per_cell());

    // solution values
    std::vector<Tensor<1, dim>> present_velocity_values(quadrature_formula.size());
    std::vector<Tensor<2, dim>> present_velocity_gradients(quadrature_formula.size());
    std::vector<double>         present_pressure_values(quadrature_formula.size());

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

      const double nu{1.0 / reynolds_number};

      // solution values
      fe_values[velocity].get_function_values(evaluation_point,
                                              present_velocity_values);
      fe_values[velocity].get_function_gradients(evaluation_point,
                                                 present_velocity_gradients);

      fe_values[pressure].get_function_values(evaluation_point,
                                              present_pressure_values);

      for (const auto q: fe_values.quadrature_point_indices())
      {
        for (const auto i: fe_values.dof_indices())
        {
          phi_velocity[i] = fe_values[velocity].value(i, q);
          grad_phi_velocity[i] = fe_values[velocity].gradient(i, q);
          div_phi_velocity[i] = fe_values[velocity].divergence(i, q);
          phi_pressure[i] = fe_values[pressure].value(i, q);
        }

        const double present_velocity_divergence{trace(present_velocity_gradients[q])};
        const double JxW{fe_values.JxW(q)};

        for (const auto i: fe_values.dof_indices())
        {
          const Tensor<1, dim> &velocity_test_function = phi_velocity[i];
          const Tensor<2, dim> &velocity_test_function_gradient = grad_phi_velocity[i];
          const double          pressure_test_function = phi_pressure[i];

          for (const auto j: fe_values.dof_indices())
          {
            double matrix{-(div_phi_velocity[j] * pressure_test_function +
                            phi_pressure[j] * div_phi_velocity[i])};

            matrix += (present_velocity_gradients[q] * phi_velocity[j] +
                       grad_phi_velocity[j] * present_velocity_values[q]) *
                      velocity_test_function;
            matrix += nu * scalar_product(grad_phi_velocity[j],
                                          velocity_test_function_gradient);
            matrix += gamma * div_phi_velocity[j] * div_phi_velocity[i];
            matrix += (nu + gamma) * phi_pressure[j] * phi_pressure[i];

            local_matrix(i, j) +=  matrix * JxW;
          }

          double rhs{present_velocity_divergence * pressure_test_function +
                     present_pressure_values[q] * div_phi_velocity[i] -
                     (present_velocity_gradients[q] * present_velocity_values[q]) *
                     phi_velocity[i]};

          rhs -= nu * scalar_product(present_velocity_gradients[q],
                                     grad_phi_velocity[i]);
          rhs -= gamma * present_velocity_divergence * div_phi_velocity[i];

          local_rhs(i) += rhs * JxW;
        }
      } // end loop over cell quadrature points

      inhomogeneous_constraints.distribute_local_to_global(local_matrix,
                                                           local_rhs,
                                                           local_dof_indices,
                                                           system_matrix,
                                                           system_rhs);
    }
  }
  scaled_pressure_mass_matrix.reinit(sparsity_pattern.block(1, 1));
  scaled_pressure_mass_matrix.copy_from(system_matrix.block(1, 1));
  system_matrix.block(1, 1) = 0;

  {
    SparseDirectUMFPACK preconditioner_A;
    preconditioner_A.initialize(system_matrix.block(0, 0));

    SparseILU<double> preconditioner_S;
    preconditioner_S.initialize(scaled_pressure_mass_matrix);

    const Preconditioning::BlockSchurPreconditioner<SparseDirectUMFPACK, SparseILU<double>>
    preconditioner(system_matrix,
                   scaled_pressure_mass_matrix,
                   preconditioner_A,
                   preconditioner_S,
                   true,
                   true,
                   true);

    SolverControl solver_control(1000, 1e-10 * system_rhs.l2_norm());
    SolverFGMRES<LA::BlockVector> solver(solver_control);
    try
    {
      solver.solve(system_matrix,
                   present_solution,
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
    inhomogeneous_constraints.distribute(present_solution);
  }

  // correct mean value
  {
    FEValuesExtractors::Scalar  pressure(dim);
    const double mean_value{VectorTools::compute_mean_value(dof_handler,
                                                            QGauss<dim>(velocity_fe_degree + 1),
                                                            present_solution,
                                                            dim)};
    const IndexSet  pressure_dofs = DoFTools::extract_dofs(dof_handler,
                                                           fe_system.component_mask(pressure));

    IndexSet::ElementIterator idx = pressure_dofs.begin();
    IndexSet::ElementIterator endidx = pressure_dofs.end();
    for(; idx != endidx; ++idx)
      present_solution[*idx] -= mean_value;
  }

  // assemble system matrix
  {
    evaluation_point = present_solution;

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
    std::vector<Tensor<1, dim>> phi_velocity(fe_system.n_dofs_per_cell());
    std::vector<Tensor<2, dim>> grad_phi_velocity(fe_system.n_dofs_per_cell());
    std::vector<double>         div_phi_velocity(fe_system.n_dofs_per_cell());
    std::vector<double>         phi_pressure(fe_system.n_dofs_per_cell());

    // solution values
    std::vector<Tensor<1, dim>> present_velocity_values(quadrature_formula.size());
    std::vector<Tensor<2, dim>> present_velocity_gradients(quadrature_formula.size());
    std::vector<double>         present_pressure_values(quadrature_formula.size());

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

      const double nu{1.0 / reynolds_number};

      // solution values
      fe_values[velocity].get_function_values(evaluation_point,
                                              present_velocity_values);
      fe_values[velocity].get_function_gradients(evaluation_point,
                                                 present_velocity_gradients);

      fe_values[pressure].get_function_values(evaluation_point,
                                              present_pressure_values);

      for (const auto q: fe_values.quadrature_point_indices())
      {
        for (const auto i: fe_values.dof_indices())
        {
          phi_velocity[i] = fe_values[velocity].value(i, q);
          grad_phi_velocity[i] = fe_values[velocity].gradient(i, q);
          div_phi_velocity[i] = fe_values[velocity].divergence(i, q);
          phi_pressure[i] = fe_values[pressure].value(i, q);
        }

        const double present_velocity_divergence{trace(present_velocity_gradients[q])};
        const double JxW{fe_values.JxW(q)};

        for (const auto i: fe_values.dof_indices())
        {
          const Tensor<1, dim> &velocity_test_function = phi_velocity[i];
          const Tensor<2, dim> &velocity_test_function_gradient = grad_phi_velocity[i];
          const double          pressure_test_function = phi_pressure[i];

          for (const auto j: fe_values.dof_indices())
          {
            double matrix{-(div_phi_velocity[j] * pressure_test_function +
                            phi_pressure[j] * div_phi_velocity[i])};

            matrix += (present_velocity_gradients[q] * phi_velocity[j] +
                       grad_phi_velocity[j] * present_velocity_values[q]) *
                      velocity_test_function;
            matrix += nu * scalar_product(grad_phi_velocity[j],
                                          velocity_test_function_gradient);
            matrix += gamma * div_phi_velocity[j] * div_phi_velocity[i];
            local_matrix(i, j) +=  matrix * JxW;
          }

          double rhs{present_velocity_divergence * pressure_test_function +
                     present_pressure_values[q] * div_phi_velocity[i] -
                     (present_velocity_gradients[q] * present_velocity_values[q]) *
                     phi_velocity[i]};

          rhs -= nu * scalar_product(present_velocity_gradients[q],
                                     grad_phi_velocity[i]);
          rhs -= gamma * present_velocity_divergence * div_phi_velocity[i];

          local_rhs(i) += rhs * JxW;
        }
      } // end loop over cell quadrature points

      inhomogeneous_constraints.distribute_local_to_global(local_matrix,
                                                           local_rhs,
                                                           local_dof_indices,
                                                           system_matrix,
                                                           system_rhs);
    }
  }
  {
    SparseDirectUMFPACK preconditioner_A;
    preconditioner_A.initialize(system_matrix.block(0, 0));

    SparseILU<double> preconditioner_S;
    preconditioner_S.initialize(scaled_pressure_mass_matrix);

    const Preconditioning::BlockSchurPreconditioner<SparseDirectUMFPACK, SparseILU<double>>
    preconditioner(system_matrix,
                   scaled_pressure_mass_matrix,
                   preconditioner_A,
                   preconditioner_S,
                   true,
                   true,
                   true);

    SolverControl solver_control(1000, 1e-10 * system_rhs.l2_norm());
    SolverFGMRES<LA::BlockVector> solver(solver_control);
    try
    {
      solver.solve(system_matrix,
                   present_solution,
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
    inhomogeneous_constraints.distribute(present_solution);
  }

  // correct mean value
  {
    FEValuesExtractors::Scalar  pressure(dim);
    const double mean_value{VectorTools::compute_mean_value(dof_handler,
                                                            QGauss<dim>(velocity_fe_degree + 1),
                                                            present_solution,
                                                            dim)};
    const IndexSet  pressure_dofs = DoFTools::extract_dofs(dof_handler,
                                                           fe_system.component_mask(pressure));

    IndexSet::ElementIterator idx = pressure_dofs.begin();
    IndexSet::ElementIterator endidx = pressure_dofs.end();
    for(; idx != endidx; ++idx)
      present_solution[*idx] -= mean_value;
  }

  std::cout << "Solved the linear system for " << std::endl
            << "\t- pressure mean value constraint: "
            << std::boolalpha << add_mean_value_constraint << std::noboolalpha << std::endl
            << "\t- Re: " << reynolds_number << std::endl;
}

}  // namespace TestSpace



int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
    deallog.depth_console(0);

    for (unsigned int i=4; i<6; ++i)
    {
      TestSpace::test_assembly_serial<2>(i, 100.0);
      TestSpace::test_assembly_serial<2>(i, 100.0, true);
      TestSpace::test_assembly_serial<2>(i, 1000.0);
      TestSpace::test_assembly_serial<2>(i, 1000.0, true);
      TestSpace::test_assembly_serial<2>(i, 10000.0);
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



