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
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>

#include <linear_algebra_container.h>

namespace TestSpace
{

using namespace dealii;

template <int dim, typename Container>
void
test_assembly_serial
(const unsigned int n_refinements)
{
  // setup parameters
  const double reynolds_number{1.0};

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

  const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler);

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
  inhomogeneous_constraints.close();
  homogeneous_constraints.close();

  std::cout << "inhomogeneous_constraints" << std::endl;
  inhomogeneous_constraints.print(std::cout);
  std::cout << "homogeneous_constraints" << std::endl;
  homogeneous_constraints.print(std::cout);

  // velocity-pressure coupling
  Table<2, DoFTools::Coupling>  coupling_table;
  coupling_table.reinit(fe_system.n_components(),
                        fe_system.n_components());
  for (unsigned int c=0; c<dim+1; ++c)
    for (unsigned int d=0; d<dim+1; ++d)
      if (c<dim || d<dim)
        coupling_table[c][d] = DoFTools::always;
      else if ((c==dim && d<dim) || (c<dim && d==dim))
        coupling_table[c][d] = DoFTools::always;
      else
        coupling_table[c][d] = DoFTools::none;

  // setup container
  Container container(tria.get_communicator());
  container.setup(dof_handler,
                  homogeneous_constraints,
                  coupling_table,
                  fe_system.n_blocks());

  // set solution
  using VectorType = typename Container::vector_type;
  VectorType  evaluation_point;
  VectorType  present_solution;
  container.setup_vector(evaluation_point);
  container.setup_vector(present_solution);

  container.distribute_constraints(inhomogeneous_constraints,
                                   evaluation_point);

  std::cout << "evaluation_point (set solution)" << std::endl;
  evaluation_point.print(std::cout, 3, true, false);

  // compute initial residual
  {
    container.system_rhs = 0.0;

    const UpdateFlags update_flags = update_values|
                                     update_gradients|
                                     update_JxW_values;
    const MappingQGeneric<dim>  mapping(1);
    const QGauss<dim>     quadrature_formula(velocity_fe_degree + 1);
    const QGauss<dim-1>   face_quadrature_formula(velocity_fe_degree + 1);

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

    Vector<double>      local_rhs(fe_system.n_dofs_per_cell());

    std::vector<types::global_dof_index> local_dof_indices(fe_system.n_dofs_per_cell());

    const FEValuesExtractors::Vector  velocity(0);
    const FEValuesExtractors::Scalar  pressure(dim);

    for (const auto &cell: dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      local_rhs = 0;

      cell->get_dof_indices(local_dof_indices);

      const FEValuesExtractors::Vector  velocity(0);
      const FEValuesExtractors::Scalar  pressure(dim);

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
          double rhs{present_velocity_divergence * phi_pressure[i] +
                     present_pressure_values[q] * div_phi_velocity[i] -
                     (present_velocity_gradients[q] * present_velocity_values[q]) *
                     phi_velocity[i]};

          rhs -= nu * scalar_product(present_velocity_gradients[q],
                                     grad_phi_velocity[i]);

          local_rhs(i) += rhs * JxW;
        }
      } // end loop over cell quadrature points

      inhomogeneous_constraints.distribute_local_to_global(local_rhs,
                                                           local_dof_indices,
                                                           container.system_rhs);
    }
  }

  std::cout << "system_rhs (compute initial residual)" << std::endl;
  container.system_rhs.print(std::cout, 3, true, false);

  // assemble system matrix
  {
    container.set_vector(present_solution, evaluation_point);

    container.system_matrix = 0;
    container.system_rhs = 0;

    const UpdateFlags update_flags = update_values|
                                     update_gradients|
                                     update_JxW_values;
    const MappingQGeneric<dim>  mapping(1);
    const QGauss<dim>   quadrature_formula(velocity_fe_degree + 1);
    const QGauss<dim-1> face_quadrature_formula(velocity_fe_degree + 1);

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

            local_matrix(i, j) +=  matrix * JxW;
          }

          double rhs{present_velocity_divergence * pressure_test_function +
                     present_pressure_values[q] * div_phi_velocity[i] -
                     (present_velocity_gradients[q] * present_velocity_values[q]) *
                     phi_velocity[i]};

          rhs -= nu * scalar_product(present_velocity_gradients[q],
                                     grad_phi_velocity[i]);

          local_rhs(i) += rhs * JxW;
        }
      } // end loop over cell quadrature points

      inhomogeneous_constraints.distribute_local_to_global(local_matrix,
                                                           local_rhs,
                                                           local_dof_indices,
                                                           container.system_matrix,
                                                           container.system_rhs);
    }
  }

  std::cout << "system_matrix (assemble linear system)" << std::endl;
  container.system_matrix.print(std::cout);

  std::cout << "system_rhs (assemble linear system)" << std::endl;
  container.system_rhs.print(std::cout, 3, true, false);

  // solve linear system
  SparseDirectUMFPACK direct_solver;
  direct_solver.solve(container.system_matrix, container.system_rhs);
  container.set_vector(container.system_rhs, present_solution);
  container.distribute_constraints(inhomogeneous_constraints,
                                   present_solution);

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

  std::cout << "present_solution (solve linear system)" << std::endl;
  present_solution.print(std::cout, 3, true, false);

  // compute current residual
  {
    container.set_vector(present_solution, evaluation_point);

    container.system_rhs = 0.0;

    const UpdateFlags update_flags = update_values|
                                     update_gradients|
                                     update_JxW_values;
    const MappingQGeneric<dim>  mapping(1);
    const QGauss<dim>     quadrature_formula(velocity_fe_degree + 1);
    const QGauss<dim-1>   face_quadrature_formula(velocity_fe_degree + 1);

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

    Vector<double>      local_rhs(fe_system.n_dofs_per_cell());

    std::vector<types::global_dof_index> local_dof_indices(fe_system.n_dofs_per_cell());

    const FEValuesExtractors::Vector  velocity(0);
    const FEValuesExtractors::Scalar  pressure(dim);

    for (const auto &cell: dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
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
          double rhs{present_velocity_divergence * phi_pressure[i] +
                     present_pressure_values[q] * div_phi_velocity[i] -
                     (present_velocity_gradients[q] * present_velocity_values[q]) *
                     phi_velocity[i]};

          rhs -= nu * scalar_product(present_velocity_gradients[q],
                                     grad_phi_velocity[i]);

          local_rhs(i) += rhs * JxW;
        }
      } // end loop over cell quadrature points

      homogeneous_constraints.distribute_local_to_global(local_rhs,
                                                         local_dof_indices,
                                                         container.system_rhs);
    }
  }

  std::cout << "system_rhs (compute current residual)" << std::endl;
  container.system_rhs.print(std::cout, 3, true, false);
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

    TestSpace::test_assembly_serial<2, Container>(1);
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



