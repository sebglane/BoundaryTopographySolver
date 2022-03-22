/*
 * linear_algebra_container_03.cc
 *
 *  Created on: Mar 22, 2021
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

#include <deal.II/numerics/vector_tools.h>

#include <assembly_functions.h>
#include <hydrodynamic_solver.h>
#include <linear_algebra_container.h>

namespace TestSpace
{

using namespace dealii;

template <int dim, typename Container>
void
test_assembly_serial
(const unsigned int n_refinements,
 const unsigned int degree = 1)
{
  // setup parameters
  const double reynolds_number{1.0};

  // setup triangulation
  std::cout << "Make grid" << std::endl;
  Triangulation<dim>            tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(n_refinements);

  const types::boundary_id  left_bndry_id{0};
  const types::boundary_id  right_bndry_id{1};
  const types::boundary_id  bottom_bndry_id{2};
  const types::boundary_id  top_bndry_id{3};
  const types::boundary_id  back_bndry_id{4};
  const types::boundary_id  front_bndry_id{5};

  // setup finite element
  FESystem<dim> fe_system(FESystem<dim>(FE_Q<dim>(degree + 1), dim), 1,
                          FE_Q<dim>(degree), 1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_system);
  DoFRenumbering::block_wise(dof_handler);

  const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler);
  const unsigned int n_blocks{2};
  std::cout << "    Number of active cells: "
            << tria.n_global_active_cells()
            << std::endl
            << "    Number of total degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  // setup boundary conditions
  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  {
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

    std::vector<double> value(dim + 1);
    value[0] = 1.0;
    Functions::ConstantFunction<dim>  velocity_function(value);
    function_map[top_bndry_id] = &velocity_function;

    FEValuesExtractors::Vector  velocity(0);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             function_map,
                                             constraints,
                                             fe_system.component_mask(velocity));
  }
  constraints.close();

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
                  constraints,
                  coupling_table,
                  n_blocks);

  // set solution
  container.set_evaluation_point(container.present_solution);
  container.distribute_constraints(container.evaluation_point, constraints);


  // assembly right-hand side
  UpdateFlags update_flags = update_values|
                             update_gradients|
                             update_JxW_values;
  UpdateFlags face_update_flags = update_values|
                                  update_quadrature_points|
                                  update_JxW_values;
  MappingQ1<dim>      mapping;
  const QGauss<dim>   quadrature_formula(degree + 1);
  const QGauss<dim-1> face_quadrature_formula(degree + 1);

  Hydrodynamic::AssemblyData::RightHandSide::Scratch<dim>
  scratch(mapping,
          quadrature_formula,
          fe_system,
          update_flags,
          face_quadrature_formula,
          face_update_flags,
          StabilizationFlags::apply_none);

  AssemblyBaseData::RightHandSide::Copy
  data(fe_system.n_dofs_per_cell());

  for (const auto &cell: dof_handler.active_cell_iterators())
  {
    scratch.fe_values.reinit(cell);

    data.local_rhs = 0;

    cell->get_dof_indices(data.local_dof_indices);

    const FEValuesExtractors::Vector  velocity(0);
    const FEValuesExtractors::Scalar  pressure(dim);

    const double nu{1.0 / reynolds_number};

    Hydrodynamic::OptionalArgumentsWeakForm<dim> &weak_form_options = scratch.optional_arguments_weak_from;
    Hydrodynamic::OptionalArgumentsStrongForm<dim> &strong_form_options = scratch.optional_arguments_strong_from;
    weak_form_options.use_stress_form = false;
    strong_form_options.use_stress_form = false;

    // solution values
    scratch.fe_values[velocity].get_function_values(container.evaluation_point,
                                                    scratch.present_velocity_values);
    scratch.fe_values[velocity].get_function_gradients(container.evaluation_point,
                                                       scratch.present_velocity_gradients);

    scratch.fe_values[pressure].get_function_values(container.evaluation_point,
                                                    scratch.present_pressure_values);

    for (const auto q: scratch.fe_values.quadrature_point_indices())
    {
      for (const auto i: scratch.fe_values.dof_indices())
      {
        scratch.phi_velocity[i] = scratch.fe_values[velocity].value(i, q);
        scratch.grad_phi_velocity[i] = scratch.fe_values[velocity].gradient(i, q);
        scratch.div_phi_velocity[i] = scratch.fe_values[velocity].divergence(i, q);
        scratch.phi_pressure[i] = scratch.fe_values[pressure].value(i, q);
      }

      const double JxW{scratch.fe_values.JxW(q)};

      for (const auto i: scratch.fe_values.dof_indices())
      {
        double rhs = Hydrodynamic::compute_rhs(scratch.phi_velocity[i],
                                               scratch.grad_phi_velocity[i],
                                               scratch.present_velocity_values[q],
                                               scratch.present_velocity_gradients[q],
                                               scratch.present_pressure_values[q],
                                               scratch.phi_pressure[i],
                                               nu,
                                               weak_form_options);
        data.local_rhs(i) += rhs * JxW;
      }
    } // end loop over cell quadrature points

    constraints.distribute_local_to_global(data.local_rhs,
                                           data.local_dof_indices,
                                           container.system_rhs);
  }

  std::cout << std::scientific
            << container.system_rhs.l2_norm()
            << std::defaultfloat
            << std::endl;

  std::cout << std::scientific;
  for (const auto residual: container.get_residual_components())
    std::cout << residual << ", ";
  std::cout << std::defaultfloat
            << std::endl;
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

    TestSpace::test_assembly_serial<2, BlockContainer>(4);
    TestSpace::test_assembly_serial<2, Container>(4);
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



