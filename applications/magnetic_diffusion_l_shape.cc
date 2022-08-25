/*
 * magnetic_induction_problem.cc
 *
 *  Created on: Aug 23, 2022
 *      Author: sg
 */

#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <magnetic_induction_problem.h>

#include <filesystem>
#include <memory>
#include <set>

namespace MagneticDiffusionProblem {

using namespace MagneticInduction;


template <int dim>
class ExactMagneticField: public Function<dim>
{
public:
  ExactMagneticField(const unsigned int n);

  virtual double value(const Point<dim>    &point,
                       const unsigned int   component) const;

  virtual void vector_value(const Point<dim>   &point,
                            Vector<double>     &value) const;

private:
  const unsigned int n;

};



template <int dim>
ExactMagneticField<dim>::ExactMagneticField(const unsigned int n)
:
Function<dim>(dim),
n(n)
{
  AssertThrow(n >= 1, ExcLowerRange(n, 1));
}



template <int dim>
double ExactMagneticField<dim>::value
(const Point<dim>    &point,
 const unsigned int   component) const
{
  std::array<double, dim> spherical_coords{GeometricUtilities::Coordinates::to_spherical(point)};

  const double factor{2. / 3. * n * std::pow(spherical_coords[0], 2.0 / 3.0 * n - 2.0)};

  switch (component)
  {
    case 0:
      return (factor * (-point[1] * std::cos(2.0 / 3.0 * n * spherical_coords[1]) +
                         point[0] * std::sin(2.0 / 3.0 * n * spherical_coords[1])));
    case 1:
      return (factor * (point[0] * std::cos(2.0 / 3.0 * n * spherical_coords[1]) +
                        point[1] * std::sin(2.0 / 3.0 * n * spherical_coords[1])));
    case 2:
      return (1.0);
    default:
      break;
  }

  AssertThrow(dim <= 3, ExcImpossibleInDim(dim));

  return (0.0);
}



template <int dim>
void ExactMagneticField<dim>::vector_value
(const Point<dim>   &point,
 Vector<double>     &value) const
{
  AssertDimension(value.size(), dim);

  std::array<double, dim> spherical_coords{GeometricUtilities::Coordinates::to_spherical(point)};

  const double factor{2. / 3. * n * std::pow(spherical_coords[0], n / 3.0 - 1.0)};

  value[0] = factor * (-point[1] * std::cos(2.0 / 3.0 * n * spherical_coords[1]) +
                        point[0] * std::sin(2.0 / 3.0 * n * spherical_coords[1]));

  value[1] = factor * (point[0] * std::cos(2.0 / 3.0 * n * spherical_coords[1]) +
                       point[1] * std::sin(2.0 / 3.0 * n * spherical_coords[1]));

  if (dim == 3)
    value[2] = 1.0;
}



template <int dim>
class Problem : public MagneticInductionProblem<dim>
{
public:
  Problem(ProblemParameters &parameters);

protected:
  virtual void make_grid() override;

  virtual void set_boundary_conditions() override;

private:
  const types::boundary_id  dirichlet_bndry_id;

  const unsigned int n;
};



template <>
Problem<2>::Problem(ProblemParameters &parameters)
:
MagneticInductionProblem<2>(parameters),
dirichlet_bndry_id(1),
n(4)
{
  std::cout << "Solving magnetic diffusion problem on L-shaped domain" << std::endl;
}



template <>
Problem<3>::Problem(ProblemParameters &parameters)
:
MagneticInductionProblem<3>(parameters),
dirichlet_bndry_id(1),
n(4)
{
  std::cout << "Solving magnetic diffusion problem on L-shaped domain" << std::endl;
}



template <int dim>
void Problem<dim>::make_grid()
{
  std::cout << "    Make grid..." << std::endl;

  GridGenerator::hyper_L(this->triangulation, -1.0, 1.0);

  // assignment of boundary identifiers
  for (const auto &cell: this->triangulation.active_cell_iterators())
    if (cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
          face->set_boundary_id(dirichlet_bndry_id);

  this->triangulation.refine_global(this->n_initial_refinements);

  // initial boundary refinements
  if (this->n_initial_bndry_refinements > 0)
  {
    for (unsigned int step=0; step<this->n_initial_bndry_refinements; ++step)
    {
      for (const auto &cell: this->triangulation.active_cell_iterators())
        if (cell->at_boundary())
          for (const auto &face : cell->face_iterators())
            if (face->at_boundary())
              cell->set_refine_flag();
      this->triangulation.execute_coarsening_and_refinement();
    }
  }

  {
    const unsigned int magnetic_fe_degree{2};

    FESystem<dim> fe_system(FE_Q<dim>(magnetic_fe_degree), dim);
    DoFHandler<dim> dof_handler;
    dof_handler.reinit(this->triangulation);
    dof_handler.distribute_dofs(fe_system);

    const std::shared_ptr<Function<dim>> magnetic_field_function =
        std::make_shared<ExactMagneticField<dim>>(n);

    Vector<double> interpolated_exact_solution(dof_handler.n_dofs());

    VectorTools::interpolate(dof_handler,
                             *magnetic_field_function,
                             interpolated_exact_solution);

    AffineConstraints<double> constraints;
    std::set<types::boundary_id> boundary_ids;
    boundary_ids.insert(dirichlet_bndry_id);

    std::map<types::boundary_id, const Function<dim> *> function_map;
    function_map[dirichlet_bndry_id] = magnetic_field_function.get();

    VectorTools::compute_nonzero_tangential_flux_constraints
    (dof_handler,
     0,
     boundary_ids,
     function_map,
     constraints);
    constraints.close();

    Vector<double>  projected_exact_solution(dof_handler.n_dofs());
    VectorTools::project(dof_handler,
                         constraints,
                         QGauss<dim>(magnetic_fe_degree + 1),
                         *magnetic_field_function, projected_exact_solution);
    constraints.distribute(projected_exact_solution);

    // prepare data out object
    DataOut<dim>  data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(interpolated_exact_solution,
                             std::vector<std::string>(dim, "interpolated_magnetic_field"),
                             data_out.type_dof_data,
                             std::vector<DataComponentInterpretation::DataComponentInterpretation>(dim, DataComponentInterpretation::component_is_part_of_vector));
    data_out.add_data_vector(projected_exact_solution,
                             std::vector<std::string>(dim, "projected_magnetic_field"),
                             data_out.type_dof_data,
                             std::vector<DataComponentInterpretation::DataComponentInterpretation>(dim, DataComponentInterpretation::component_is_part_of_vector));
    data_out.build_patches(magnetic_fe_degree);

    // write output to disk
    const std::string filename = "exact-solution.vtk";
    std::filesystem::path output_file(".");
    output_file /= filename;

    std::ofstream fstream(output_file.c_str());
    data_out.write_vtk(fstream);
  }
}



template <int dim>
void Problem<dim>::set_boundary_conditions()
{
  std::cout << "    Set boundary conditions..." << std::endl;

  // magnetic field bcs
  {
    VectorBoundaryConditions<dim> &bcs = this->solver.get_magnetic_field_bcs();

    bcs.clear();

    bcs.extract_boundary_ids();

    const std::shared_ptr<Function<dim>> magnetic_field_function =
        std::make_shared<ExactMagneticField<dim>>(n);

    bcs.set_tangential_flux_bc(dirichlet_bndry_id, magnetic_field_function);

    bcs.close();
  }

  // magnetic pressure bcs
  {
    ScalarBoundaryConditions<dim> &bcs = this->solver.get_magnetic_pressure_bcs();

    bcs.clear();

    bcs.extract_boundary_ids();

    bcs.set_dirichlet_bc(dirichlet_bndry_id);

    bcs.close();
  }
}

}  // namespace MagneticDiffusionProblem



int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

    std::string parameter_filename;
    if (argc >= 2)
      parameter_filename = argv[1];
    else
      parameter_filename = "magnetic_diffusion_l_shape.prm";

    MagneticInduction::ProblemParameters parameters(parameter_filename);
    if (parameters.space_dim == 2)
    {
      MagneticDiffusionProblem::Problem<2> problem(parameters);
      problem.run();
    }
    else
    {
      MagneticDiffusionProblem::Problem<3> problem(parameters);
      problem.run();
    }
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
    std::cerr << "Exception on processing: " << std::endl
            << exc.what() << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
    std::cerr << "Unknown exception!" << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
    return 1;
  }
  return 0;
}
