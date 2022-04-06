/*
 * cavity_problem.cc
 *
 *  Created on: Sep 1, 2021
 *      Author: sg
 */

#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/grid/grid_generator.h>

#include <hydrodynamic_problem.h>

using namespace Hydrodynamic;

template <int dim>
class CavityProblem : public HydrodynamicProblem<dim>
{
public:
  CavityProblem(ProblemParameters &parameters);

protected:
  virtual void make_grid() override;

  virtual void set_boundary_conditions() override;

private:

  const types::boundary_id  left_bndry_id;
  const types::boundary_id  right_bndry_id;
  const types::boundary_id  bottom_bndry_id;
  const types::boundary_id  top_bndry_id;
  const types::boundary_id  back_bndry_id;
  const types::boundary_id  front_bndry_id;

};



template <int dim>
CavityProblem<dim>::CavityProblem(ProblemParameters &parameters)
:
HydrodynamicProblem<dim>(parameters),
left_bndry_id(0),
right_bndry_id(1),
bottom_bndry_id(2),
top_bndry_id(3),
back_bndry_id(4),
front_bndry_id(5)
{
  std::cout << "Solving cavity problem" << std::endl;
}



template <int dim>
void CavityProblem<dim>::make_grid()
{
  std::cout << "    Make grid..." << std::endl;

  GridGenerator::hyper_cube(this->triangulation, 0.0, 1.0, true);

  this->triangulation.refine_global(this->n_initial_refinements);
}



template <int dim>
void CavityProblem<dim>::set_boundary_conditions()
{
  std::cout << "    Set boundary conditions..." << std::endl;

  VectorBoundaryConditions<dim> &velocity_bcs = this->solver.get_velocity_bcs();
  ScalarBoundaryConditions<dim> &pressure_bcs = this->solver.get_pressure_bcs();

  velocity_bcs.clear();
  pressure_bcs.clear();

  velocity_bcs.extract_boundary_ids();
  pressure_bcs.extract_boundary_ids();

  velocity_bcs.set_dirichlet_bc(left_bndry_id);
  velocity_bcs.set_dirichlet_bc(right_bndry_id);
  velocity_bcs.set_dirichlet_bc(bottom_bndry_id);

  std::vector<double> value(dim);
  value[0] = 1.0;
  const std::shared_ptr<Function<dim>> velocity_top_bndry =
      std::make_shared<Functions::ConstantFunction<dim>>(value);

  velocity_bcs.set_dirichlet_bc(top_bndry_id, velocity_top_bndry);

  pressure_bcs.set_datum_at_boundary();

  velocity_bcs.close();
  pressure_bcs.close();
}



int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

    std::string parameter_filename;
    if (argc >= 2)
      parameter_filename = argv[1];
    else
      parameter_filename = "cavity_problem.prm";

    ProblemParameters parameters(parameter_filename);
    if (parameters.space_dim == 2)
    {
      CavityProblem<2> problem(parameters);
      problem.run();
    }
    else
    {
      CavityProblem<3> problem(parameters);
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
