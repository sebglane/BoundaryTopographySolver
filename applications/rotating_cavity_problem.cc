/*
 * cavity_problem.cc
 *
 *  Created on: Sep 1, 2021
 *      Author: sg
 */

#include <deal.II/base/function_lib.h>
#include <deal.II/grid/grid_generator.h>

#include <angular_velocity.h>
#include <hydrodynamic_problem.h>

#include <memory>

using namespace Hydrodynamic;


template <int dim>
class ConstantAngularVelocity : public Utility::AngularVelocity<dim>
{
public:
  ConstantAngularVelocity(const double time = 0);

  virtual typename Utility::AngularVelocity<dim>::value_type value() const;
};



template <int dim>
ConstantAngularVelocity<dim>::ConstantAngularVelocity(const double time)
:
Utility::AngularVelocity<dim>(time)
{}



template <>
typename Utility::AngularVelocity<2>::value_type
ConstantAngularVelocity<2>::value() const
{
  value_type value;
  value[0] = 1.0;

  return (value);
}



template <>
typename Utility::AngularVelocity<3>::value_type
ConstantAngularVelocity<3>::value() const
{
  value_type value;
  value[2] = 1.0;

  return (value);
}



template <int dim>
class RotatingCavityProblem : public HydrodynamicProblem<dim>
{
public:
  RotatingCavityProblem(ProblemParameters &parameters);

protected:
  virtual void make_grid() override;

  virtual void set_boundary_conditions() override;

  virtual void set_angular_velocity() override;

private:
  std::shared_ptr<const ConstantAngularVelocity<dim>> angular_velocity_ptr;

  const types::boundary_id  left_bndry_id;
  const types::boundary_id  right_bndry_id;
  const types::boundary_id  bottom_bndry_id;
  const types::boundary_id  top_bndry_id;
  const types::boundary_id  back_bndry_id;
  const types::boundary_id  front_bndry_id;

};



template <int dim>
RotatingCavityProblem<dim>::RotatingCavityProblem(ProblemParameters &parameters)
:
HydrodynamicProblem<dim>(parameters),
angular_velocity_ptr(new ConstantAngularVelocity<dim>()),
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
void RotatingCavityProblem<dim>::make_grid()
{
  std::cout << "    Make grid..." << std::endl;

  GridGenerator::hyper_cube(this->triangulation, 0.0, 1.0, true);

  this->triangulation.refine_global(this->n_initial_refinements);
}



template <int dim>
void RotatingCavityProblem<dim>::set_boundary_conditions()
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

  velocity_bcs.close();
  pressure_bcs.close();
}



template <int dim>
void RotatingCavityProblem<dim>::set_angular_velocity()
{
  this->solver.set_angular_velocity(angular_velocity_ptr);
}



int main(int argc, char *argv[])
{
  try
  {
    std::string parameter_filename;
    if (argc >= 2)
      parameter_filename = argv[1];
    else
      parameter_filename = "rotating_cavity_problem.prm";

    ProblemParameters parameters(parameter_filename);
    if (parameters.space_dim == 2)
    {
      RotatingCavityProblem<2> problem(parameters);
      problem.run();
    }
    else
    {
      RotatingCavityProblem<3> problem(parameters);
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
