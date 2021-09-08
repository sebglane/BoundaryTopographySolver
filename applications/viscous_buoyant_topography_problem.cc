/*
 * cavity_problem.cc
 *
 *  Created on: Sep 1, 2021
 *      Author: sg
 */

#include <deal.II/base/function_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/grid/grid_tools.h>

#include <buoyant_hydrodynamic_problem.h>
#include <grid_factory.h>

namespace TopographyProblem {

using namespace BuoyantHydrodynamic;

template <int dim>
class ReferenceDensity : public Function<dim>
{
public:
  ReferenceDensity();

  virtual void gradient_list(const std::vector<Point<dim>> &points,
                             std::vector<Tensor<1, dim>>   &gradients,
                             const unsigned int    component = 0) const;
};



template <int dim>
ReferenceDensity<dim>::ReferenceDensity()
:
Function<dim>(1)
{}



template <int dim>
void ReferenceDensity<dim>::gradient_list
(const std::vector<Point<dim>> &/* points */,
 std::vector<Tensor<1, dim>>   &gradients,
 const unsigned int /* component */) const
{
  Tensor<1, dim> gradient_value;
  gradient_value[dim-1] = -1.0;

  for (auto &gradient: gradients)
    gradient = gradient_value;

  return;
}



template <int dim>
class ViscousProblem : public BuoyantHydrodynamicProblem<dim>
{
public:
  ViscousProblem(ProblemParameters &parameters);

protected:
  virtual void make_grid() override;

  virtual void set_boundary_conditions() override;

  virtual void set_gravity_field() override;

  virtual void set_reference_density() override;

private:
  ConstantTensorFunction<1, dim>  gravity_field;

  ReferenceDensity<dim> reference_density;

  types::boundary_id  left_bndry_id;
  types::boundary_id  right_bndry_id;
  types::boundary_id  bottom_bndry_id;
  types::boundary_id  top_bndry_id;
  types::boundary_id  topographic_bndry_id;
  types::boundary_id  back_bndry_id;
  types::boundary_id  front_bndry_id;
};



template <>
ViscousProblem<2>::ViscousProblem(ProblemParameters &parameters)
:
BuoyantHydrodynamicProblem<2>(parameters),
gravity_field(Tensor<1, 2>({0.0, -1.0})),
reference_density()
{
  std::cout << "Solving viscous buoyant topography problem" << std::endl;
}



template <>
ViscousProblem<3>::ViscousProblem(ProblemParameters &parameters)
:
BuoyantHydrodynamicProblem<3>(parameters),
gravity_field(Tensor<1, 3>({0.0, 0.0, -1.0})),
reference_density()
{
  std::cout << "Solving viscous buoyant topography problem" << std::endl;
}



template <int dim>
void ViscousProblem<dim>::set_gravity_field()
{
  this->solver.set_gravity_field(gravity_field);
}



template <int dim>
void ViscousProblem<dim>::set_reference_density()
{
  this->solver.set_reference_density(reference_density);
}



template <int dim>
void ViscousProblem<dim>::make_grid()
{
  std::cout << "    Make grid..." << std::endl;

  GridFactory::TopographyBox<dim> topography_box(2.0 * numbers::PI, 0.1);
  left_bndry_id = topography_box.left;
  right_bndry_id = topography_box.right;
  bottom_bndry_id = topography_box.bottom;
  top_bndry_id = topography_box.top;
  back_bndry_id = topography_box.back;
  front_bndry_id = topography_box.front;

  topographic_bndry_id = topography_box.topographic_boundary;

  topography_box.create_coarse_mesh(this->triangulation);

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
  periodicity_vector;

  GridTools::collect_periodic_faces(this->triangulation,
                                    left_bndry_id,
                                    right_bndry_id,
                                    0,
                                    periodicity_vector);
  if (dim == 3)
    GridTools::collect_periodic_faces(this->triangulation,
                                      bottom_bndry_id,
                                      top_bndry_id,
                                      1,
                                      periodicity_vector);

  this->triangulation.add_periodicity(periodicity_vector);

  this->triangulation.refine_global(4);

}



template <int dim>
void ViscousProblem<dim>::set_boundary_conditions()
{
  std::cout << "    Set boundary conditions..." << std::endl;

  VectorBoundaryConditions<dim> &velocity_bcs = this->solver.get_velocity_bcs();
  ScalarBoundaryConditions<dim> &pressure_bcs = this->solver.get_pressure_bcs();
  ScalarBoundaryConditions<dim> &density_bcs = this->solver.get_density_bcs();

  velocity_bcs.clear();
  pressure_bcs.clear();
  density_bcs.clear();

  velocity_bcs.extract_boundary_ids();
  pressure_bcs.extract_boundary_ids();
  density_bcs.extract_boundary_ids();

  velocity_bcs.set_periodic_bc(left_bndry_id, right_bndry_id, 0);
  pressure_bcs.set_periodic_bc(left_bndry_id, right_bndry_id, 0);
  density_bcs.set_periodic_bc(left_bndry_id, right_bndry_id, 0);

  std::vector<double> value(dim);
  value[0] = 1.0;
  const std::shared_ptr<Function<dim>> velocity_function =
      std::make_shared<Functions::ConstantFunction<dim>>(value);

  if (dim == 2)
  {
    velocity_bcs.set_dirichlet_bc(bottom_bndry_id, velocity_function);
    velocity_bcs.set_normal_flux_bc(topographic_bndry_id);

    density_bcs.set_dirichlet_bc(bottom_bndry_id);
  }
  else if (dim == 3)
  {
    velocity_bcs.set_periodic_bc(bottom_bndry_id, top_bndry_id, 1);
    pressure_bcs.set_periodic_bc(bottom_bndry_id, top_bndry_id, 1);
    density_bcs.set_periodic_bc(bottom_bndry_id, top_bndry_id, 1);

    velocity_bcs.set_dirichlet_bc(back_bndry_id, velocity_function);
    velocity_bcs.set_normal_flux_bc(topographic_bndry_id);

    density_bcs.set_dirichlet_bc(back_bndry_id);
  }

  velocity_bcs.close();
  pressure_bcs.close();
  density_bcs.close();
}

}  // namespace TopographyProblem

int main(int argc, char *argv[])
{
  try
  {
    std::string parameter_filename;
    if (argc >= 2)
      parameter_filename = argv[1];
    else
      parameter_filename = "viscous_buoyant_topography_problem.prm";

    TopographyProblem::ProblemParameters parameters(parameter_filename);

    TopographyProblem::ViscousProblem<2> problem(parameters);
    problem.run();
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