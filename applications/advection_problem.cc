/*
 * cavity_problem.cc
 *
 *  Created on: Sep 1, 2021
 *      Author: sg
 */

#include <deal.II/base/function_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/grid/grid_tools.h>

#include <advection_problem.h>
#include <grid_factory.h>

namespace TopographyProblem {

using namespace Advection;

template <int dim>
class SourceFunction : public Function<dim>
{
public:
  SourceFunction(const double wavenumber,
                 const double offset = 1.0);

  virtual double value(const Point<dim>  &point,
                       const unsigned int component) const;

private:
  const double wavenumber;

  const double offset;

};



template <int dim>
SourceFunction<dim>::SourceFunction
(const double wavenumber,
 const double offset)
:
Function<dim>(),
wavenumber(wavenumber),
offset(offset)
{}



template <int dim>
double SourceFunction<dim>::value
(const Point<dim>  &point,
 const unsigned int /* component */) const
{
  double coord{point[dim-1]};
  coord -= offset;

  return std::sin(wavenumber * point[0]) * std::exp(4.0 * coord);
}



template <int dim>
class Problem : public AdvectionProblem<dim>
{
public:
  Problem(ProblemParameters &parameters);

protected:
  virtual void make_grid() override;

  virtual void set_boundary_conditions() override;

  virtual void set_advection_field() override;

  virtual void set_source_term() override;

private:
  ConstantTensorFunction<1, dim>  advection_field;

  SourceFunction<dim>             source_term;

  types::boundary_id  left_bndry_id;
  types::boundary_id  right_bndry_id;
  types::boundary_id  bottom_bndry_id;
  types::boundary_id  top_bndry_id;
  types::boundary_id  topographic_bndry_id;
  types::boundary_id  back_bndry_id;
  types::boundary_id  front_bndry_id;
};



template <>
Problem<2>::Problem(ProblemParameters &parameters)
:
AdvectionProblem<2>(parameters),
advection_field(Tensor<1, 2>({numbers::SQRT1_2, numbers::SQRT1_2})),
source_term(2.0 * numbers::PI)
{
  std::cout << "Solving viscous buoyant topography problem" << std::endl;
}



template <>
Problem<3>::Problem(ProblemParameters &parameters)
:
AdvectionProblem<3>(parameters),
advection_field(Tensor<1, 3>({1.0 / std::sqrt(3.0),
                              1.0 / std::sqrt(3.0),
                              1.0 / std::sqrt(3.0)})),
source_term(2.0 * numbers::PI)
{
  std::cout << "Solving viscous buoyant topography problem" << std::endl;
}



template <int dim>
void Problem<dim>::set_advection_field()
{
  this->solver.set_advection_field(advection_field);
}



template <int dim>
void Problem<dim>::set_source_term()
{
  this->solver.set_source_term(source_term);
}



template <int dim>
void Problem<dim>::make_grid()
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
void Problem<dim>::set_boundary_conditions()
{
  std::cout << "    Set boundary conditions..." << std::endl;

  ScalarBoundaryConditions<dim> &bcs = this->solver.get_bcs();

  bcs.clear();

  bcs.extract_boundary_ids();

  bcs.set_periodic_bc(left_bndry_id, right_bndry_id, 0);

  const std::shared_ptr<Function<dim>> function =
      std::make_shared<Functions::ConstantFunction<dim>>(1.0);
  bcs.set_dirichlet_bc(topographic_bndry_id, function);

  if (dim == 2)
  {
    bcs.set_dirichlet_bc(bottom_bndry_id);
  }
  else if (dim == 3)
  {
    bcs.set_periodic_bc(bottom_bndry_id, top_bndry_id, 1);

    bcs.set_dirichlet_bc(back_bndry_id);
  }

  bcs.close();
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
      parameter_filename = "advection_problem.prm";

    TopographyProblem::ProblemParameters parameters(parameter_filename);
    if (parameters.space_dim == 2)
    {
      TopographyProblem::Problem<2> problem(parameters);
      problem.run();
    }
    else
    {
      TopographyProblem::Problem<3> problem(parameters);
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
