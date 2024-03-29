/*
 * advection_problem.cc
 *
 *  Created on: Sep 1, 2021
 *      Author: sg
 */

#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_tools.h>

#include <advection_problem.h>
#include <grid_factory.h>

#include <memory>

namespace AdvectionProblem {

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
  std::shared_ptr<const ConstantTensorFunction<1, dim>> advection_field_ptr;

  std::shared_ptr<const SourceFunction<dim>> source_term_ptr;

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
advection_field_ptr(new ConstantTensorFunction<1,2>{Tensor<1, 2>({numbers::SQRT1_2,
                                                                  numbers::SQRT1_2})}),
source_term_ptr(new SourceFunction<2>{2.0 * numbers::PI}),
left_bndry_id(numbers::invalid_boundary_id),
right_bndry_id(numbers::invalid_boundary_id),
bottom_bndry_id(numbers::invalid_boundary_id),
top_bndry_id(numbers::invalid_boundary_id),
topographic_bndry_id(numbers::invalid_boundary_id),
back_bndry_id(numbers::invalid_boundary_id),
front_bndry_id(numbers::invalid_boundary_id)
{
  std::cout << "Solving advection problem with topography" << std::endl;
}



template <>
Problem<3>::Problem(ProblemParameters &parameters)
:
AdvectionProblem<3>(parameters),
advection_field_ptr(new ConstantTensorFunction<1,3>{Tensor<1, 3>({1.0 / std::sqrt(3.0),
                                                                  1.0 / std::sqrt(3.0),
                                                                  1.0 / std::sqrt(3.0)})}),
source_term_ptr(new SourceFunction<3>{2.0 * numbers::PI}),
left_bndry_id(numbers::invalid_boundary_id),
right_bndry_id(numbers::invalid_boundary_id),
bottom_bndry_id(numbers::invalid_boundary_id),
top_bndry_id(numbers::invalid_boundary_id),
topographic_bndry_id(numbers::invalid_boundary_id),
back_bndry_id(numbers::invalid_boundary_id),
front_bndry_id(numbers::invalid_boundary_id)
{
  std::cout << "Solving advection problem with topography" << std::endl;
}



template <int dim>
void Problem<dim>::set_advection_field()
{
  this->solver.set_advection_field(advection_field_ptr);
}



template <int dim>
void Problem<dim>::set_source_term()
{
  this->solver.set_source_term(source_term_ptr);
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

  this->triangulation.refine_global(this->n_initial_refinements);

  // initial boundary refinements
  if (this->n_initial_bndry_refinements > 0)
  {
    for (unsigned int step=0; step<this->n_initial_bndry_refinements; ++step)
    {
      for (const auto &cell: this->triangulation.active_cell_iterators())
        if (cell->at_boundary())
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->face(f)->boundary_id() == topographic_bndry_id)
              cell->set_refine_flag();
      this->triangulation.execute_coarsening_and_refinement();
    }
  }

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

}  // namespace AdvectionProblem

int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

    std::string parameter_filename;
    if (argc >= 2)
      parameter_filename = argv[1];
    else
      parameter_filename = "advection_problem.prm";

    Advection::ProblemParameters parameters(parameter_filename);
    if (parameters.space_dim == 2)
    {
      AdvectionProblem::Problem<2> problem(parameters);
      problem.run();
    }
    else
    {
      AdvectionProblem::Problem<3> problem(parameters);
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
