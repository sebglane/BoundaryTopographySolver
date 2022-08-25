/*
 * magnetic_induction_problem.cc
 *
 *  Created on: Aug 23, 2022
 *      Author: sg
 */

#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_tools.h>

#include <magnetic_induction_problem.h>
#include <grid_factory.h>

#include <memory>

namespace MagneticInductionProblem {

using namespace MagneticInduction;


template <int dim>
class Problem : public MagneticInductionProblem<dim>
{
public:
  Problem(ProblemParameters &parameters);

protected:
  virtual void make_grid() override;

  virtual void set_boundary_conditions() override;

  virtual void set_velocity_field() override;

private:
  std::shared_ptr<const ConstantTensorFunction<1, dim>> velocity_field_ptr;

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
MagneticInductionProblem<2>(parameters),
velocity_field_ptr(new ConstantTensorFunction<1,2>{Tensor<1, 2>({numbers::SQRT1_2,
                                                                 numbers::SQRT1_2})}),
left_bndry_id(numbers::invalid_boundary_id),
right_bndry_id(numbers::invalid_boundary_id),
bottom_bndry_id(numbers::invalid_boundary_id),
top_bndry_id(numbers::invalid_boundary_id),
topographic_bndry_id(numbers::invalid_boundary_id),
back_bndry_id(numbers::invalid_boundary_id),
front_bndry_id(numbers::invalid_boundary_id)
{
  std::cout << "Solving magnetic induction problem with topography" << std::endl;
}



template <>
Problem<3>::Problem(ProblemParameters &parameters)
:
MagneticInductionProblem<3>(parameters),
velocity_field_ptr(new ConstantTensorFunction<1,3>{Tensor<1, 3>({1.0 / std::sqrt(3.0),
                                                                  1.0 / std::sqrt(3.0),
                                                                  1.0 / std::sqrt(3.0)})}),
left_bndry_id(numbers::invalid_boundary_id),
right_bndry_id(numbers::invalid_boundary_id),
bottom_bndry_id(numbers::invalid_boundary_id),
top_bndry_id(numbers::invalid_boundary_id),
topographic_bndry_id(numbers::invalid_boundary_id),
back_bndry_id(numbers::invalid_boundary_id),
front_bndry_id(numbers::invalid_boundary_id)
{
  std::cout << "Solving magnetic induction problem with topography" << std::endl;
}



template <int dim>
void Problem<dim>::set_velocity_field()
{
  this->solver.set_velocity_field(velocity_field_ptr);
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

  // magnetic field bcs
  {
    VectorBoundaryConditions<dim> &bcs = this->solver.get_magnetic_field_bcs();

    bcs.clear();

    bcs.extract_boundary_ids();

    bcs.set_periodic_bc(left_bndry_id, right_bndry_id, 0);

    std::vector<double> value(dim);
    value[0] = -1.0 / numbers::SQRT2;
    value[1] = 1.0 / numbers::SQRT2;
    const std::shared_ptr<Function<dim>> magnetic_field_function =
        std::make_shared<Functions::ConstantFunction<dim>>(value);

    if constexpr(dim == 2)
    {
      bcs.set_tangential_flux_bc(bottom_bndry_id, magnetic_field_function);
      bcs.set_tangential_flux_bc(topographic_bndry_id);
    }
    else if constexpr(dim == 3)
    {
      bcs.set_periodic_bc(bottom_bndry_id, top_bndry_id, 1);

      bcs.set_tangential_flux_bc(back_bndry_id, magnetic_field_function);
      bcs.set_tangential_flux_bc(topographic_bndry_id);
    }

    bcs.close();
  }

  // magnetic pressure bcs
  {
    ScalarBoundaryConditions<dim> &bcs = this->solver.get_magnetic_pressure_bcs();

    bcs.clear();

    bcs.extract_boundary_ids();

    bcs.set_periodic_bc(left_bndry_id, right_bndry_id, 0);

    if constexpr(dim == 2)
    {
      bcs.set_dirichlet_bc(bottom_bndry_id);
      bcs.set_dirichlet_bc(topographic_bndry_id);
    }
    else if constexpr(dim == 3)
    {
      bcs.set_periodic_bc(bottom_bndry_id, top_bndry_id, 1);

      bcs.set_dirichlet_bc(back_bndry_id);
      bcs.set_dirichlet_bc(topographic_bndry_id);
    }

    bcs.close();
  }
}

}  // namespace MagneticInductionProblem



int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

    std::string parameter_filename;
    if (argc >= 2)
      parameter_filename = argv[1];
    else
      parameter_filename = "magnetic_induction_problem.prm";

    MagneticInduction::ProblemParameters parameters(parameter_filename);
    if (parameters.space_dim == 2)
    {
      MagneticInductionProblem::Problem<2> problem(parameters);
      problem.run();
    }
    else
    {
      MagneticInductionProblem::Problem<3> problem(parameters);
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
