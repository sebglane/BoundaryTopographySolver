/*
 * topography_mesh.cc
 *
 *  Created on: Sep 24, 2021
 *      Author: sg
 */
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <grid_factory.h>

#include <fstream>
#include <string>

using namespace dealii;

template<int dim>
void make_grid();

template <>
void make_grid<2>()
{
  std::cout << "    Make grid..." << std::endl;

  constexpr int dim{2};

  GridFactory::TopographyBox<dim> topography_box(2.0 * numbers::PI, 0.1);

  Triangulation<dim>  triangulation;
  topography_box.create_coarse_mesh(triangulation);

  triangulation.refine_global(4);

  std::string fname("Mesh2D.vtk");
  std::ofstream out(fname);

  GridOut().write(triangulation, out, GridOut::OutputFormat::vtk);
}



template <>
void make_grid<3>()
{
  std::cout << "    Make grid..." << std::endl;

  constexpr int dim{3};

  GridFactory::TopographyBox<dim> topography_box(2.0 * numbers::PI, 0.1);

  Triangulation<dim>  triangulation;
  topography_box.create_coarse_mesh(triangulation);

  for (unsigned int i=0; i<4; ++i)
  {
    triangulation.set_all_refine_flags();
    for (const auto &cell: triangulation.active_cell_iterators())
      if (cell->refine_flag_set())
        cell->set_refine_flag(RefinementCase<dim>::cut_xz);
    triangulation.execute_coarsening_and_refinement();
  }

  std::string fname("Mesh3D.vtk");
  std::ofstream out(fname);

  GridOut().write(triangulation, out, GridOut::OutputFormat::vtk);

}

int main(int /* argc */, char **/* argv[] */)
{
  try
  {
    make_grid<2>();
    make_grid<3>();
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
