/*
 * make_grid.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>

#include <equation_data.h>
#include <grid_factory.h>
#include <solver.h>

namespace TopographyProblem {

template<>
void TopographySolver<2>::make_grid()
{
    const unsigned int dim = 2;

    TimerOutput::Scope timer_section(computing_timer, "make grid");

    std::cout << "   Making grid..." << std::endl;

    GridFactory::TopographyBox<dim>     topography_mesh(2.*numbers::PI,
                                                        parameters.amplitude / parameters.wavelength);

    topography_mesh.create_coarse_mesh(triangulation);

    const unsigned int n_global_refinements = parameters.n_initial_refinements;
    // initial global refinements
    if (n_global_refinements > 0)
    {
        triangulation.refine_global(n_global_refinements);
        std::cout << "      Number of cells after "
                  << n_global_refinements
                  << " global refinements: "
                  << triangulation.n_active_cells()
                  << std::endl;
    }

    const unsigned int n_boundary_refinements = parameters.n_boundary_refinements;
    // initial boundary refinements
    if (n_boundary_refinements > 0)
    {
        for (unsigned int step=0; step<n_boundary_refinements; ++step)
        {
            for (auto cell: triangulation.active_cell_iterators())
                if (cell->at_boundary())
                    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                        if (cell->face(f)->boundary_id() == DomainIdentifiers::TopoBndry)
                            cell->set_refine_flag();
            triangulation.execute_coarsening_and_refinement();
        }
        std::cout << "      Number of cells after "
                  << n_boundary_refinements
                  << " interface refinements: "
                  << triangulation.n_active_cells()
                  << std::endl;
    }

    std::ofstream out("grid.vtk");
    GridOut grid_out;
    grid_out.write_vtk(triangulation, out);
    std::cout << "      Grid written to grid.vtk" << std::endl;
    out.close();
}

template<>
void TopographySolver<3>::make_grid()
{
    const unsigned int dim = 3;

    TimerOutput::Scope timer_section(computing_timer, "make grid");

    std::cout << "   Making grid..." << std::endl;

    Triangulation<dim-1>                plane_triangulation;

    GridFactory::TopographyBox<dim-1>   topography_mesh(2.*numbers::PI,
                                                        parameters.amplitude / parameters.wavelength);

    topography_mesh.create_coarse_mesh(plane_triangulation);

    const unsigned int n_global_refinements = parameters.n_initial_refinements;
    // initial global refinements
    if (n_global_refinements > 0)
    {
        plane_triangulation.refine_global(n_global_refinements);

        std::cout << "      Number of cells in 2D after "
                  << n_global_refinements
                  << " global refinements: "
                  << plane_triangulation.n_active_cells()
                  << std::endl;
    }

    const unsigned int n_boundary_refinements = parameters.n_boundary_refinements;
    // initial boundary refinements
    if (n_boundary_refinements > 0)
    {
        for (unsigned int step=0; step<n_boundary_refinements; ++step)
        {
            for (auto cell: plane_triangulation.active_cell_iterators())
                if (cell->at_boundary())
                    for (unsigned int f=0; f<GeometryInfo<dim-1>::faces_per_cell; ++f)
                        if (cell->face(f)->boundary_id() == DomainIdentifiers::TopoBndry)
                            cell->set_refine_flag();
            plane_triangulation.execute_coarsening_and_refinement();
        }
        std::cout << "      Number of cells in 2D after "
                  << n_boundary_refinements
                  << " interface refinements: "
                  << plane_triangulation.n_active_cells()
                  << std::endl;
    }

    std::ofstream out("plane-grid.msh");
    GridOut grid_out;
    grid_out.write_msh(plane_triangulation, out);
    std::cout << "      2D Grid written to plane-grid.msh" << std::endl;
    out.close();

    plane_triangulation.clear();

    GridIn<dim-1>   grid_in;
    grid_in.attach_triangulation(plane_triangulation);
    std::ifstream   in("plane-grid.msh");
    grid_in.read_msh(in);
    std::cout << "      2D Grid read from plane-grid.msh" << std::endl;
    in.close();

    const double depth = 0.1;
    GridGenerator::extrude_triangulation(plane_triangulation,
                                         4,
                                         depth,
                                         triangulation);

    triangulation.set_all_manifold_ids(0);
    triangulation.set_all_manifold_ids_on_boundary(0);

    const unsigned int normal_direction = 1;
    const unsigned int wave_direction = 0;

    GridFactory::SinusoidalManifold<dim> sinus_manifold(2.*numbers::PI,
                                                        parameters.amplitude / parameters.wavelength,
                                                        normal_direction,
                                                        true,
                                                        wave_direction);

    // assignment of manifold and boundary identifiers on topography
    const double tol = 1e-12;
    for (auto cell: triangulation.active_cell_iterators())
        if (cell->at_boundary())
            for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->at_boundary())
                {
                    std::vector<double> coord(GeometryInfo<dim>::vertices_per_face);
                    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                    {
                        const Point<dim>    vertex = cell->face(f)->vertex(v);
                        const Point<dim>    space_point = sinus_manifold.push_forward(sinus_manifold.pull_back(vertex));
                        coord[v] = vertex[normal_direction] - space_point[normal_direction];
                    }
                    if (std::all_of(coord.begin(), coord.end(),
                            [&](double d)->bool{return std::abs(d) < tol;}))
                    {
                        cell->face(f)->set_manifold_id(1);
                        // topography boundary
                        cell->face(f)->set_boundary_id(DomainIdentifiers::TopoBndry);

                    }
                }

    TransfiniteInterpolationManifold<dim>   interpolation_manifold;
    interpolation_manifold.initialize(triangulation);

    triangulation.set_manifold(0, interpolation_manifold);
    triangulation.set_manifold(1, sinus_manifold);

    // assignment of boundary identifiers
    for (auto cell: triangulation.active_cell_iterators())
        if (cell->at_boundary())
            for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->at_boundary())
                {
                    std::vector<std::vector<double>> coords(dim);
                    for (unsigned int d=0; d<dim; ++d)
                        coords[d].resize(GeometryInfo<dim>::vertices_per_face);
                    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                        for (unsigned int d=0; d<dim; ++d)
                            coords[d][v] = cell->face(f)->vertex(v)[d];
                    // x-coordinates
                    // left boundary
                    if (std::all_of(coords[0].begin(), coords[0].end(),
                            [&](double d)->bool{return std::abs(d) < tol;}))
                        cell->face(f)->set_boundary_id(DomainIdentifiers::Left);
                    // right boundary
                    else if (std::all_of(coords[0].begin(), coords[0].end(),
                            [&](double d)->bool{return std::abs(d - 1.0) < tol;}))
                        cell->face(f)->set_boundary_id(DomainIdentifiers::Right);

                    // y-coordinates
                    // bottom boundary
                    if (std::all_of(coords[1].begin(), coords[1].end(),
                            [&](double d)->bool{return std::abs(d) < tol;}))
                        cell->face(f)->set_boundary_id(DomainIdentifiers::Bottom);

                    // z-coordinates
                    // front boundary
                    if (std::all_of(coords[2].begin(), coords[2].end(),
                            [&](double d)->bool{return std::abs(d) < tol;}))
                        cell->face(f)->set_boundary_id(DomainIdentifiers::Front);
                    // back boundary
                    else if (std::all_of(coords[2].begin(), coords[2].end(),
                            [&](double d)->bool{return std::abs(d - depth) < tol;}))
                        cell->face(f)->set_boundary_id(DomainIdentifiers::Back);
                }

    out.open("grid.vtk");
    grid_out.write_vtk(triangulation, out);
    std::cout << "      Grid written to grid.vtk" << std::endl;
    out.close();
}
}  // namespace TopographyProblem

// explicit instantiation
template void TopographyProblem::TopographySolver<3>::make_grid();
