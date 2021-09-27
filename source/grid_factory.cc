/*
 * grid_factor.cc
 *
 *  Created on: Nov 21, 2018
 *      Author: sg
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/grid/grid_generator.h>

#include <grid_factory.h>

#include <algorithm>
#include <cmath>

namespace GridFactory
{

template <int dim>
SinusoidalManifold<dim>::SinusoidalManifold
(const double        wavenumber_,
 const double        amplitude_,
 const double        offset_,
 const double        angle_,
 const unsigned int  topography_coordinate_,
 const bool          plane_wave_)
:
ChartManifold<dim,dim,dim-1>(),
wavenumber(wavenumber_),
amplitude(amplitude_),
offset(offset_),
angle(angle_),
topography_coordinate(topography_coordinate_),
plane_wave(plane_wave_)
{
  AssertIsFinite(wavenumber);

  AssertIsFinite(amplitude);

  AssertIsFinite(offset);
  Assert(0.0 < offset,
         ExcLowerRangeType<double>(0.0, offset));

  AssertIsFinite(angle);
  Assert(0.0 <= angle,
         ExcLowerRangeType<double>(0, angle));

  Assert(topography_coordinate < dim,
         ExcLowerRange(topography_coordinate, dim));

  if (!plane_wave)
    Assert(dim == 3, ExcImpossibleInDim(dim));

  initialize_wavenumber_vector();
}



template <>
void SinusoidalManifold<2>::initialize_wavenumber_vector()
{
  constexpr int dim{2};
  wavenumber_vectors.push_back(Tensor<1, dim-1>({wavenumber}));
}



template <>
void SinusoidalManifold<3>::initialize_wavenumber_vector()
{
  constexpr int dim{3};
  wavenumber_vectors.push_back(Tensor<1, dim-1>({std::cos(angle), std::sin(angle)}));
  if (!plane_wave)
    wavenumber_vectors.push_back(Tensor<1, dim-1>({-std::sin(angle), std::cos(angle)}));

  for (auto &vector: wavenumber_vectors)
    vector *= wavenumber;
}



template <int dim>
std::unique_ptr<Manifold<dim,dim>> SinusoidalManifold<dim>::clone() const
{
  return std::make_unique<SinusoidalManifold<dim>>(wavenumber,
                                                   amplitude,
                                                   offset,
                                                   angle,
                                                   topography_coordinate,
                                                   plane_wave);
}



template<int dim>
Point<dim-1> SinusoidalManifold<dim>::pull_back(const Point<dim> &space_point) const
{
  Point<dim-1> chart_point;
  if (topography_coordinate == 0)
  {
    for (unsigned int d=1; d<dim; ++d)
      chart_point[d] = space_point[d];

    return (chart_point);
  }
  else if (topography_coordinate == dim -1)
  {
    for (unsigned int d=0; d<dim-1; ++d)
      chart_point[d] = space_point[d];

    return (chart_point);
  }
  else if (topography_coordinate == 1)
  {
    AssertThrow(dim == 3, ExcImpossibleInDim(dim));

    chart_point[0] = space_point[0];
    chart_point[1] = space_point[2];

    return (chart_point);
  }
  else
    Assert(false, ExcInternalError());
}



template<int dim>
Point<dim> SinusoidalManifold<dim>::push_forward(const Point<dim-1> &chart_point) const
{
  Point<dim> space_point;
  if (topography_coordinate == 0)
  {
    space_point[0] = amplitude;
    for (auto &vector: wavenumber_vectors)
      space_point[0] *= std::sin(vector * chart_point);
    space_point[0] += offset;

    for (unsigned int d=1; d<dim; ++d)
      space_point[d] = chart_point[d];

    return space_point;
  }
  else if (topography_coordinate == dim -1)
  {
    space_point[dim-1] = amplitude;
    for (auto &vector: wavenumber_vectors)
      space_point[dim-1] *= std::sin(vector * chart_point);
    space_point[dim-1] += offset;

    for (unsigned int d=0; d<dim-1; ++d)
      space_point[d] = chart_point[d];

    return space_point;
  }
  else if (topography_coordinate == 1)
  {
    AssertThrow(dim == 3, ExcImpossibleInDim(dim));

    space_point[1] = amplitude;
    for (auto &vector: wavenumber_vectors)
      space_point[1] *= std::sin(vector * chart_point);
    space_point[1] += offset;

    space_point[0] = chart_point[0];
    space_point[2] = chart_point[1];

    return space_point;
  }
  else
    Assert(false, ExcNotImplemented());
}



template<int dim>
DerivativeForm<1, dim-1, dim> SinusoidalManifold<dim>::push_forward_gradient
(const Point<dim-1> &chart_point) const
{
  DerivativeForm<1, dim-1, dim> F;

  auto compute_derivative = [&, this](Tensor<1, dim-1> &tensor)
  {
    for (unsigned d=0; d<dim-1; ++d)
    {
      tensor[d] = this->amplitude;

      for (std::size_t i=0; i<this->wavenumber_vectors.size(); ++i)
      {
        tensor[d] *= this->wavenumber_vectors[i][d] * std::cos(this->wavenumber_vectors[i] * chart_point);

        for (std::size_t j=0; j<this->wavenumber_vectors.size(); ++j)
          if (i==j)
            continue;
          else
            tensor[d] *= std::sin(this->wavenumber_vectors[j] * chart_point);
      }
    }
  };

  if (topography_coordinate == 0)
  {
    compute_derivative(F[0]);
    for (unsigned int d=1; d<dim; ++d)
      F[d][d-1] = 1.0;

    return F;
  }
  else if (topography_coordinate == dim-1)
  {
    for (unsigned int d=0; d<dim-1; ++d)
      F[d][d] = 1.0;

    compute_derivative(F[dim-1]);

    return F;
  }
  else if (topography_coordinate == 1)
  {
    AssertThrow(dim == 3, ExcImpossibleInDim(dim));

    F[0][0] = 1.0;
    F[2][1] = 1.0;

    compute_derivative(F[1]);

    return F;
  }
  else
    Assert(false, ExcNotImplemented());
}



template<int dim>
TopographyBox<dim>::TopographyBox
(const double  wavenumber,
 const double  amplitude,
 const double  angle,
 const bool    plane_wave,
 const bool    include_exterior,
 const double  exterior_length)
:
plane_wave(plane_wave),
include_exterior(include_exterior),
exterior_length(exterior_length),
sinusoidal_manifold(wavenumber, amplitude, 1.0, angle, dim-1, plane_wave)
{
  Assert(amplitude < 0.5, ExcLowerRangeType<double>(amplitude, 0.5));

  Assert(amplitude < exterior_length, ExcLowerRangeType<double>(amplitude, exterior_length));
}



template<int dim>
void TopographyBox<dim>::create_coarse_mesh(Triangulation<dim> &coarse_grid)
{
  if (!include_exterior)
  {
    const Point<dim>    origin;

    Point<dim>    corner;
    if (dim == 2)
    {
      for (unsigned int d=0; d<dim; ++d)
        corner[d] = 1.0;
    }
    else if (dim == 3)
    {
      if (plane_wave)
      {
        corner[0] = 1.0;
        corner[1] = 0.1;
        corner[2] = 1.0;
      }
      else
        for (unsigned int d=0; d<dim; ++d)
          corner[d] = 1.0;
    }

    GridGenerator::hyper_rectangle(coarse_grid,
                                   origin,
                                   corner);

    coarse_grid.set_all_manifold_ids(interpolation);
    coarse_grid.set_all_manifold_ids_on_boundary(interpolation);

    for (const auto &cell: coarse_grid.active_cell_iterators())
    {
      cell->set_material_id(fluid);

      if (cell->at_boundary())
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary())
          {
            std::vector<double> coord(GeometryInfo<dim>::vertices_per_face);
            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
              coord[v] = face->vertex(v)[dim-1];

            if (std::all_of(coord.begin(), coord.end(),
                [&](double d)->bool{return std::abs(d - 1.0) < tol;}))
            {
              face->set_boundary_id(topographic_boundary);
              face->set_manifold_id(sinusoidal);
              break;
            }
          }
    }
    interpolation_manifold.initialize(coarse_grid);
    coarse_grid.set_manifold(interpolation, interpolation_manifold);
    coarse_grid.set_manifold(sinusoidal, sinusoidal_manifold);
  }
  else if (include_exterior)
  {
    const Point<dim> origin;

    Point<dim>    corner;
    std::vector<unsigned int> repetitions(dim);
    for (unsigned int d=0; d<dim; ++d)
      corner[d] = 1.0;
    corner[dim-1] = exterior_length + 1.0;

    std::vector<std::vector<double>> step_sizes;
    for (unsigned int d=0; d<dim-1; ++d)
      step_sizes.push_back(std::vector<double>(1, 1.0));

    step_sizes.push_back(std::vector<double>{1.0, exterior_length});

    GridGenerator::subdivided_hyper_rectangle(coarse_grid,
                                              step_sizes,
                                              origin,
                                              corner);

    coarse_grid.set_all_manifold_ids(interpolation);
    coarse_grid.set_all_manifold_ids_on_boundary(interpolation);

    for (const auto &cell: coarse_grid.active_cell_iterators())
    {
      if (cell->center()[dim-1] < 1.0)
        cell->set_material_id(fluid);
      else if (cell->center()[dim-1] > 1.0)
        cell->set_material_id(other);

      if (cell->at_boundary())
        for (const auto &face : cell->face_iterators())
          if (!face->at_boundary())
          {
            std::vector<double> coord(GeometryInfo<dim>::vertices_per_face);
            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
              coord[v] = face->vertex(v)[dim-1];

            if (std::all_of(coord.begin(), coord.end(),
                [&](double d)->bool{return std::abs(d - 1.0) < tol;}))
            {
              face->set_manifold_id(sinusoidal);
            }
          }
    }
    interpolation_manifold.initialize(coarse_grid);
    coarse_grid.set_manifold(interpolation, interpolation_manifold);
    coarse_grid.set_manifold(sinusoidal, sinusoidal_manifold);
  }
  else
      Assert(false, ExcInternalError());

  // assignment of boundary identifiers
  for (const auto &cell: coarse_grid.active_cell_iterators())
    if (cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
        {
          std::vector<double> coords(GeometryInfo<dim>::vertices_per_face);

          // x-coordinates
          for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
            coords[v] = face->vertex(v)[0];
          // left boundary
          if (std::all_of(coords.begin(), coords.end(),
              [&](double d)->bool{return std::abs(d) < tol;}))
            face->set_boundary_id(left);
          // right boundary
          else if (std::all_of(coords.begin(), coords.end(),
                   [&](double d)->bool{return std::abs(d - 1.0) < tol;}))
            face->set_boundary_id(right);

          switch (dim)
          {
            case 2:
            {
              // y-coordinates
              for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                coords[v] = face->vertex(v)[1];
              const double height = (include_exterior ? 1.0 + exterior_length: 1.0);
              // bottom boundary
              if (std::all_of(coords.begin(), coords.end(),
                              [&](double d)->bool{return std::abs(d) < tol;}))
                face->set_boundary_id(bottom);
              // top boundary
              else if (std::all_of(coords.begin(), coords.end(),
                                   [&](double d)->bool{return std::abs(d - height) < tol;}) &&
                       include_exterior)
                face->set_boundary_id(top);
              break;
            }
            case 3:
            {
              const double height{include_exterior ? 1.0 + exterior_length: 1.0};
              const double depth{plane_wave ? 0.1 : 1.0};

              // y-coordinate
              for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                coords[v] = face->vertex(v)[1];
              // back boundary
              if (std::all_of(coords.begin(), coords.end(),
                              [&](double d)->bool{return std::abs(d) < tol;}))
                face->set_boundary_id(bottom);
              // front boundary
              else if (std::all_of(coords.begin(), coords.end(),
                                   [&](double d)->bool{return std::abs(d - depth) < tol;}))
                face->set_boundary_id(top);

              // z-coordinate
              for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                coords[v] = face->vertex(v)[2];
              // bottom boundary
              if (std::all_of(coords.begin(), coords.end(),
                              [&](double d)->bool{return std::abs(d) < tol;}))
                face->set_boundary_id(back);
              // top boundary
              else if (std::all_of(coords.begin(), coords.end(),
                                   [&](double d)->bool{return std::abs(d - height) < tol;}) &&
                       include_exterior)
                face->set_boundary_id(front);
              break;
            }
            default:
              Assert(false, ExcImpossibleInDim(dim));
              break;
          } // switch
        } // loop over faces
}


// explicit instantiations
template class SinusoidalManifold<2>;
template class SinusoidalManifold<3>;

template class TopographyBox<2>;
template class TopographyBox<3>;

}  // namespace GridFactory

