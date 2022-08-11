/*
 * grid_factory.h
 *
 *  Created on: Nov 21, 2018
 *      Author: sg
 */

#ifndef INCLUDE_GRID_FACTORY_H_
#define INCLUDE_GRID_FACTORY_H_

#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/manifold_lib.h>

namespace GridFactory {

using namespace dealii;

/**
 * @todo Add documentation.
 */
template<int dim>
class SinusoidalManifold: public ChartManifold<dim,dim,dim-1>
{
public:
  SinusoidalManifold(const double         wavenumber,
                     const double         amplitude,
                     const double         offset,
                     const double         angle = 0.0,
                     const unsigned int   topography_coordinate = dim-1,
                     const bool           plane_wave = true);

  virtual std::unique_ptr<Manifold<dim,dim>> clone() const;

  virtual Point<dim-1>    pull_back(const Point<dim> &space_point) const;

  virtual Point<dim>      push_forward(const Point<dim-1> &chart_point) const;

  virtual DerivativeForm<1,dim-1, dim> push_forward_gradient
  (const Point<dim-1> &chart_point) const;

private:
  void initialize_wavenumber_vector();

  const double    wavenumber;

  const double    amplitude;

  const double    offset;

  const double    angle;

  const unsigned int  topography_coordinate;

  const bool      plane_wave;

  std::vector<Tensor<1, dim-1>>  wavenumber_vectors;
};


template<int dim>
class TopographyBox
{
public:
  TopographyBox(const double  wavenumber,
                const double  amplitude,
                const double  angle = 0.0,
                const bool    plane_wave = true,
                const bool    include_exterior = false,
                const double  exterior_length = 2.0);

  /*!
   * @enum BoundaryIds
   *
   * @brief Enumeration representing boundary identifiers.
   */
  enum BoundaryIds: types::boundary_id
  {
    left = 0,
    right = 1,
    bottom = 2,
    top = 3,
    back = 4,
    front = 5,
    // topographic boundary
    topographic_boundary = 6,
    // interior topographic boundary
    interior_topographic_boundary = 7
  };

  static const types::material_id  fluid{0};
  static const types::material_id  other{1};

  static const types::manifold_id  sinusoidal{1};
  static const types::manifold_id  interpolation{2};

  void create_coarse_mesh(Triangulation<dim> &coarse_grid);

private:
  const bool      plane_wave;

  const bool      include_exterior;

  const double    exterior_length;

  SinusoidalManifold<dim> sinusoidal_manifold;

  TransfiniteInterpolationManifold<dim> interpolation_manifold;

  static constexpr double  tol{1e-12};
};

}  // namespace GridFactory


#endif /* INCLUDE_GRID_FACTORY_H_ */
