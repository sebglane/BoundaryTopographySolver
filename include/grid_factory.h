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

template<int dim>
class SinusoidalManifold: public ChartManifold<dim,dim,dim-1>
{
public:

    SinusoidalManifold(const double         wavenumber = 2. * numbers::PI,
                       const double         amplitude = 0.1,
                       const unsigned int   normal_direction = 1,
                       const bool           single_wave = true,
                       const unsigned int   wave_direction = 0);

    virtual std::unique_ptr<Manifold<dim,dim>> clone() const;

    virtual Point<dim-1>    pull_back(const Point<dim> &space_point) const;

    virtual Point<dim>      push_forward(const Point<dim-1> &chart_point) const;

    virtual DerivativeForm<1,dim-1, dim> push_forward_gradient
    (const Point<dim-1> &chart_point) const;

private:

    const double    wavenumber;

    const double    amplitude;

    const bool      single_wave;

    const unsigned int  normal_direction;
    const unsigned int  wave_direction;
};


template<int dim>
class TopographyBox
{
public:
    TopographyBox(const double  wavenumber,
                  const double  amplitude,
                  const bool    single_wave = true,
                  const bool    include_exterior = false,
                  const double  exterior_length = 2.0);

    void create_coarse_mesh(Triangulation<dim> &coarse_grid);

private:
    const bool      single_wave;
    const bool      include_exterior;
    const double    exterior_length;

    SinusoidalManifold<dim>     sinus_manifold;

    TransfiniteInterpolationManifold<dim> interpolation_manifold;

    const double    tol = 1e-12;
};

}  // namespace GridFactory


#endif /* INCLUDE_GRID_FACTORY_H_ */
