/*
 * equation_data.cc
 *
 *  Created on: Dec 7, 2018
 *      Author: sg
 */

#include <deal.II/lac/vector.h>

#include <equation_data.h>

namespace EquationData {

template<int dim>
VelocityBoundaryValues<dim>::VelocityBoundaryValues()
:
Function<dim>(dim)
{
    direction_vector[0] = 1.0;
}

template<int dim>
void VelocityBoundaryValues<dim>::vector_value(const Point<dim>    &/* point */,
                                               Vector<double>      &value) const
{
    Assert(value.size() == this->n_components,
           ExcDimensionMismatch(this->n_components, value.size()));
    for (unsigned int d=0; d<this->n_components; ++d)
        value[d] = -direction_vector[d];
}

template<int dim>
BackgroundVelocity<dim>::BackgroundVelocity()
:
Function<dim>(dim)
{
    direction_vector[0] = 1.0;
}

template<int dim>
void BackgroundVelocity<dim>::vector_value(const Point<dim>    &/* point */,
                                           Vector<double>      &value) const
{
    Assert(value.size() == this->n_components,
           ExcDimensionMismatch(this->n_components, value.size()));
    for (unsigned int d=0; d<this->n_components; ++d)
        value[d] = direction_vector[d];
}

template<int dim>
BackgroundMagneticField<dim>::BackgroundMagneticField()
:
Function<dim>(dim)
{
    direction_vector[dim-1] = 1.0;
}

template<int dim>
void BackgroundMagneticField<dim>::vector_value(const Point<dim>    &/* point */,
                                                Vector<double>      &value) const
{
    Assert(value.size() == this->n_components,
           ExcDimensionMismatch(this->n_components, value.size()));
    for (unsigned int d=0; d<this->n_components; ++d)
        value[d] = direction_vector[d];
}

}  // namespace EquationData

// explicit instantiation
template class EquationData::VelocityBoundaryValues<2>;
template class EquationData::VelocityBoundaryValues<3>;

template class EquationData::BackgroundVelocity<2>;
template class EquationData::BackgroundVelocity<3>;

template class EquationData::BackgroundMagneticField<3>;
