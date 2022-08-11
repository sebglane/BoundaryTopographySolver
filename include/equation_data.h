/*
 * equation_data.h
 *
 *  Created on: Dec 5, 2018
 *      Author: sg
 */

#ifndef INCLUDE_EQUATION_DATA_H_
#define INCLUDE_EQUATION_DATA_H_

#include <deal.II/base/function.h>


namespace EquationData {

using namespace dealii;

/**
 * @todo Add documentation.
 */
template<int dim>
class VelocityBoundaryValues : public Function<dim>
{
public:
    VelocityBoundaryValues();

    virtual void    vector_value(const Point<dim>   &point,
                                 Vector<double>     &value) const;

private:
    Tensor<1,dim>           direction_vector;
};

/**
 * @todo Add documentation.
 */
template<int dim>
class BackgroundVelocity : public Function<dim>
{
public:
    BackgroundVelocity();

    virtual void    vector_value(const Point<dim>   &point,
                                 Vector<double>     &value) const;

private:
    Tensor<1,dim>           direction_vector;
};

/**
 * @todo Add documentation.
 */
template<int dim>
class BackgroundMagneticField : public Function<dim>
{
public:
    BackgroundMagneticField();

    virtual void    vector_value(const Point<dim>   &point,
                                 Vector<double>     &value) const;

private:
    Tensor<1,dim>           direction_vector;
};
}  // namespace EquationData




#endif /* INCLUDE_EQUATION_DATA_H_ */
