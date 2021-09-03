/*
 * equation_data.h
 *
 *  Created on: Dec 5, 2018
 *      Author: sg
 */

#ifndef INCLUDE_EQUATION_DATA_H_
#define INCLUDE_EQUATION_DATA_H_

#include <deal.II/base/function.h>

namespace DomainIdentifiers {
/*
 *
 * enumeration for boundary identifiers
 *
 */
enum BoundaryIds
{
    // topographic boundary
    TopoBndry,
    // inner core boundary
    ICB,
    // core mantle boundary
    CMB,
    // fictitious vacuum boundary
    FVB,
    // other boundaries
    Left,
    Right,
    Bottom,
    Front,
    Back
};

/*
 *
 * enumeration for material identifiers
 *
 */
enum MaterialIds
{
    Fluid,
    Vacuum,
    Solid
};
}  // namespace DomainIdentifiers

namespace EquationData {

using namespace dealii;

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
