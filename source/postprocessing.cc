/*
 * postprocessing.cc
 *
 *  Created on: Dec 13, 2018
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>

#include <equation_data.h>
#include <solver.h>

namespace TopographyProblem {

template<int dim>
Tensor<1,dim> TopographySolver<dim>::compute_boundary_traction() const
{
    Tensor<1,dim>       traction;
    double              surface_area = 0;

    QGauss<dim-1>       face_quadrature(parameters.velocity_degree + 1);

    FEFaceValues<dim>   fe_face_values(fe_system,
                                       face_quadrature,
                                       update_values|
                                       update_normal_vectors|
                                       update_JxW_values);

    const FEValuesExtractors::Scalar    pressure(dim+1);

    const unsigned int      n_face_q_points = face_quadrature.size();
    std::vector<double>     present_pressure_values(n_face_q_points);

    for (auto cell: dof_handler.active_cell_iterators())
        if (cell->at_boundary())
            for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
                if (cell->face(face_number)->at_boundary() &&
                        cell->face(face_number)->boundary_id() == DomainIdentifiers::TopoBndry)
                {
                    fe_face_values.reinit(cell, face_number);

                    fe_face_values[pressure].get_function_values(present_solution,
                                                                 present_pressure_values);

                    const std::vector<Tensor<1,dim>> normal_vectors = fe_face_values.get_normal_vectors();

                    for (unsigned int q=0; q<n_face_q_points; ++q)
                    {
                        traction += present_pressure_values[q] * normal_vectors[q]
                                             * fe_face_values.JxW(q);
                        surface_area += fe_face_values.JxW(q);
                    }

                }
    Assert(surface_area > 0., ExcLowerRangeType<double>(0., surface_area));

    return traction / surface_area;
}
}  // namespace TopographyProblem

// explicit instantiation
template dealii::Tensor<1,3> TopographyProblem::TopographySolver<3>::compute_boundary_traction() const;
