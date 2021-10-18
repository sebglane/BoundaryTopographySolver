/*
 * evaluation_boundary_traction.cc
 *
 *  Created on: Sep 27, 2021
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_values.h>

#include <evaluation_boundary_traction.h>

namespace Hydrodynamic {

using namespace dealii;

template <int dim>
EvaluationBoundaryTraction<dim>::EvaluationBoundaryTraction
(const unsigned int velocity_start_index,
 const unsigned int pressure_index,
 const double reynolds_number)
:
boundary_id(numbers::invalid_boundary_id),
velocity_start_index(velocity_start_index),
pressure_index(pressure_index),
reynolds_number(reynolds_number)
{
  traction_table.declare_column("cycle");
  traction_table.declare_column("x-direction");
  traction_table.declare_column("y-direction");
  traction_table.set_scientific("x-direction", true);
  traction_table.set_scientific("y-direction", true);
  traction_table.add_column_to_supercolumn("x-direction", "average traction");
  traction_table.add_column_to_supercolumn("y-direction", "average traction");

  pressure_table.declare_column("cycle");
  pressure_table.declare_column("x-direction");
  pressure_table.declare_column("y-direction");
  pressure_table.set_scientific("x-direction", true);
  pressure_table.set_scientific("y-direction", true);
  pressure_table.add_column_to_supercolumn("x-direction", "average pressure component");
  pressure_table.add_column_to_supercolumn("y-direction", "average pressure component");

  viscous_table.declare_column("cycle");
  viscous_table.declare_column("x-direction");
  viscous_table.declare_column("y-direction");
  viscous_table.set_scientific("x-direction", true);
  viscous_table.set_scientific("y-direction", true);
  viscous_table.add_column_to_supercolumn("x-direction", "average viscous component");
  viscous_table.add_column_to_supercolumn("y-direction", "average viscous component");

  if (dim==3)
  {
    traction_table.declare_column("z-direction");
    traction_table.set_scientific("z-direction", true);
    traction_table.add_column_to_supercolumn("z-direction", "average traction");

    pressure_table.declare_column("z-direction");
    pressure_table.set_scientific("z-direction", true);
    pressure_table.add_column_to_supercolumn("z-direction", "average pressure component");


    viscous_table.declare_column("z-direction");
    viscous_table.set_scientific("z-direction", true);
    viscous_table.add_column_to_supercolumn("z-direction", "average viscous component");
  }
}



template <int dim>
EvaluationBoundaryTraction<dim>::~EvaluationBoundaryTraction()
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << std::endl;
    traction_table.write_text(std::cout);

    std::cout << std::endl;
    pressure_table.write_text(std::cout);

    std::cout << std::endl;
    viscous_table.write_text(std::cout);
  }
}



template <int dim>
void EvaluationBoundaryTraction<dim>::operator()
(const Mapping<dim>        &mapping,
 const FiniteElement<dim>  &fe,
 const DoFHandler<dim>     &dof_handler,
 const Vector<double>      &solution)
{
  evaluate(mapping, fe, dof_handler, solution);
}



template <int dim>
void EvaluationBoundaryTraction<dim>::operator()
(const Mapping<dim>        &mapping,
 const FiniteElement<dim>  &fe,
 const DoFHandler<dim>     &dof_handler,
 const BlockVector<double> &solution)
{
  evaluate(mapping, fe, dof_handler, solution);
}



template <int dim>
template <typename VectorType>
void EvaluationBoundaryTraction<dim>::evaluate
(const Mapping<dim>        &mapping,
 const FiniteElement<dim>  &fe,
 const DoFHandler<dim>     &dof_handler,
 const VectorType          &solution)
{
  AssertThrow(boundary_id != numbers::invalid_boundary_id,
              ExcMessage("Boundary id was not specified."));

  Tensor<1, dim>  traction;
  Tensor<1, dim>  pressure_component;
  Tensor<1, dim>  viscous_component;
  double          area = 0;

  QGauss<dim-1>   face_quadrature(fe.degree + 1);

  FEFaceValues<dim> fe_face_values(mapping,
                                   fe,
                                   face_quadrature,
                                   update_values|
                                   update_gradients|
                                   update_normal_vectors|
                                   update_JxW_values);

  const FEValuesExtractors::Vector  velocity(velocity_start_index);
  const FEValuesExtractors::Scalar  pressure(pressure_index);

  const unsigned int n_face_q_points{face_quadrature.size()};
  std::vector<Tensor<1, dim>> face_normal_vectors(n_face_q_points);

  std::vector<SymmetricTensor<2, dim>>  present_velocity_gradients(n_face_q_points);
  std::vector<double>                   present_pressure_values(n_face_q_points);


  for (const auto cell: dof_handler.active_cell_iterators())
    if (cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() &&
            face->boundary_id() == boundary_id)
        {
          fe_face_values.reinit(cell, face);

          fe_face_values[velocity].get_function_symmetric_gradients(solution,
                                                                    present_velocity_gradients);

          fe_face_values[pressure].get_function_values(solution,
                                                       present_pressure_values);

          face_normal_vectors = fe_face_values.get_normal_vectors();

          // Loop over face quadrature points
          for (const auto q: fe_face_values.quadrature_point_indices())
          {
            const double JxW_face{fe_face_values.JxW(q)};

            traction += (- present_pressure_values[q] * face_normal_vectors[q] +
                         2.0 / reynolds_number * present_velocity_gradients[q] * face_normal_vectors[q]) *
                        JxW_face;
            pressure_component += - present_pressure_values[q] * face_normal_vectors[q] *
                                  JxW_face;
            viscous_component += 2.0 / reynolds_number * present_velocity_gradients[q] * face_normal_vectors[q] *
                                 JxW_face;

            area += JxW_face;

          } // Loop over face quadrature points

        }

  Assert(area > 0.0, ExcLowerRangeType<double>(0.0, area));

  traction /= area;
  pressure_component /= area;
  viscous_component /= area;

  traction_table.add_value("cycle", this->cycle);
  traction_table.add_value("x-direction", traction[0]);
  traction_table.add_value("y-direction", traction[1]);

  pressure_table.add_value("cycle", this->cycle);
  pressure_table.add_value("x-direction", pressure_component[0]);
  pressure_table.add_value("y-direction", pressure_component[1]);

  viscous_table.add_value("cycle", this->cycle);
  viscous_table.add_value("x-direction", viscous_component[0]);
  viscous_table.add_value("y-direction", viscous_component[1]);

  if constexpr(dim==3)
  {
    traction_table.add_value("z-direction", traction[2]);
    pressure_table.add_value("z-direction", pressure_component[2]);
    viscous_table.add_value("z-direction", viscous_component[2]);
  }
}

// explicit instantiations
template class EvaluationBoundaryTraction<2>;
template class EvaluationBoundaryTraction<3>;

}  // namespace Hydrodynamic
