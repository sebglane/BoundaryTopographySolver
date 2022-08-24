/*
 * assemble_system_local.cc
 *
 *  Created on: Aug 23, 2022
 *      Author: sg
 */


#include <assembly_functions.h>
#include <magnetic_induction_solver.h>

namespace MagneticInduction {

template <int dim, typename TriangulationType>
void Solver<dim, TriangulationType>::
assemble_system_local_cell
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 AssemblyData::Matrix::ScratchData<dim>                &scratch,
 MeshWorker::CopyData<1,1,1>                           &data,
 const bool                                             /* use_newton_linearization */) const
{
  data.matrices[0] = 0;
  data.vectors[0] = 0;
  cell->get_dof_indices(data.local_dof_indices[0]);

  const auto &fe_values = scratch.reinit(cell);
  scratch.extract_local_dof_values("evaluation_point",
                                   this->evaluation_point);
  const auto &JxW = scratch.get_JxW_values();

  const FEValuesExtractors::Vector  magnetic_field(magnetic_field_fe_index);
  const FEValuesExtractors::Scalar  magnetic_pressure(magnetic_pressure_fe_index);

  // viscosity
  const double lambda{1.0 / magnetic_reynolds_number};

  // stabilization parameters
  const double tau{lambda * std::pow(cell->diameter() / c, 2)};
  const double upsilon{std::pow(c, 2) / lambda};

  // solution values
  scratch.present_magnetic_field_values = scratch.get_values("evaluation_point",
                                                             magnetic_field);
  scratch.present_magnetic_field_curls = scratch.get_curls("evaluation_point",
                                                           magnetic_field);
  scratch.present_magnetic_field_divergences = scratch.get_divergences("evaluation_point",
                                                                       magnetic_field);
  scratch.present_magnetic_pressure_gradients = scratch.get_gradients("evaluation_point",
                                                                      magnetic_pressure);

  // assign vector options
  scratch.assign_vector_options(velocity_field_ptr,
                                background_magnetic_field_ptr,
                                source_term_ptr);
  scratch.adjust_magnetic_field_local_cell();

  for (const auto q: fe_values.quadrature_point_indices())
  {
    for (const auto i: fe_values.dof_indices())
    {
      scratch.phi_magnetic_field[i] = fe_values[magnetic_field].value(i, q);
      scratch.curl_phi_magnetic_field[i] = fe_values[magnetic_field].curl(i, q);
      scratch.div_phi_magnetic_field[i] = fe_values[magnetic_field].divergence(i, q);
      scratch.grad_phi_magnetic_pressure[i] = fe_values[magnetic_pressure].gradient(i, q);
    }

    for (const auto i: fe_values.dof_indices())
    {
      for (const auto j: fe_values.dof_indices())
      {
        const double matrix{compute_matrix(scratch,
                                           i,
                                           j,
                                           q,
                                           lambda,
                                           tau,
                                           upsilon)};

        data.matrices[0](i, j) +=  matrix * JxW[q];
      }

      const double rhs{compute_rhs(scratch,
                                   i,
                                   q,
                                   lambda,
                                   tau,
                                   upsilon)};

      data.vectors[0](i) += rhs * JxW[q];
    }
  } // end loop over cell quadrature points

}



// explicit instantiation
template
void
Solver<2>::
assemble_system_local_cell
(const typename DoFHandler<2>::active_cell_iterator &,
 AssemblyData::Matrix::ScratchData<2>               &,
 MeshWorker::CopyData<1,1,1>                        &,
 const bool                                           ) const;
template
void
Solver<3>::
assemble_system_local_cell
(const typename DoFHandler<3>::active_cell_iterator &,
 AssemblyData::Matrix::ScratchData<3>               &,
 MeshWorker::CopyData<1,1,1>                        &,
 const bool                                           ) const;


}  // namespace MagneticInduction



