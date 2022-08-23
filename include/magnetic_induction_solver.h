/*
 * magnetic_induction_solver.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_MAGNETIC_INDUCTION_SOLVER_H_
#define INCLUDE_MAGNETIC_INDUCTION_SOLVER_H_

#include <angular_velocity.h>
#include <base.h>
#include <boundary_conditions.h>
#include <magnetic_induction_options.h>
#include <magnetic_induction_assembly_data.h>
#include <stabilization_flags.h>

#include <memory>
#include <optional>

namespace MagneticInduction {

using namespace BoundaryConditions;

/*!
 * @struct SolverParameters
 *
 * @brief A structure containing all the parameters of the Navier-Stokes
 * solver.
 */
struct SolverParameters: virtual Base::Parameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  SolverParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   *
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const SolverParameters &prm);

  /*!
   * @brief Stabilization parameter controlling the SUPG and PSPG terms.
   */
  double  c;

};



/*!
 * @brief Method forwarding parameters to a stream object.
 */
template <typename Stream>
Stream& operator<<(Stream &stream, const SolverParameters &prm);



template <int dim,
          typename TriangulationType = Triangulation<dim>>
class Solver: virtual public Base::Solver<dim, TriangulationType>
{
public:
  Solver(TriangulationType   &tria,
         Mapping<dim>        &mapping,
         const SolverParameters &parameters,
         const double         magnetic_reynolds_number = 1.0);

  void set_background_magnetic_field(const std::shared_ptr<const TensorFunction<1, dim>> &magnetic_field);

  void set_velocity_field(const std::shared_ptr<const TensorFunction<1, dim>> &velocity_field);

  VectorBoundaryConditions<dim>&  get_magnetic_field_bcs();
  const VectorBoundaryConditions<dim>&  get_magnetic_field_bcs() const;

  ScalarBoundaryConditions<dim>&  get_magnetic_pressure_bcs();
  const ScalarBoundaryConditions<dim>&  get_magnetic_pressure_bcs() const;

  double get_magnetic_reynolds_number() const;

protected:
  virtual void apply_boundary_conditions();

private:
  virtual void assemble_system(const bool use_homogenenous_constraints,
                               const bool use_newton_linearization);

  virtual void assemble_rhs(const bool use_homogenenous_constraints);

  virtual void setup_dofs();

  virtual void setup_fe_system();

  virtual void output_results(const unsigned int cycle = 0) const;

  void assemble_system_local_cell
  (const typename DoFHandler<dim>::active_cell_iterator  &cell,
   AssemblyData::Matrix::ScratchData<dim>                &scratch,
   MeshWorker::CopyData<1,1,1>                           &data,
   const bool                                             use_newton_linearization) const;

  void assemble_rhs_local_cell
  (const typename DoFHandler<dim>::active_cell_iterator  &cell,
   AssemblyData::RightHandSide::ScratchData<dim>         &scratch,
   MeshWorker::CopyData<0,1,1>                           &data) const;

protected:
  template <int n_matrices, int n_vectors, int n_dof_indices>
  void assemble_local_boundary
  (const typename DoFHandler<dim>::active_cell_iterator      &cell,
   const unsigned int                                         face_number,
   AssemblyData::ScratchData<dim>                            &scratch,
   MeshWorker::CopyData<n_matrices,n_vectors,n_dof_indices>  &data) const;

  VectorBoundaryConditions<dim> magnetic_field_boundary_conditions;

  ScalarBoundaryConditions<dim> magnetic_pressure_boundary_conditions;

  std::shared_ptr<const TensorFunction<1, dim>> background_magnetic_field_ptr;

  std::shared_ptr<const TensorFunction<1, dim>> velocity_field_ptr;

  const unsigned int  magnetic_fe_degree;

  const double        magnetic_reynolds_number;

  const double        c;

  unsigned int        magnetic_field_fe_index;

  unsigned int        magnetic_pressure_fe_index;

  unsigned int        magnetic_field_block_index;

  unsigned int        magnetic_pressure_block_index;

};


// inline functions
template <int dim, typename TriangulationType>
inline void Solver<dim, TriangulationType>::set_background_magnetic_field
(const std::shared_ptr<const TensorFunction<1, dim>> &magnetic_field)
{
  background_magnetic_field_ptr = magnetic_field;
}



template <int dim, typename TriangulationType>
inline void Solver<dim, TriangulationType>::set_velocity_field
(const std::shared_ptr<const TensorFunction<1, dim>> &velocity_field)
{
  velocity_field_ptr = velocity_field;
}



template <int dim, typename TriangulationType>
inline VectorBoundaryConditions<dim> &
Solver<dim, TriangulationType>::get_magnetic_field_bcs()
{
  return magnetic_field_boundary_conditions;
}



template <int dim, typename TriangulationType>
inline const VectorBoundaryConditions<dim> &
Solver<dim, TriangulationType>::get_magnetic_field_bcs() const
{
  return magnetic_field_boundary_conditions;
}


template <int dim, typename TriangulationType>
inline ScalarBoundaryConditions<dim> &
Solver<dim, TriangulationType>::get_magnetic_pressure_bcs()
{
  return magnetic_pressure_boundary_conditions;
}



template <int dim, typename TriangulationType>
inline const ScalarBoundaryConditions<dim> &
Solver<dim, TriangulationType>::get_magnetic_pressure_bcs() const
{
  return magnetic_pressure_boundary_conditions;
}



template <int dim, typename TriangulationType>
inline double Solver<dim, TriangulationType>::get_magnetic_reynolds_number() const
{
  return magnetic_reynolds_number;
}

}  // namespace MagneticInduction

#endif /* INCLUDE_MAGNETIC_INDUCTION_SOLVER_H_ */
