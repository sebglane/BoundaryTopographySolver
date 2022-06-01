/*
 * buoyant_hydrodynamic_solver.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_BUOYANT_HYDRODYNAMIC_SOLVER_H_
#define INCLUDE_BUOYANT_HYDRODYNAMIC_SOLVER_H_

#include <advection_solver.h>
#include <buoyant_hydrodynamic_assembly_data.h>
#include <buoyant_hydrodynamic_options.h>
#include <hydrodynamic_solver.h>

#include <memory>

namespace BuoyantHydrodynamic {

using namespace BoundaryConditions;

/*!
 * @struct SolverParameters
 *
 * @brief A structure containing all the parameters of the Navier-Stokes
 * solver.
 */
struct SolverParameters: Hydrodynamic::SolverParameters, Advection::SolverParameters
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

};



/*!
 * @brief Method forwarding parameters to a stream object.
 */
template <typename Stream>
Stream& operator<<(Stream &stream, const SolverParameters &prm);



template <int dim,
          typename TriangulationType = Triangulation<dim>>
class Solver: public Hydrodynamic::Solver<dim, TriangulationType>,
              public Advection::Solver<dim, TriangulationType>
{

public:
  Solver(TriangulationType       &tria,
         Mapping<dim>            &mapping,
         const SolverParameters  &parameters,
         const double             reynolds_number = 1.0,
         const double             froude_number = 0.0,
         const double             stratification_number = 1.0,
         const double             rossby_number = 0.0);

  void set_gravity_field(const std::shared_ptr<const TensorFunction<1, dim>> &gravity_field);

private:
  virtual void setup_fe_system();

  virtual void setup_dofs();

  virtual void apply_boundary_conditions();

  virtual void assemble_system(const bool use_homogenenous_constraints,
                               const bool use_newton_linearization);

  virtual void assemble_rhs(const bool use_homogenenous_constraints);

  void assemble_system_local_cell
  (const typename DoFHandler<dim>::active_cell_iterator  &cell,
   AssemblyData::Matrix::ScratchData<dim>                &scratch,
   MeshWorker::CopyData<1,1,1>                           &data,
   const bool                                             use_newton_linearization,
   const bool                                             use_stress_tensor) const;

  void assemble_system_local_boundary
  (const typename DoFHandler<dim>::active_cell_iterator  &cell,
   const unsigned int                                     face_number,
   AssemblyData::Matrix::ScratchData<dim>                &scratch,
   MeshWorker::CopyData<1,1,1>                           &data,
   const bool                                             use_newton_linearization,
   const bool                                             use_stress_tensor) const;

  void assemble_rhs_local_cell
  (const typename DoFHandler<dim>::active_cell_iterator  &cell,
   AssemblyData::RightHandSide::ScratchData<dim>         &scratch,
   MeshWorker::CopyData<0,1,1>                           &data,
   const bool                                             use_stress_form) const;

  void assemble_rhs_local_boundary
  (const typename DoFHandler<dim>::active_cell_iterator  &cell,
   const unsigned int                                     face_number,
   AssemblyData::RightHandSide::ScratchData<dim>         &scratch,
   MeshWorker::CopyData<0,1,1>                           &data,
   const bool                                             use_stress_form) const;

  virtual void output_results(const unsigned int cycle = 0) const;

  virtual void postprocess_newton_iteration(const unsigned int iteration,
                                            const bool         is_initial_cycle);

  std::shared_ptr<const TensorFunction<1, dim>> gravity_field_ptr;

};

// inline functions
template <int dim, typename TriangulationType>
inline void Solver<dim, TriangulationType>::set_gravity_field
(const std::shared_ptr<const TensorFunction<1, dim>> &gravity_field)
{
  gravity_field_ptr = gravity_field;
  return;
}

}  // namespace TopographyProblem



#endif /* INCLUDE_BUOYANT_HYDRODYNAMIC_SOLVER_H_ */
