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



namespace LegacyAssemblyData {

namespace Matrix {

template <int dim>
struct Scratch : Hydrodynamic::LegacyAssemblyData::Matrix::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags,
          const Quadrature<dim-1>  &face_quadrature_formula,
          const UpdateFlags         face_update_flags,
          const StabilizationFlags  stabilization_flags,
          const bool                use_stress_form = false,
          const bool                allocate_background_velocity = false,
          const bool                allocate_body_force = false,
          const bool                allocate_face_stresses = false,
          const bool                allocate_traction = false,
          const bool                allocate_gravity_field = false,
          const bool                allocate_reference_density = false,
          const bool                allocate_density_bc = false);

  Scratch(const Scratch<dim>  &data);

  OptionalVectorArguments<dim>  strong_form_options;

  OptionalScalarArguments<dim>  weak_form_options;

  Advection::OptionalVectorArguments<dim> density_strong_form_options;
  Advection::OptionalScalarArguments<dim> density_weak_form_options;

  // shape functions
  std::vector<double>         phi_density;
  std::vector<Tensor<1, dim>> grad_phi_density;

  // solution values
  std::vector<double>         present_density_values;
  std::vector<Tensor<1, dim>> present_density_gradients;

  // stabilization related quantities
  std::vector<double>         present_strong_density_residuals;

  // solution face values
  std::vector<double>         present_density_face_values;
  std::vector<Tensor<1, dim>> present_velocity_face_values;

  // source term face values
  std::vector<double>         density_boundary_values;

};

} // namespace Matrix

namespace RightHandSide
{

template <int dim>
struct Scratch : Hydrodynamic::LegacyAssemblyData::RightHandSide::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags,
          const Quadrature<dim-1>  &face_quadrature_formula,
          const UpdateFlags         face_update_flags,
          const StabilizationFlags  stabilization_flags,
          const bool                use_stress_form = false,
          const bool                allocate_background_velocity = false,
          const bool                allocate_body_force = false,
          const bool                allocate_face_stresses = false,
          const bool                allocate_traction = false,
          const bool                allocate_gravity_field = false,
          const bool                allocate_reference_density = false,
          const bool                allocate_density_bc = false);

  Scratch(const Scratch<dim>  &data);

  OptionalVectorArguments<dim>  strong_form_options;

  OptionalScalarArguments<dim>  weak_form_options;

  Advection::OptionalVectorArguments<dim> density_strong_form_options;
  Advection::OptionalScalarArguments<dim> density_weak_form_options;

  // shape functions
  std::vector<double>         phi_density;
  std::vector<Tensor<1, dim>> grad_phi_density;

  // solution values
  std::vector<double>         present_density_values;
  std::vector<Tensor<1, dim>> present_density_gradients;

  // stabilization related quantities
  std::vector<double>         present_strong_density_residuals;

  // solution face values
  std::vector<double>         present_density_face_values;
  std::vector<Tensor<1, dim>> present_velocity_face_values;

  // source term face values
  std::vector<double> density_boundary_values;

};

} // namespace RightHandSide

} // namespace LegacyAssemblyData

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
