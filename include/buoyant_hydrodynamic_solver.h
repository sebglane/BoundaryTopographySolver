/*
 * buoyant_hydrodynamic_solver.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_BUOYANT_HYDRODYNAMIC_SOLVER_H_
#define INCLUDE_BUOYANT_HYDRODYNAMIC_SOLVER_H_

#include <hydrodynamic_solver.h>

namespace BuoyantHydrodynamic {

using namespace BoundaryConditions;

/*!
 * @struct SolverParameters
 *
 * @brief A structure containing all the parameters of the Navier-Stokes
 * solver.
 */
struct SolverParameters: Hydrodynamic::SolverParameters
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
   * @brief Stabilization parameter controlling the SUPG term of the density equation.
   */
  double  c_density;

  /*!
   * @brief Minimal viscosity to stabilize the density equation in case of a
   * vanishing velocity.
   */
  double  nu_density;

};



/*!
 * @brief Method forwarding parameters to a stream object.
 */
template <typename Stream>
Stream& operator<<(Stream &stream, const SolverParameters &prm);



namespace AssemblyData {

namespace Matrix {

template <int dim>
struct Scratch : Hydrodynamic::AssemblyData::Matrix::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags,
          const Quadrature<dim-1>  &face_quadrature_formula,
          const UpdateFlags         face_update_flags,
          const StabilizationFlags  stabilization_flags,
          const bool                allocate_body_force,
          const bool                allocate_traction,
          const bool                allocate_density_bc);

  Scratch(const Scratch<dim>  &data);

  // shape functions
  std::vector<double>         phi_density;
  std::vector<Tensor<1, dim>> grad_phi_density;

  // solution values
  std::vector<double>         present_density_values;
  std::vector<Tensor<1, dim>> present_density_gradients;

  // source term values
  std::vector<Tensor<1,dim>>  reference_density_gradients;
  std::vector<Tensor<1,dim>>  gravity_field_values;

  // solution face values
  std::vector<double>         present_density_face_values;
  std::vector<Tensor<1, dim>> present_velocity_face_values;
  std::vector<Tensor<1, dim>> face_normal_vectors;

  // source term face values
  std::vector<double>         density_boundary_values;

};

} // namespace Matrix

namespace RightHandSide
{

template <int dim>
struct Scratch : Hydrodynamic::AssemblyData::RightHandSide::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags,
          const Quadrature<dim-1>  &face_quadrature_formula,
          const UpdateFlags         face_update_flags,
          const StabilizationFlags  stabilization_flags,
          const bool                allocate_body_force,
          const bool                allocate_traction,
          const bool                allocate_density_bc);

  Scratch(const Scratch<dim>  &data);

  // shape functions
  std::vector<double>         phi_density;
  std::vector<Tensor<1, dim>> grad_phi_density;

  // solution values
  std::vector<double>         present_density_values;
  std::vector<Tensor<1, dim>> present_density_gradients;

  // source term values
  std::vector<Tensor<1,dim>>  reference_density_gradients;
  std::vector<Tensor<1,dim>>  gravity_field_values;

  // solution face values
  std::vector<double>         present_density_face_values;
  std::vector<Tensor<1, dim>> present_velocity_face_values;
  std::vector<Tensor<1, dim>> face_normal_vectors;

  // source term face values
  std::vector<double>         density_boundary_values;

};

} // namespace RightHandSide

} // namespace AssemblyData

template <int dim>
class Solver: public Hydrodynamic::Solver<dim>
{

public:
  Solver(Triangulation<dim>  &tria,
         Mapping<dim>        &mapping,
         const SolverParameters &parameters,
         const double         reynolds_number = 1.0,
         const double         froude_number = 0.0,
         const double         stratification_number = 1.0,
         const double         rossby_number = 0.0);

  void set_reference_density(const Function<dim> &reference_density);

  void set_gravity_field(const TensorFunction<1, dim> &gravity_field);

  ScalarBoundaryConditions<dim>&  get_density_bcs();

  const ScalarBoundaryConditions<dim>&  get_density_bcs() const;

  double get_stratification_number() const;

private:
  virtual void setup_fe_system();

  virtual void setup_dofs();

  virtual void apply_boundary_conditions();

  virtual void assemble_system(const bool initial_step);

  virtual void assemble_rhs(const bool initial_step);

  void assemble_local_system
  (const typename DoFHandler<dim>::active_cell_iterator &cell,
   AssemblyData::Matrix::Scratch<dim> &scratch,
   AssemblyBaseData::Matrix::Copy     &data) const;

  void assemble_local_rhs
  (const typename DoFHandler<dim>::active_cell_iterator &cell,
   AssemblyData::RightHandSide::Scratch<dim> &scratch,
   AssemblyBaseData::RightHandSide::Copy     &data) const;

  virtual void output_results(const unsigned int cycle = 0) const;

  virtual void preprocess_newton_iteration(const unsigned int iteration,
                                           const bool         is_initial_cycle);

  ScalarBoundaryConditions<dim> density_boundary_conditions;

  const Function<dim>          *reference_density_ptr;

  const TensorFunction<1, dim> *gravity_field_ptr;

  const double        stratification_number;

  const unsigned int  density_fe_degree;

  const double        c_density;

  const double        nu_density;

};

// inline functions
template <int dim>
inline const ScalarBoundaryConditions<dim> &
Solver<dim>::get_density_bcs() const
{
  return density_boundary_conditions;
}



template <int dim>
inline ScalarBoundaryConditions<dim> &
Solver<dim>::get_density_bcs()
{
  return density_boundary_conditions;
}



template <int dim>
inline void Solver<dim>::set_reference_density(const Function<dim> &reference_density)
{
  reference_density_ptr = &reference_density;
  return;
}



template <int dim>
inline void Solver<dim>::set_gravity_field(const TensorFunction<1, dim> &gravity_field)
{
  gravity_field_ptr = &gravity_field;
  return;
}



template <int dim>
inline double Solver<dim>::get_stratification_number() const
{
  return (stratification_number);
}

}  // namespace TopographyProblem



#endif /* INCLUDE_BUOYANT_HYDRODYNAMIC_SOLVER_H_ */
