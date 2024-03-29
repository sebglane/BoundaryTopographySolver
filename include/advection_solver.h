/*
 * advection_solver.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_ADVECTION_SOLVER_H_
#define INCLUDE_ADVECTION_SOLVER_H_

#include <deal.II/meshworker/copy_data.h>

#include <advection_assembly_data.h>
#include <base.h>

namespace Advection {

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
   * @brief Stabilization parameter controlling the SUPG term.
   */
  double  c;

  /*!
   * @brief Minimal viscosity to stabilize the advection equation in case of a
   * vanishing advection field.
   */
  double  nu;

};



/*!
 * @brief Method forwarding parameters to a stream object.
 */
template <typename Stream>
Stream& operator<<(Stream &stream, const SolverParameters &prm);



/**
 * @todo Add documentation.
 */
template <int dim,
          typename TriangulationType = Triangulation<dim>>
class Solver: virtual public Base::Solver<dim, TriangulationType>
{

public:
  Solver(TriangulationType       &tria,
         Mapping<dim>            &mapping,
         const SolverParameters  &parameters,
         const double             gradient_scaling_number = 0.0);

  void set_advection_field(const std::shared_ptr<const TensorFunction<1, dim>> &advection_field);

  void set_reference_field(const std::shared_ptr<const Function<dim>> &reference_field);

  void set_source_term(const std::shared_ptr<const Function<dim>> &reference_density);

  double get_gradient_scaling_number() const;

  ScalarBoundaryConditions<dim>&  get_bcs();

  const ScalarBoundaryConditions<dim>&  get_bcs() const;

protected:
  virtual void apply_boundary_conditions();

private:
  virtual void setup_fe_system();

  virtual void setup_dofs();

  virtual void assemble_system(const bool use_homogenenous_constraints,
                               const bool use_newton_linearization);

  virtual void assemble_rhs(const bool use_homogenenous_constraints);

  virtual void output_results(const unsigned int cycle = 0) const;

  void assemble_system_local_cell
  (const typename DoFHandler<dim>::active_cell_iterator  &cell,
   AssemblyData::Matrix::ScratchData<dim>                &scratch,
   MeshWorker::CopyData<1,1,1>                           &data) const;

  void assemble_rhs_local_cell
  (const typename DoFHandler<dim>::active_cell_iterator  &cell,
   AssemblyData::RightHandSide::ScratchData<dim>         &scratch,
   MeshWorker::CopyData<0,1,1>                           &data) const;

  template <int n_matrices, int n_vectors, int n_dof_indices>
  void assemble_local_boundary
  (const typename DoFHandler<dim>::active_cell_iterator      &cell,
   const unsigned int                                         face_number,
   AssemblyData::RightHandSide::ScratchData<dim>             &scratch,
   MeshWorker::CopyData<n_matrices,n_vectors,n_dof_indices>  &data) const;

protected:
  ScalarBoundaryConditions<dim>                 scalar_boundary_conditions;

  std::shared_ptr<const TensorFunction<1, dim>> advection_field_ptr;

  std::shared_ptr<const Function<dim>>          reference_field_ptr;

  std::shared_ptr<const Function<dim>>          source_term_ptr;

  const double        gradient_scaling_number;

  const unsigned int  scalar_fe_degree;

  const double        c;

  const double        nu;

  unsigned int        scalar_fe_index;

  unsigned int        scalar_block_index;

};

// inline functions
template <int dim, typename TriangulationType>
inline double Solver<dim, TriangulationType>::get_gradient_scaling_number() const
{
  return (gradient_scaling_number);
}



template <int dim, typename TriangulationType>
inline const ScalarBoundaryConditions<dim> &
Solver<dim, TriangulationType>::get_bcs() const
{
  return scalar_boundary_conditions;
}



template <int dim, typename TriangulationType>
inline ScalarBoundaryConditions<dim> &
Solver<dim, TriangulationType>::get_bcs()
{
  return scalar_boundary_conditions;
}



template <int dim, typename TriangulationType>
inline void Solver<dim, TriangulationType>::set_advection_field
(const std::shared_ptr<const TensorFunction<1, dim>> &advection_field)
{
  advection_field_ptr = advection_field;
  return;
}



template <int dim, typename TriangulationType>
inline void Solver<dim, TriangulationType>::set_reference_field
(const std::shared_ptr<const Function<dim>> &reference_field)
{
  reference_field_ptr = reference_field;
  return;
}



template <int dim, typename TriangulationType>
inline void Solver<dim, TriangulationType>::set_source_term
(const std::shared_ptr<const Function<dim>> &source_term)
{
  source_term_ptr = source_term;
  return;
}

}  // namespace Advection



#endif /* INCLUDE_ADVECTION_SOLVER_H_ */
