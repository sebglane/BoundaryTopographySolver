/*
 * solver_base.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_SOLVER_H_
#define INCLUDE_SOLVER_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/meshworker/copy_data.h>

#include <boundary_conditions.h>
#include <evaluation_base.h>
#include <parameters.h>

#include <memory>
#include <vector>



namespace LA
{
  using namespace dealii::LinearAlgebraDealII;
}



namespace Base {

using namespace dealii;

using namespace BoundaryConditions;

/**
 * @todo Add documentation.
 */
struct Parameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  Parameters();

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
   */
  template <typename Stream>
  friend Stream& operator<<(Stream &stream,
                            const Parameters &prm);

  /*!
   * @brief Parameters of the adaptive mesh refinement.
   */
  Utility::RefinementParameters refinement_parameters;

  /*!
   * @brief The spatial dimension of the problem.
   */
  unsigned int        space_dim;

  /*!
   * @brief The maximum number of Newton iterations.
   */
  unsigned int        n_iterations;

  /*!
   * @brief Absolute tolerance used for the convergence criterion Newton
   * iteration.
   */
  double              absolute_tolerance;

  /*!
   * @brief Relative tolerance used for the convergence criterion Newton
   * iteration.
   */
  double              relative_tolerance;

  /*!
   * @brief Number of Picard iterations performed.
   */
  unsigned int        n_picard_iterations;

  /*!
   * @brief Boolean flag to switch the Picard iteration on or off.
   */
  bool                apply_picard_iteration;

  /*!
   * @brief Boolean flag to enable verbose output on the terminal.
   */
  bool                 verbose;

  /*!
   * @brief Boolean flag to enable verbose output on the terminal.
   */
  bool                 print_timings;

  /*!
   * @brief Directory where the graphical output should be written.
   */
  std::string          graphical_output_directory;
};



/*!
 * @brief Method forwarding parameters to a stream object.
 */
template <typename Stream>
Stream& operator<<(Stream &stream, const Parameters &prm);



/*
 * @class Solver
 *
 * @todo Add documentation.
 *
 */
template <int dim,
          typename TriangulationType = Triangulation<dim>>
class Solver
{
public:
  using VectorType = LA::BlockVector;

  Solver(TriangulationType  &tria,
         Mapping<dim>       &mapping,
         const Parameters   &parameters);

  void add_postprocessor(const std::shared_ptr<EvaluationBase<dim, VectorType>> &postprocessor);

  void solve();

  ConditionalOStream& get_conditional_output_stream();

protected:
  virtual void apply_boundary_conditions() = 0;

  virtual void
  copy_local_to_global_system
  (const MeshWorker::CopyData<1,1,1> &data,
   const bool                         use_homogeneous_constraints);

  virtual void
  copy_local_to_global_rhs
  (const MeshWorker::CopyData<0,1,1> &data,
   const bool                         use_homogeneous_constraints);

  virtual void setup_dofs();

  void apply_dirichlet_constraints
  (const typename BoundaryConditionsBase<dim>::BCMapping &dirichlet_bcs,
   const ComponentMask                                   &mask);

  void apply_hanging_node_constraints();

  void apply_normal_flux_constraints
  (const typename BoundaryConditionsBase<dim>::BCMapping &normal_flux_bcs,
   const ComponentMask                                   &mask);

  void apply_periodicity_constraints
  (std::vector<PeriodicBoundaryData<dim>> &periodic_bcs,
   const ComponentMask                    &mask);

  void apply_mean_value_constraint
  (const ComponentMask &mask,
   const double         mean_value = 0);

  void setup_system_matrix(const Table<2, DoFTools::Coupling> &coupling_table);

  void setup_vectors();

private:
  virtual void assemble_system(const bool use_homogenenous_constraints = false,
                               const bool use_newton_linearization = true) = 0;

  virtual void assemble_rhs(const bool use_homogenenous_constraints = false) = 0;

  virtual void setup_fe_system() = 0;

  virtual void output_results(const unsigned int cycle = 0) const = 0;

  virtual void preprocess_newton_iteration(const unsigned int iteration,
                                           const bool         is_initial_cycle);

  virtual void postprocess_newton_iteration(const unsigned int iteration,
                                            const bool         is_initial_cycle);

  virtual void preprocess_picard_iteration(const unsigned int iteration);

  virtual void refine_mesh();

  void execute_mesh_refinement();

  std::vector<double> get_residual_components() const;

  void newton_iteration(const bool is_initial_step);

  void picard_iteration();

  void postprocess_solution(const unsigned int cycle = 0) const;

  void solve_linear_system(const bool initial_step);

protected:
  const std::string   graphical_output_directory;

  const bool          verbose;

  ConditionalOStream  pcout;

  TriangulationType  &triangulation;
  Mapping<dim>       &mapping;

  std::shared_ptr<FESystem<dim>> fe_system;

  DoFHandler<dim>             dof_handler;

  // constraints
  AffineConstraints<double>   nonzero_constraints;
  AffineConstraints<double>   zero_constraints;
  std::map<unsigned int, double>  component_mean_values;

  // linear algebra
  BlockSparsityPattern            sparsity_pattern;
  LA::BlockSparseMatrix  system_matrix;

  VectorType  system_rhs;

  VectorType  present_solution;
  VectorType  evaluation_point;
  VectorType  solution_update;

  // monitor of computing times
  TimerOutput                 computing_timer;

private:
  std::vector<std::shared_ptr<EvaluationBase<dim, VectorType>>> postprocessor_ptrs;

  const Utility::RefinementParameters refinement_parameters;

  const unsigned int  n_maximum_iterations;

  const unsigned int  n_picard_iterations;

  const double        absolute_tolerance;

  const double        relative_tolerance;

  const bool          print_timings;

  const bool          apply_picard_iteration;
};

// inline methods
template <int dim, typename TriangulationType>
inline ConditionalOStream&
Solver<dim, TriangulationType>::get_conditional_output_stream()
{
  return (pcout);
}



template <int dim, typename TriangulationType>
inline void Solver<dim, TriangulationType>::add_postprocessor
(const std::shared_ptr<EvaluationBase<dim, VectorType>> &postprocessor)
{
  postprocessor_ptrs.push_back(postprocessor);
}



template <int dim, typename TriangulationType>
inline void Solver<dim, TriangulationType>::
preprocess_newton_iteration(const unsigned int, const bool)
{
  return;
}



template <int dim, typename TriangulationType>
inline void Solver<dim, TriangulationType>::
postprocess_newton_iteration(const unsigned int, const bool)
{
  return;
}



template <int dim, typename TriangulationType>
inline void Solver<dim, TriangulationType>::
preprocess_picard_iteration(const unsigned int)
{
  return;
}

template <int dim, typename TriangulationType>
inline std::vector<double>
Solver<dim, TriangulationType>::get_residual_components() const
{
  std::vector<double> residual;

  for (unsigned int i=0; i<system_rhs.n_blocks(); ++i)
    residual.push_back(system_rhs.block(i).l2_norm());

  return (residual);
}

}  // namespace Base

#endif /* INCLUDE_SOLVER_H_ */
