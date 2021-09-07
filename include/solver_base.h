/*
 * solver_base.h
 *
 *  Created on: Aug 30, 2021
 *      Author: sg
 */

#ifndef INCLUDE_SOLVER_H_
#define INCLUDE_SOLVER_H_

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>

#include <boundary_conditions.h>
#include <evaluation_base.h>
#include <parameters.h>

#include <memory>
#include <vector>

namespace SolverBase {

using namespace dealii;

using namespace BoundaryConditions;

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
 */
template <int dim>
class Solver
{
public:
  Solver(Triangulation<dim> &tria,
         Mapping<dim>       &mapping,
         const Parameters &parameters);

  void set_postprocessor(EvaluationBase<dim> &postprocessor);

  void solve();

protected:
  virtual void apply_boundary_conditions() = 0;

  void apply_dirichlet_constraints
  (const typename BoundaryConditionsBase<dim>::BCMapping &dirichlet_bcs,
   const ComponentMask                                   &mask);

  void apply_hanging_node_constraints();

  void apply_normal_flux_constraints
  (const typename BoundaryConditionsBase<dim>::BCMapping &normal_flux_bcs,
   const ComponentMask                                   &mask);

  void apply_periodicity_constraints
  (std::vector<PeriodicBoundaryData<dim>> &periodic_bcs);

  virtual void assemble_system(const bool use_homogenenous_constraints = false) = 0;

  virtual void assemble_rhs(const bool use_homogenenous_constraints = false) = 0;

  virtual void setup_dofs();

  virtual void setup_fe_system() = 0;

  void setup_system_matrix
  (const std::vector<types::global_dof_index> &dofs_per_block,
   const Table<2, DoFTools::Coupling>         &coupling_table);

  void setup_vectors(const std::vector<types::global_dof_index> &dofs_per_block);

  virtual void preprocess_newton_iteration();

  virtual void output_results(const unsigned int cycle = 0) const = 0;

  Triangulation<dim>         &triangulation;
  Mapping<dim>               &mapping;

  FESystem<dim>              *fe_system;

  DoFHandler<dim>             dof_handler;

  // constraints
  AffineConstraints<double>   nonzero_constraints;
  AffineConstraints<double>   zero_constraints;

  // system matrix
  BlockSparsityPattern        sparsity_pattern;
  BlockSparseMatrix<double>   system_matrix;

  // vectors
  BlockVector<double>         evaluation_point;
  BlockVector<double>         present_solution;
  BlockVector<double>         solution_update;
  BlockVector<double>         system_rhs;

  // monitor of computing times
  TimerOutput                 computing_timer;

private:
  void newton_iteration(const bool is_initial_step);

  void postprocess_solution(const unsigned int cycle = 0) const;

  virtual void refine_mesh();

  void solve_linear_system(const bool initial_step);

  EvaluationBase<dim> *postprocessor_ptr;

  const Utility::RefinementParameters refinement_parameters;

  const unsigned int  n_maximum_iterations;

  const double        absolute_tolerance;

  const double        relative_tolerance;

  const bool          print_timings;

protected:
  const std::string   graphical_output_directory;

  const bool verbose;
};

// inline methods
template <int dim>
inline void Solver<dim>::set_postprocessor(EvaluationBase<dim> &postprocessor)
{
  postprocessor_ptr = &postprocessor;
}



template <int dim>
inline void Solver<dim>::preprocess_newton_iteration()
{
  return;
}

}  // namespace SolverBase

#endif /* INCLUDE_SOLVER_H_ */
