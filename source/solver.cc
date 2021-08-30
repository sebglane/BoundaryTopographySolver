/*
 * solver.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <solver.h>
#include <postprocessor.h>

#include <map>

namespace TopographyProblem {

template<int dim>
TopographySolver<dim>::TopographySolver(Parameters &parameters_)
:
parameters(parameters_),
// coefficients
equation_coefficients{parameters.S,
                      1. / parameters.Rossby,
                      1. / (parameters.Froude * parameters.Froude),
                      parameters.Alfven * parameters.Alfven,
                      parameters.magReynolds},
// model parameters
rotation_vector((dim == 3? Point<dim>::unit_vector(1): Point<dim>::unit_vector(dim-1))),
gravity_vector((dim == 3? -Point<dim>::unit_vector(1): Point<dim>::unit_vector(dim-1))),
background_velocity_value(Point<dim>::unit_vector(0)),
background_field_value((dim == 3? -Point<dim>::unit_vector(1): Point<dim>::unit_vector(dim-1))),
background_density_gradient((dim == 3? -Point<dim>::unit_vector(1): Point<dim>::unit_vector(dim-1))),
background_velocity_gradient(),
background_field_curl(),
background_field_gradient(),
// triangulation
triangulation(Triangulation<dim>::maximum_smoothing),
// finite element part
fe_system(FE_Q<dim>(parameters.density_degree), 1,
          FESystem<dim>(FE_Q<dim>(parameters.velocity_degree), dim), 1,
          FE_Q<dim>(parameters.velocity_degree - 1), 1,
          FESystem<dim>(FE_Q<dim>(parameters.magnetic_degree), dim), 1,
          FE_Q<dim>(parameters.magnetic_degree), 1),
dof_handler(triangulation),
// monitor
computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
{
    std::cout << "Topography solver by S. Glane\n"
            << "This program solves inviscid flow over topography in a stratified layer.\n"
            << "The governing equations are\n\n"
            << "\t-- Continuity equation:\n\t\t div(rho V) = -S v . grad(rho_0),\n\n"
            << "\t-- Incompressibility constraint:\n\t\t div(v) = 0,\n\n";
    if (parameters.include_rotation)
        std::cout << "\t-- Navier-Stokes equation:\n\t\t V . grad(v) + v . grad(V) + v . grad(v) + 2 / Ro  Omega x v\n";
    else
        std::cout << "\t-- Navier-Stokes equation:\n\t\t V . grad(v) + v . grad(V) + v . grad(v)\n";
    std::cout << "\t\t\t\t= - grad(p) + (1 / Fr^2) rho g + \n"
              << "\t\t\t\t  + Al^2 (B . grad(b) + b . grad(B) + b . grad(b)),\n\n"
              << "\t-- Induction equation:\n\t\t div(grad(b)) + Rm (curl(v x B) + curl(V x b) + curl(v x b)) = 0\n\n"
              << "The stratification parameter, S, the Rossby number, Ro, the Froude number, Fr, the Alfven number, Al,\n"
              << "and the magnetic Reynolds number, Rm, are given by:\n\n";

    // generate a nice table of the equation coefficients
    std::cout << "+-----------+---------------+---------------+---------------+---------------+\n"
            << "|    S      |      Ro       |      Fr       |      Al       |      Rm       |\n"
            << "+-----------+---------------+---------------+---------------+---------------+\n"
            << "| N^2 l / g | V / (Omega l) | V / sqrt(g l) |     Va / v    |   v l / eta   |\n"
            << "+-----------+---------------+---------------+---------------+---------------+\n";

    std::cout << std::endl << "You have chosen the following parameter set:";

    std::stringstream ss;
    if (parameters.include_rotation)
        ss << "+----------+----------+----------+----------+----------+----------+----------+\n"
           << "|    k     |    h     |    S     |    Ro    |    Fr    |    Al    |    Rm    |\n"
           << "+----------+----------+----------+----------+----------+----------+----------+\n"
           << "| "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.wavelength
           << " | "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.amplitude
           << " | "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.S
           << " | "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Rossby
           << " | "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Froude
           << " | "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Alfven
           << " | "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.magReynolds
           << " |\n"
           << "+----------+----------+----------+----------+----------+----------+----------+\n";
    else
        ss << "+----------+----------+----------+----------+----------+----------+\n"
           << "|    k     |    h     |    S     |    Fr    |    Al    |    Rm    |\n"
           << "+----------+----------+----------+----------+----------+----------+\n"
           << "| "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.wavelength
           << " | "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.amplitude
           << " | "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.S
           << " | "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Froude
           << " | "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.Alfven
           << " | "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.magReynolds
           << " |\n"
           << "+----------+----------+----------+----------+----------+----------+\n";

    std::cout << std::endl << ss.str() << std::endl;
    std::cout << std::fixed << std::flush;

   const double omega = 2. * numbers::PI * parameters.Froude;
   const double N = std::sqrt(parameters.S);

   if (omega > N)
       std::cout << "omega^2 > N^2" << std::endl;
   else if (omega < N)
       std::cout << "omega^2 < N^2" << std::endl;
}

template<int dim>
void TopographySolver<dim>::output_results(
        const unsigned int level,
        const bool initial_step) const
{
    std::cout << "   Output results..." << std::endl;

    PostProcessor<dim>  postprocessor;

    // prepare data out object
    DataOut<dim, DoFHandler<dim>>    data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution, postprocessor);

    // compute cell viscosity
    Vector<double>  cell_viscosity_density(triangulation.n_active_cells());
    Vector<double>  cell_viscosity_velocity(triangulation.n_active_cells());
    {
        QMidpoint<dim>      quadrature;

        FEValues<dim>       fe_values(fe_system,
                                      quadrature,
                                      update_values);

        const unsigned int n_q_points    = quadrature.size();

        // density part
        std::vector<double>         present_density_values(n_q_points);
        std::vector<Tensor<1,dim>>  present_density_gradients(n_q_points);

        // momentum part
        std::vector<Tensor<1,dim>>  present_velocity_values(n_q_points);
        std::vector<double>         present_velocity_divergences(n_q_points);

        const FEValuesExtractors::Scalar    density(0);
        const FEValuesExtractors::Vector    velocity(1);

        for (auto cell: dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);

            // compute present values for entropy viscosity
            fe_values[velocity].get_function_values(present_solution,
                                                    present_velocity_values);
            // entropy viscosity density equation
            const double nu_density = compute_density_viscosity(present_velocity_values,
                                                                cell->diameter());
            cell_viscosity_density(cell->index()) = nu_density;
            // entropy viscosity momentum equation
            const double nu_velocity = compute_velocity_viscosity(present_velocity_values,
                                                                  cell->diameter());
            cell_viscosity_velocity(cell->index()) = nu_velocity;
        }
    }
    data_out.add_data_vector(cell_viscosity_density,
                             "cell_viscosity_density");
    data_out.add_data_vector(cell_viscosity_velocity,
                             "cell_viscosity_velocity");

    data_out.build_patches(std::min(parameters.density_degree, parameters.velocity_degree));

    // write output to disk
    const std::string filename = ("solution-" +
                                  Utilities::int_to_string(level, 2) +
                                  (initial_step ? "-initial": "") +
                                  ".vtk");
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
}

template<int dim>
void TopographySolver<dim>::refine_mesh()
{
    TimerOutput::Scope timer_section(computing_timer, "refine mesh");

    std::cout << "   Mesh refinement..." << std::endl;

    // error estimation based on temperature
    Vector<float>   estimated_error_per_cell(triangulation.n_active_cells());
    const FEValuesExtractors::Vector    velocity();



    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim-1>(parameters.velocity_degree + 1),
                                       std::map<types::boundary_id, const Function<dim> *>(),
                                       present_solution,
                                       estimated_error_per_cell);
    // set refinement flags
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.8, 0.0);

    // preparing temperature solution transfer
    std::vector<BlockVector<double>> x_solution(1);
    x_solution[0] = present_solution;
    SolutionTransfer<dim,BlockVector<double>> solution_transfer(dof_handler);

    // preparing triangulation refinement
    triangulation.prepare_coarsening_and_refinement();
    solution_transfer.prepare_for_coarsening_and_refinement(x_solution);

    // refine triangulation
    triangulation.execute_coarsening_and_refinement();

    // setup dofs and constraints on refined mesh
    setup_dofs();

    // transfer of solution
    {
        std::vector<BlockVector<double>> tmp_solution(1);
        tmp_solution[0].reinit(present_solution);
        solution_transfer.interpolate(x_solution, tmp_solution);

        present_solution = tmp_solution[0];

        nonzero_constraints.distribute(present_solution);
    }
}

template<int dim>
void TopographySolver<dim>::run()
{
    bool initial_step = true;

    for (unsigned int cycle = 0; cycle < parameters.n_refinements; ++cycle)
    {
        std::cout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
        {
            make_grid();
            setup_dofs();
        }
        else
            refine_mesh();

        newton_iteration(parameters.tolerance,
                         parameters.max_iter,
                         initial_step,
                         cycle);


        const Tensor<1,dim> average_boundary_traction
        = compute_boundary_traction();

        std::cout << "   Average traction: " << average_boundary_traction << std::endl;

        output_results(cycle);

        if (cycle == 0)
            initial_step = false;

    }
}
}  // namespace TopographyProblem

// explicit instantiation
template class TopographyProblem::TopographySolver<3>;
