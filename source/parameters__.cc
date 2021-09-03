/*
 * parameters.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

#include <parameters__.h>
#include <fstream>
#include <string>

namespace TopographyProblem {

Parameters::Parameters(const std::string &parameter_filename)
:
read_dimensional_input(false),
include_rotation(true),
constrain_pressure(false),
// geometry parameters
amplitude(50),
wavelength(1e5),
// physics parameters
Froude(1.0e-6),
S(1.0e-4),
Rossby(1.0e-4),
Alfven(0.0),
magReynolds(0.0),
// linear solver parameters
rel_tol(1e-6),
abs_tol(1e-12),
n_max_iter(100),
// discretization parameters
density_degree(1),
velocity_degree(2),
magnetic_degree(1),
// entropy viscosity parameters
c_density(0.5),
c_velocity(0.5),
default_viscosity(0.1),
// newton iteration control
tolerance(1e-12),
max_iter(15),
// refinement parameters
n_refinements(1),
n_initial_refinements(4),
n_boundary_refinements(1)
{
    ParameterHandler prm;
    declare_parameters(prm);

    std::ifstream parameter_file(parameter_filename.c_str());

    if (!parameter_file)
    {
        parameter_file.close();

        std::ostringstream message;
        message << "Input parameter file <"
                << parameter_filename << "> not found. Creating a"
                << std::endl
                << "template file of the same name."
                << std::endl;

        std::ofstream parameter_out(parameter_filename.c_str());
        prm.print_parameters(parameter_out,
                ParameterHandler::OutputStyle::Text);

        AssertThrow(false, ExcMessage(message.str().c_str()));
    }

    prm.parse_input(parameter_file);

    parse_parameters(prm);
}


void Parameters::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("geometry parameters");
    {
        prm.declare_entry("wave_length",
                          "1e5",
                          Patterns::Double(0.),
                          "wave length of the topography");

        prm.declare_entry("amplitude",
                          "50",
                          Patterns::Double(0.),
                          "amplitude of the topography");

    }
    prm.leave_subsection();

    prm.enter_subsection("runtime parameters");
    {
        prm.declare_entry("read_dimensional_input",
                          "false",
                          Patterns::Bool(),
                          "program reads dimensional parameter and computes"
                          "dimensionless numbers");

        prm.declare_entry("include_rotation",
                          "true",
                          Patterns::Bool(),
                          "flag to include Coriolis term");

        prm.declare_entry("constrain_pressure",
                          "false",
                          Patterns::Bool(),
                          "flag to constrain pressure field at the bottom");
    }
    prm.leave_subsection();

    prm.enter_subsection("dimensional physics parameters");
    {
        prm.declare_entry("buoyancy_frequency",
                "0.729e-4",
                Patterns::Double(0.),
                "buoyancy frequency in 1 / s");

        prm.declare_entry("reference_velocity",
                "5.0e-4",
                Patterns::Double(0.),
                "fluid velocity in m / s");

        prm.declare_entry("reference_rotation_rate",
                "0.729e-4",
                Patterns::Double(0.),
                "planetary rotation rate in 1 / s");

        prm.declare_entry("reference_density",
                "1.0e4",
                Patterns::Double(0.),
                "fluid density in kg / m^3");

        prm.declare_entry("reference_field",
                "0.65e-3",
                Patterns::Double(0.),
                "magnetic field in T");

        prm.declare_entry("reference_gravity",
                "10.0",
                Patterns::Double(0.),
                "gravitional acceleration in m / s^2");

        prm.declare_entry("magnetic_diffusivity",
                "0.8e-2",
                Patterns::Double(0.),
                "magnetic diffusivity in m^2 / s");
    }
    prm.leave_subsection();

    prm.enter_subsection("discretization parameters");
    {
        prm.declare_entry("velocity_degree",
                "2",
                Patterns::Integer(1,2),
                "Polynomial degree of the velocity discretization. The polynomial "
                "degree of the pressure is automatically set to one less than the velocity.");

        prm.declare_entry("density_degree",
                "1",
                Patterns::Integer(1,2),
                "Polynomial degree of the density discretization.");

        prm.declare_entry("magnetic_degree",
                "1",
                Patterns::Integer(1,2),
                "Polynomial degree of the magnetic discretization.");

        prm.enter_subsection("artificial viscosity parameters");
        {
            prm.declare_entry("c_density",
                    "0.5",
                    Patterns::Double(0.),
                    "viscosity control parameter for density equation");

            prm.declare_entry("c_velocity",
                    "0.5",
                    Patterns::Double(0.),
                    "viscosity control parameter for momentum equation");

            prm.declare_entry("default_viscosity",
                    "0.1",
                    Patterns::Double(0.),
                    "default viscosity applied in initial step");
        }
        prm.leave_subsection();

        prm.enter_subsection("refinement parameters");
        {
            prm.declare_entry("n_refinements",
                    "1",
                    Patterns::Integer(),
                    "number of refinements.");

            prm.declare_entry("n_initial_refinements",
                    "1",
                    Patterns::Integer(),
                    "number of initial refinements");

            prm.declare_entry("n_boundary_refinements",
                    "1",
                    Patterns::Integer(),
                    "number of initial boundary refinements");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("dimensionless physics parameters");
    {
        prm.declare_entry("Froude",
                "1.0e-6",
                Patterns::Double(0.),
                "Froude number");

        prm.declare_entry("stratification_number",
                "1.0e-4",
                Patterns::Double(),
                "dimensionless number describing strength of stratification: N^2 l / g ");

        prm.declare_entry("Rossby",
                "1.0e-4",
                Patterns::Double(0.),
                "Rossby number");

        prm.declare_entry("Alfven",
                "0.0e+0",
                Patterns::Double(),
                "Alfven number. Va / V, ratio of Alfven velocity to fluid velocity");

        prm.declare_entry("magReynolds",
                "0.0e+0",
                Patterns::Double(),
                "magnetic Reynolds number");

    }
    prm.leave_subsection();

    prm.enter_subsection("Newton iteration control");
    {
        prm.declare_entry("tolerance",
                "1e-6",
                Patterns::Double(),
                "tolerance for termination of Newton iteration");

        prm.declare_entry("max_iter",
                "15",
                Patterns::Integer(0),
                "maximum number of iterations for the Newton solver");
    }
    prm.leave_subsection();
}

void Parameters::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("geometry parameters");
    {
        wavelength = prm.get_double("wave_length");
        amplitude = prm.get_double("amplitude");
    }
    prm.leave_subsection();

    prm.enter_subsection("runtime parameters");
    {
        read_dimensional_input = prm.get_bool("read_dimensional_input");
        include_rotation = prm.get_bool("include_rotation");
        constrain_pressure = prm.get_bool("constrain_pressure");
    }
    prm.leave_subsection();

    prm.enter_subsection("discretization parameters");
    {
        velocity_degree = prm.get_integer("velocity_degree");
        density_degree = prm.get_integer("density_degree");
        magnetic_degree = prm.get_integer("magnetic_degree");

        prm.enter_subsection("refinement parameters");
        {
            n_refinements = prm.get_integer("n_refinements");
            n_initial_refinements = prm.get_integer("n_initial_refinements");
            n_boundary_refinements = prm.get_integer("n_boundary_refinements");
        }
        prm.leave_subsection();

        prm.enter_subsection("artificial viscosity parameters");
        {
            c_density = prm.get_double("c_density");
            c_velocity = prm.get_double("c_velocity");
            default_viscosity =  prm.get_double("default_viscosity");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();

    if (read_dimensional_input)
    {

        prm.enter_subsection("dimensional physics parameters");
        {
            buoyancy_frequency = prm.get_double("buoyancy_frequency");

            reference_rotation_rate = prm.get_double("reference_rotation_rate");
            reference_density = prm.get_double("reference_density");
            reference_velocity = prm.get_double("reference_velocity");
            reference_gravity = prm.get_double("reference_gravity");
            reference_field = prm.get_double("reference_field");

            compute_dimensionless_numbers();
        }
        prm.leave_subsection();
    }
    else
    {
        prm.enter_subsection("dimensionless physics parameters");
        {
            Rossby = prm.get_double("Rossby");
            Froude = prm.get_double("Froude");
            S = prm.get_double("stratification_number");
            Alfven= prm.get_double("Alfven");
            magReynolds = prm.get_double("magReynolds");
        }
        prm.leave_subsection();
    }

    prm.enter_subsection("Newton iteration control");
    {
        tolerance = prm.get_double("tolerance");
        max_iter = prm.get_integer("max_iter");
    }
    prm.leave_subsection();
}

void Parameters::compute_dimensionless_numbers()
{
    Froude = reference_velocity / std::sqrt(reference_gravity * wavelength);
    Rossby = reference_velocity / reference_rotation_rate / wavelength;
    S = buoyancy_frequency * buoyancy_frequency * wavelength / reference_gravity;

    const double magnetic_permeability =  4. * numbers::PI * 1.0e-7;
    const double alfven_velocity = reference_field
            / std::sqrt(reference_density * magnetic_permeability);
    Alfven = alfven_velocity / reference_velocity;
    magReynolds = reference_velocity * wavelength / magnetic_diffusivity;
}


}  // namespace BuoyantFluid
