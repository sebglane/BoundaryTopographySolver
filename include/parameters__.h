/*
 * parameters.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_PARAMETERS___H_
#define INCLUDE_PARAMETERS___H_

#include <deal.II/base/parameter_handler.h>

namespace TopographyProblem {

using namespace dealii;

struct Parameters
{
    Parameters(const std::string &parameter_filename);
    static void declare_parameters(ParameterHandler &prm);
    void parse_parameters(ParameterHandler &prm);

    void compute_dimensionless_numbers();

    // runtime parameters
    bool    read_dimensional_input;
    bool    include_rotation;
    bool    constrain_pressure;

    // mesh parameters
    double amplitude;
    double wavelength;

    // physics parameters
    double buoyancy_frequency;
    double magnetic_diffusivity;

    double reference_rotation_rate;
    double reference_velocity;
    double reference_gravity;
    double reference_density;
    double reference_field;

    // dimensionless physics parameters
    double Froude;
    double S;
    double Rossby;
    double Alfven;
    double magReynolds;

    // linear solver parameters
    double          rel_tol;
    double          abs_tol;
    unsigned int    n_max_iter;

    // discretization parameters
    unsigned int density_degree;
    unsigned int velocity_degree;
    unsigned int magnetic_degree;

    // entropy viscosity parameters
    double  c_density;
    double  c_velocity;
    double  default_viscosity;

    // newton iteration
    double          tolerance;
    unsigned int    max_iter;

    // refinement parameters
    unsigned int n_refinements;
    unsigned int n_initial_refinements;
    unsigned int n_boundary_refinements;
};


}  // namespace BuoyantFluid

#endif /* INCLUDE_PARAMETERS___H_ */
