#include <iostream>
#include <exception>

#include "solver.h"

int main(int argc, char *argv[])
{
    try
    {
        using namespace TopographyProblem;

        std::string parameter_filename;
        if (argc>=2)
            parameter_filename = argv[1];
        else
            parameter_filename = "default_parameters.prm";

        Parameters  parameters(parameter_filename);

        TopographySolver<3> problem(parameters);
        problem.run();

    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
        return 1;
    }
    return 0;
}
