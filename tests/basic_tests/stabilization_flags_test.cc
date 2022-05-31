/*
 * stabilization_flags_test.cc
 *
 *  Created on: Mar 22, 2022
 *      Author: sg
 */

#include <stabilization_flags.h>
#include <parameters.h>


namespace TestSpace
{

template<typename Stream>
void print_flag(Stream &stream, const StabilizationFlags &f)
{
    std::stringstream sstream;
    sstream << f;

    Utility::add_line(stream, "Stabilization type", sstream.str().c_str());

    return;
}

void test()
{
  print_flag(std::cout, StabilizationFlags::apply_grad_div);
  print_flag(std::cout, StabilizationFlags::apply_none);
  print_flag(std::cout, StabilizationFlags::apply_pspg);
  print_flag(std::cout, StabilizationFlags::apply_supg);
  print_flag(std::cout, StabilizationFlags::apply_grad_div|
                        StabilizationFlags::apply_pspg);
  print_flag(std::cout, StabilizationFlags::apply_grad_div|
                        StabilizationFlags::apply_supg);
  print_flag(std::cout, StabilizationFlags::apply_pspg|
                        StabilizationFlags::apply_supg);
  print_flag(std::cout, StabilizationFlags::apply_grad_div|
                        StabilizationFlags::apply_pspg|
                        StabilizationFlags::apply_supg);
}

}


int main(void)
{
  try
  {
    TestSpace::test();
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
