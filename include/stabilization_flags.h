/*
 * stabilization_flag.h
 *
 *  Created on: Sep 26, 2021
 *      Author: sg
 */

#ifndef INCLUDE_STABILIZATION_FLAGS_H_
#define INCLUDE_STABILIZATION_FLAGS_H_

#include <sstream>
#include <string>

/*!
 * @brief Enumeration for the stabilization terms.
 */
enum StabilizationFlags
{
  apply_none = 0,

  apply_supg = 0x0001,

  apply_pspg = 0x0002,

  apply_grad_div = 0x0004
};

// inline functions
template <class StreamType>
inline StreamType &
operator<<(StreamType &s, const StabilizationFlags u)
{
  if (u == apply_none)
  {
    s << "none";
    return s;
  }

  std::stringstream ss;

  if (u & apply_supg)
    ss << "SUPG|";
  if (u & apply_pspg)
    ss << "PSPG|";
  if (u & apply_grad_div)
    ss << "GradDiv|";

  std::string string{ss.str()};
  string.pop_back();
  s << string.c_str();

  return s;
}



inline StabilizationFlags
operator|(const StabilizationFlags f1, const StabilizationFlags f2)
{
  return static_cast<StabilizationFlags>(static_cast<unsigned int>(f1) |
                                         static_cast<unsigned int>(f2));
}



inline StabilizationFlags &
operator|=(StabilizationFlags &f1, const StabilizationFlags f2)
{
  f1 = f1 | f2;
  return f1;
}



inline StabilizationFlags operator&(const StabilizationFlags f1, const StabilizationFlags f2)
{
  return static_cast<StabilizationFlags>(static_cast<unsigned int>(f1) &
                                         static_cast<unsigned int>(f2));
}



inline StabilizationFlags &
operator&=(StabilizationFlags &f1, const StabilizationFlags f2)
{
  f1 = f1 & f2;
  return f1;
}



#endif /* INCLUDE_STABILIZATION_FLAGS_H_ */
