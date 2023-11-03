#ifndef PHIINV_H
#define PHIINV_H

/// \file inv_std_normal.h

#ifdef __cplusplus
#include "cmath"
extern "C" {
#else
#include <math.h>
#endif

  extern int phi_inv(const double p, double* z);

#ifdef __cplusplus
}
#endif

#endif
