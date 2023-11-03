#ifndef XOSHIRO_H
#define XOSHIRO_H

/// \file xoshiro.h

#ifdef __cplusplus
#include "cstdint"
extern "C" {
#else
#include <stdint.h>
#endif

  uint64_t next(uint64_t x);
  void set_seed(uint64_t* x, uint64_t* s);
  double xoshiro_rand(uint64_t* s);
  void jump(uint64_t* s);
  void long_jump(uint64_t* s);

#ifdef __cplusplus
}
#endif

#endif
