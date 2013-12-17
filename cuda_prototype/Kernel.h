#ifndef KERNEL_H
#define KERNEL_H

#include "Library.h"

#ifdef __CUDACC__
__global__
#endif
template <Ct ct>
inline __attribute__((always_inline))
CtTraits<ct>::bool_t Contains(typename const CtTraits<ct>::float_t a,
                              typename const CtTraits<ct>::float_t b ...

#endif /* KERNEL_H */