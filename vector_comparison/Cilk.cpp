#include "VectorTester.h"

void VectorTester::RunCilk(double const * const a, double const * const b,
                           double * const out, const int size) {

  for (int i = 0; i < size; i += kVectorSize) {

    const double (* const a_v)[kVectorSize] __attribute__((aligned(32))) =
        reinterpret_cast<const double(* const)[kVectorSize]>(&a[i]);
    const double (* const b_v)[kVectorSize] __attribute__((aligned(32))) =
        reinterpret_cast<const double(* const)[kVectorSize]>(&b[i]);
    double (* const out_v)[kVectorSize] __attribute__((aligned(32))) =
        reinterpret_cast<double(* const)[kVectorSize]>(&out[i]);

    (*out_v)[:] = (*b_v)[:] * ((*a_v)[:] + (*b_v)[:]);

  }

}