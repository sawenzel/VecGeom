#include "VectorTester.h"
#include "Cilk.h"

void VectorTester::RunCilk(double const * const __restrict__ a,
                           double const * const __restrict__ b,
                           double * const __restrict__ out, const int size) {

  for (int i = 0; i < size; i += kVectorSize) {

    const double (* const a_v)[kVectorSize] __attribute__((aligned(32))) =
        reinterpret_cast<const double(* const)[kVectorSize]>(&a[i]);
    const double (* const b_v)[kVectorSize] __attribute__((aligned(32))) =
        reinterpret_cast<const double(* const)[kVectorSize]>(&b[i]);
    double (* const out_v)[kVectorSize] __attribute__((aligned(32))) =
        reinterpret_cast<double(* const)[kVectorSize]>(&out[i]);

    (*out_v)[:] = (*b_v)[:] * ((*a_v)[:] + (*b_v)[:]) / (*a_v)[:] +
                  (*a_v)[:] * (*b_v)[:];

  }

}

void VectorTester::RunCilkWrapped(double const * const __restrict__ a,
                                  double const * const __restrict__ b,
                                  double * const __restrict__ out,
                                  const int size) {

  for (int i = 0; i < size; i += kVectorSize) {

    const CilkFloat a_v(&a[i]);
    CilkFloat b_v(&b[i]);

    b_v = b_v * (a_v + b_v) / a_v  + a_v * b_v;

    b_v.Store(&out[i]);

  }

}