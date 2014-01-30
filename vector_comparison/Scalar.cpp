#include "VectorTester.h"

void VectorTester::RunScalar(double const * const __restrict__ a,
                             double const * const __restrict__ b,
                             double * const __restrict__ out, const int size) {

  for (int i = 0; i < size; ++i) {

    out[i] = b[i] * (a[i] + b[i]) / a[i] + a[i] * b[i];

  }

}