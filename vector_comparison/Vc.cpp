#include <Vc/Vc>
#include "VectorTester.h"

void VectorTester::RunVc(double const * const __restrict__ a,
                           double const * const __restrict__ b,
                           double * const __restrict__ out, const int size) {

  for (int i = 0; i < size; i += kVectorSize) {

    Vc::double_v a_v(&a[i]);
    Vc::double_v b_v(&b[i]);

    b_v = b_v * (a_v + b_v) / a_v  + a_v * b_v;

    b_v.store(&out[i]);

  }

}