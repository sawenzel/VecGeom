#include <iostream>
#include "VectorTester.h"
#include "mm_malloc.h"

double VectorTester::Random(const double low, const double high) {
  return low + static_cast<double>(rand()) /
               static_cast<double>(RAND_MAX / (high - low));
}

void VectorTester::RunTest(const int data_size, const int iterations) {
  
  double * const a = reinterpret_cast<double * const>(
    _mm_malloc(data_size*sizeof(double), 32)
  );
  double * const b = reinterpret_cast<double * const>(
    _mm_malloc(data_size*sizeof(double), 32)
  );
  double * const out_vc = reinterpret_cast<double * const>(
    _mm_malloc(data_size*sizeof(double), 32)
  );
  double * const out_cilk = reinterpret_cast<double * const>(
    _mm_malloc(data_size*sizeof(double), 32)
  );
  double * const out_cilkwrapped = reinterpret_cast<double * const>(
    _mm_malloc(data_size*sizeof(double), 32)
  );
  double * const out_scalar = reinterpret_cast<double * const>(
    _mm_malloc(data_size*sizeof(double), 32)
  );
  for (int i = 0; i < data_size; ++i) {
    a[i] = Random(-10, 10);
    b[i] = Random(-10, 10);
  }

  double time_vc = 0, time_cilk = 0, time_cilkwrapped = 0, time_scalar = 0;
  double mismatches = 0;
  const double average_divisor = 1 / double(iterations);

  for (int i = 0; i < iterations; ++i) {
    Stopwatch timer;
    timer.Start();
    RunVc(a, b, out_vc, data_size);
    time_vc += timer.Stop() * average_divisor;
    timer.Start();
    RunCilk(a, b, out_cilk, data_size);
    time_cilk += timer.Stop() * average_divisor;
    timer.Start();
    RunCilkWrapped(a, b, out_cilkwrapped, data_size);
    time_cilkwrapped += timer.Stop() * average_divisor;
    timer.Start();
    RunScalar(a, b, out_scalar, data_size);
    time_scalar += timer.Stop() * average_divisor;
    int mismatches_local = 0;
    for (int j = 0; j < data_size; ++j) {
      if ((out_vc[j] - out_cilk[j] > 1e-12 && out_vc[j] != out_cilk[j])     ||
          (out_vc[j] - out_scalar[j] > 1e-12 && out_vc[j] != out_scalar[j]) ||
          (out_vc[j] - out_cilkwrapped[j] > 1e-12 &&
           out_vc[j] != out_cilkwrapped[j])) {
        mismatches_local++;
      }
    }
    mismatches += double(mismatches_local) * average_divisor;
  }

  _mm_free(a);
  _mm_free(b);
  _mm_free(out_vc);
  _mm_free(out_cilk);
  _mm_free(out_cilkwrapped);
  _mm_free(out_scalar);

  std::cout << "Average Vc runtime:           " << time_vc << "s\n"
            << "Average Cilk runtime:         " << time_cilk << "s\n"
            << "Average Cilk wrapped runtime: " << time_cilkwrapped << "s\n"
            << "Average scalar runtime:       " << time_scalar << "s\n"
            << "Average output mismatch: " << mismatches << " / " << data_size
                                           << std::endl;
}