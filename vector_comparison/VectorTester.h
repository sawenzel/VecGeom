#ifndef VECTORTESTER_H
#define VECTORTESTER_H

#include <Vc/Vc>
#include "tbb/tick_count.h"

class VectorTester {

public:

  static constexpr int kVectorSize = Vc::double_v::Size;

  static void RunTest(const int data_size = 1048576,
                      const int iterations = 1024);

private:

  VectorTester() {}

  static double Random(const double low, const double high);
  static void RunVc(double const * const a, double const * const b,
                    double * const out, const int size);
  static void RunCilk(double const * const a, double const * const b,
                      double * const out, const int size);
  static void RunCilkWrapped(double const * const a, double const * const b,
                             double * const out, const int size);
  static void RunScalar(double const * const a, double const * const b,
                        double * const out, const int size);

};

struct Stopwatch {
  tbb::tick_count t1;
  tbb::tick_count t2;
  void Start() { t1 = tbb::tick_count::now(); }
  double Stop() {
    t2 = tbb::tick_count::now();
    return Elapsed();
  }
  double Elapsed() const { return (t2-t1).seconds(); }
};

#endif /* VECTORTESTER_H */