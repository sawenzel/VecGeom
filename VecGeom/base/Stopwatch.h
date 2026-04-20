/// \file Stopwatch.h
/// \author Sandro Wenzel (sandro.wenzel@cern.ch)

#ifndef VECGEOM_BASE_STOPWATCH_H_
#define VECGEOM_BASE_STOPWATCH_H_

#include "VecGeom/base/Global.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <sys/times.h>
#include <unistd.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
namespace standardtimer {
// Use a monotonic host clock for elapsed-time measurements so benchmarking is
// not affected by wall-clock adjustments.
using steady_clock_t = std::chrono::steady_clock;
using count_t        = steady_clock_t::time_point;

inline count_t now() { return steady_clock_t::now(); }

inline double seconds(steady_clock_t::duration value) { return std::chrono::duration<double>(value).count(); }
} // namespace standardtimer

/**
 * @brief Lightweight host-side stopwatch for elapsed and CPU timing.
 *
 * `Elapsed()` measures monotonic wall time through `std::chrono::steady_clock`.
 * `CpuElapsed()` reports process user+system CPU time through `times(2)`.
 *
 * The class is intended for host-side benchmarking and progress reporting,
 * including host code compiled in CUDA translation units. It is not a device
 * timer.
 */
class Stopwatch {
private:
  standardtimer::count_t t1;
  standardtimer::count_t t2;
  double fCpuStart;
  double fCpuStop;

  static std::intmax_t GetTickFactor()
  {
    auto setter = []() {
      std::intmax_t result = ::sysconf(_SC_CLK_TCK);
      if (result <= 0) {
        fprintf(stderr,
                "Error StopWatch::GetTickFactor: Could not retrieve number of clock ticks per second (_SC_CLK_TCK).\n");
        result = -1;
      }
      return result;
    };
    static std::intmax_t result = setter();
    return result;
  }

  double GetCPUTime()
  {
    struct tms cpt;
    times(&cpt);
    return (double)(cpt.tms_utime + cpt.tms_stime) / GetTickFactor();
  }

public:
  inline void Start()
  {
    t1        = standardtimer::now();
    fCpuStart = GetCPUTime();
  }

  /**
   * @return Elapsed time since start.
   */
  inline double Stop()
  {
    t2       = standardtimer::now();
    fCpuStop = GetCPUTime();
    return Elapsed();
  }

  inline double Elapsed() const { return standardtimer::seconds(t2 - t1); }

  inline double CpuElapsed() const { return fCpuStop - fCpuStart; }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_BASE_STOPWATCH_H_
