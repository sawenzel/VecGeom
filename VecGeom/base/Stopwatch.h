/// \file Stopwatch.h
/// \author Sandro Wenzel (sandro.wenzel@cern.ch)

#ifndef VECGEOM_BASE_STOPWATCH_H_
#define VECGEOM_BASE_STOPWATCH_H_

#include "VecGeom/base/Global.h"

// OS X compatibility
#if defined(__MACH__) && defined(__APPLE__)
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#include <ctime>
#include <unistd.h>
#include <sys/times.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
namespace standardtimer {
// this implementation is stripped from the TBB library ( so that we don't need to link against tbb )

typedef long long count_t;

inline long long now()
{
  count_t result;
  struct timespec ts;

#if defined(__MACH__) && defined(__APPLE__)
  // OS X compatibility code taken from
  // http://stackoverflow.com/questions/5167269/
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts.tv_sec  = mts.tv_sec;
  ts.tv_nsec = mts.tv_nsec;
#else
  clock_gettime(CLOCK_REALTIME, &ts);
#endif

  result = static_cast<count_t>(1000000000UL) * static_cast<count_t>(ts.tv_sec) + static_cast<count_t>(ts.tv_nsec);
  return result;
}

inline double seconds(count_t value) { return value * 1E-9; }
} // namespace standardtimer

/**
 * @brief Timer for benchmarking purposes
 */
class Stopwatch {
  // Note see http://jogojapan.github.io/blog/2012/11/25/measuring-cpu-time/
  // for some interesting ideas on how to implement in a
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
