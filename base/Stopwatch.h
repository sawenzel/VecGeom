/// \file Stopwatch.h
/// \author Sandro Wenzel (sandro.wenzel@cern.ch)

#ifndef VECGEOM_BASE_STOPWATCH_H_
#define VECGEOM_BASE_STOPWATCH_H_

#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "base/Global.h"

// OS X compatibility
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#include <ctime>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
namespace standardtimer
{
   // this implementation is stripped from the TBB library ( so that we don't need to link against tbb )


typedef long long count_t;

inline long long now()
{
  count_t result;
  struct timespec ts;


#ifdef __MACH__
  // OS X compatibility code taken from
  // http://stackoverflow.com/questions/5167269/
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts.tv_sec = mts.tv_sec;
  ts.tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif

    result = static_cast<count_t>(1000000000UL)*static_cast<count_t>(ts.tv_sec)
           + static_cast<count_t>(ts.tv_nsec);
    return result;
}

inline double seconds( count_t value ) {
    return value*1E-9;
}

}

/**
 * @brief Timer for benchmarking purposes
 */
class Stopwatch {
private:
  standardtimer::count_t t1;
  standardtimer::count_t t2;

public:
  inline
  void Start() { t1 = standardtimer::now(); }

  /**
   * @return Elapsed time since start.
   */
  inline
  double Stop() {
    t2 = standardtimer::now();
    return Elapsed();
  }

  inline
  double Elapsed() const { return standardtimer::seconds( t2-t1 ); }
};

} } // End global namespace

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif

#endif // VECGEOM_BASE_STOPWATCH_H_
