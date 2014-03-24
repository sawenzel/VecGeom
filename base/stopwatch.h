/**
 * @file stopwatch.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_STOPWATCH_H_
#define VECGEOM_BASE_STOPWATCH_H_

#include <ctime>

/**
 * @brief Timer for benchmarking purposes
 */
namespace standardtimer
{
	// this implementation is stripped from the TBB library ( so that we don't need to link against tbb )

typedef long long count_t;

inline long long now()
{
    count_t result;
    struct timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );
    result = static_cast<count_t>(1000000000UL)*static_cast<count_t>(ts.tv_sec)
    		 + static_cast<count_t>(ts.tv_nsec);
    return result;
}

inline double seconds( count_t value ) {
    return value*1E-9;
}
}

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

#endif // VECGEOM_BASE_STOPWATCH_H_
