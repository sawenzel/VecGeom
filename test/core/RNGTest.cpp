// Purpose: Validate deterministic and thread-local RNG stream behavior.

#undef NDEBUG

#include "VecGeom/base/Assert.h"
#include "VecGeom/base/RNG.h"

#include <thread>
#include <vector>

using vecgeom::Precision;

namespace {

std::vector<Precision> GenerateStreamValues(unsigned long base_seed, unsigned long stream_id, int count)
{
  // Tests seed by logical stream id instead of thread identity so a failing
  // randomized case can be reproduced even when worker scheduling changes.
  vecgeom::RNG::SeedStream(base_seed, stream_id);
  auto &rng = vecgeom::RNG::Instance();

  std::vector<Precision> values(count);
  for (int i = 0; i < count; ++i) {
    values[i] = rng.uniform(-10., 10.);
  }
  return values;
}

void CheckRepeatedStreamSeeding()
{
  auto first  = GenerateStreamValues(13, 0, 8);
  auto second = GenerateStreamValues(13, 0, 8);
  auto other  = GenerateStreamValues(13, 1, 8);

  VECGEOM_ASSERT(vecgeom::RNG::MakeStreamSeed(13, 0) == vecgeom::RNG::MakeStreamSeed(13, 0));
  VECGEOM_ASSERT(vecgeom::RNG::MakeStreamSeed(13, 0) != vecgeom::RNG::MakeStreamSeed(13, 1));
  VECGEOM_ASSERT(first == second);
  VECGEOM_ASSERT(first != other);
}

void CheckThreadLocalIsolation()
{
  std::vector<Precision> run1_stream0, run1_stream1;
  std::vector<Precision> run2_stream0, run2_stream1;

  auto launch_pair = [](std::vector<Precision> &stream0, std::vector<Precision> &stream1) {
    std::thread worker0([&]() { stream0 = GenerateStreamValues(41, 0, 16); });
    std::thread worker1([&]() { stream1 = GenerateStreamValues(41, 1, 16); });
    worker0.join();
    worker1.join();
  };

  launch_pair(run1_stream0, run1_stream1);
  launch_pair(run2_stream0, run2_stream1);

  VECGEOM_ASSERT(run1_stream0 == run2_stream0);
  VECGEOM_ASSERT(run1_stream1 == run2_stream1);
  VECGEOM_ASSERT(run1_stream0 != run1_stream1);

  // The main thread owns a distinct thread-local instance, but the logical
  // stream contract must still reproduce the same values as worker 0.
  auto main_stream0 = GenerateStreamValues(41, 0, 16);
  VECGEOM_ASSERT(main_stream0 == run1_stream0);
}

} // namespace

int main()
{
  CheckRepeatedStreamSeeding();
  CheckThreadLocalIsolation();
  return 0;
}
