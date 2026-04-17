// Purpose: Unit tests for ShapeSampleSet sampling helpers.

#undef NDEBUG

#include "VecGeom/base/FpeEnable.h"

#include "VecGeom/base/Assert.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Tube.h"
#include "VecGeomTest/ApproxEqual.h"
#include "VecGeomTest/ShapeSampleSet.h"

using vecgeom::Precision;
using Vec_t = vecgeom::Vector3D<Precision>;

template <typename ShapeT>
void CheckSamples(ShapeT const *shape, vecgeom::test::ShapeSampleSet const &samples)
{
  for (int i = 0; i < samples.max_points_inside; ++i) {
    VECGEOM_ASSERT(shape->Inside(samples.points[i + samples.offset_inside]) == vecgeom::EInside::kInside);
  }

  for (int i = 0; i < samples.max_points_surface; ++i) {
    VECGEOM_ASSERT(shape->Inside(samples.points[i + samples.offset_surface]) == vecgeom::EInside::kSurface);
  }

  for (int i = 0; i < samples.max_points_outside; ++i) {
    VECGEOM_ASSERT(shape->Inside(samples.points[i + samples.offset_outside]) == vecgeom::EInside::kOutside);
  }

  for (int i = 0; i < samples.TotalPoints(); ++i) {
    VECGEOM_ASSERT(vecgeom::test::ApproxEqual<Precision>(samples.directions[i].Mag(), 1.0));
  }
}

void CheckEqualSampleSets(vecgeom::test::ShapeSampleSet const &lhs, vecgeom::test::ShapeSampleSet const &rhs)
{
  VECGEOM_ASSERT(lhs.TotalPoints() == rhs.TotalPoints());
  for (int i = 0; i < lhs.TotalPoints(); ++i) {
    VECGEOM_ASSERT(vecgeom::test::ApproxEqual(lhs.points[i], rhs.points[i]));
    VECGEOM_ASSERT(vecgeom::test::ApproxEqual(lhs.directions[i], rhs.directions[i]));
  }
}

void CheckDeterminismForBox()
{
  vecgeom::SimpleBox box("sampling-box", 10., 15., 20.);
  vecgeom::RNG rng;
  vecgeom::test::ShapeSampler sampler(rng);
  vecgeom::test::ShapeSamplingConfig config;
  config.max_points      = 120;
  config.inside_percent  = 25.;
  config.outside_percent = 25.;
  config.seed            = 17;
  config.reseed          = true;

  auto first  = sampler.Generate(&box, config);
  auto second = sampler.Generate(&box, config);

  VECGEOM_ASSERT(first.TotalPoints() == 120);
  VECGEOM_ASSERT(first.max_points_inside == 30);
  VECGEOM_ASSERT(first.max_points_outside == 30);
  VECGEOM_ASSERT(first.max_points_surface == 60);
  VECGEOM_ASSERT(first.max_points_edge == 0);

  CheckSamples(&box, first);
  CheckSamples(&box, second);
  CheckEqualSampleSets(first, second);
}

void CheckStreamSelectionForBox()
{
  vecgeom::SimpleBox box("sampling-box-streams", 10., 15., 20.);
  vecgeom::RNG rng;
  vecgeom::test::ShapeSampler sampler(rng);
  vecgeom::test::ShapeSamplingConfig config;
  config.max_points      = 80;
  config.inside_percent  = 25.;
  config.outside_percent = 25.;
  config.seed            = 17;
  config.reseed          = true;

  // Different stream ids should generate different randomized samples even
  // when they share the same base seed. Reusing a stream id must reproduce the
  // same sample set independent of execution order.
  config.stream_id   = 0;
  auto stream0_first = sampler.Generate(&box, config);
  auto stream0_again = sampler.Generate(&box, config);
  config.stream_id   = 1;
  auto stream1       = sampler.Generate(&box, config);

  CheckSamples(&box, stream0_first);
  CheckSamples(&box, stream0_again);
  CheckSamples(&box, stream1);
  CheckEqualSampleSets(stream0_first, stream0_again);

  bool found_difference = false;
  for (int i = 0; i < stream0_first.TotalPoints(); ++i) {
    if (!vecgeom::test::ApproxEqual(stream0_first.points[i], stream1.points[i]) ||
        !vecgeom::test::ApproxEqual(stream0_first.directions[i], stream1.directions[i])) {
      found_difference = true;
      break;
    }
  }
  VECGEOM_ASSERT(found_difference);
}

void CheckClassificationForTube()
{
  vecgeom::SimpleTube tube("sampling-tube", 5., 10., 20., 0., vecgeom::kTwoPi);
  vecgeom::RNG rng;
  vecgeom::test::ShapeSampler sampler(rng);
  vecgeom::test::ShapeSamplingConfig config;
  config.max_points                     = 90;
  config.inside_percent                 = 30.;
  config.outside_percent                = 30.;
  config.outside_max_radius_multiple    = 4.;
  config.outside_random_direction_ratio = 25.;
  config.seed                           = 23;
  config.reseed                         = true;

  auto samples = sampler.Generate(&tube, config);

  VECGEOM_ASSERT(samples.TotalPoints() == 90);
  VECGEOM_ASSERT(samples.max_points_inside == 27);
  VECGEOM_ASSERT(samples.max_points_outside == 27);
  VECGEOM_ASSERT(samples.max_points_surface == 36);

  CheckSamples(&tube, samples);
}

int main()
{
  CheckDeterminismForBox();
  CheckStreamSelectionForBox();
  CheckClassificationForTube();
  return 0;
}
