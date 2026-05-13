// Reusable ShapeTester sampling helpers.

#ifndef VECGEOM_TEST_VECGEOMTEST_SHAPESAMPLESET_HH
#define VECGEOM_TEST_VECGEOMTEST_SHAPESAMPLESET_HH

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "VecGeom/base/RNG.h"
#include "VecGeom/base/Vector3D.h"

namespace vecgeom {
namespace test {

using Precision = vecgeom::Precision;
using Vec_t     = vecgeom::Vector3D<Precision>;

struct ShapeSamplingConfig {
  int max_points                           = 10000;
  Precision inside_percent                 = 100.0 / 3;
  Precision outside_percent                = 100.0 / 3;
  Precision edge_percent                   = 0.0;
  Precision outside_max_radius_multiple    = 10.0;
  Precision outside_random_direction_ratio = 50.0;
  unsigned long seed                       = 0; // Base seed for deterministic randomized tests.
  unsigned long stream_id                  = 0; // Logical stream id so parallel workers can reproduce failures.
  bool reseed                              = false;
};

struct ShapeSampleSet {
  std::vector<Vec_t> points;
  std::vector<Vec_t> directions;

  int offset_inside      = 0;
  int offset_surface     = 0;
  int offset_edge        = 0;
  int offset_outside     = 0;
  int max_points_inside  = 0;
  int max_points_outside = 0;
  int max_points_surface = 0;
  int max_points_edge    = 0;

  int TotalPoints() const { return static_cast<int>(points.size()); }
};

class ShapeSampler {
public:
  explicit ShapeSampler(vecgeom::RNG &rng) : fRNG(rng) {}

  template <typename ImplT>
  ShapeSampleSet Generate(ImplT const *volume, const ShapeSamplingConfig &config)
  {
    ShapeSampleSet samples;
    if (config.max_points <= 0) return samples;

    if (config.reseed) {
      // The local RNG drives helper-owned sampling, while SamplePointOnSurface()
      // in many shapes still calls RNG::Instance() internally. Seed both with
      // the same logical stream so repeated runs stay reproducible and future
      // parallel test execution can map one stream id per worker.
      vecgeom::RNG::SeedEngine(fRNG, config.seed, config.stream_id);
      vecgeom::RNG::SeedStream(config.seed, config.stream_id);
    }

    const int requested_inside  = static_cast<int>(config.max_points * (config.inside_percent / 100));
    const int requested_outside = static_cast<int>(config.max_points * (config.outside_percent / 100));
    const int requested_edge    = static_cast<int>(config.max_points * (config.edge_percent / 100));
    const int requested_surface = config.max_points - requested_inside - requested_outside - requested_edge;

    // Store only successfully generated samples so later views never iterate
    // over default-filled points when a difficult shape cannot fill a bucket.
    samples.offset_inside = 0;
    samples.points.reserve(config.max_points);
    samples.directions.reserve(config.max_points);

    samples.max_points_inside = CreateInsideSamples(volume, requested_inside, samples);

    samples.offset_surface     = samples.TotalPoints();
    samples.max_points_surface = CreateSurfaceSamples(volume, requested_surface, samples);

    samples.offset_edge     = samples.TotalPoints();
    samples.max_points_edge = 0;
    if (requested_edge > 0) {
      ReportPartialSampleGeneration("edge", 0, requested_edge, 0);
    }

    samples.offset_outside     = samples.TotalPoints();
    samples.max_points_outside = CreateOutsideSamples(volume, config, requested_outside, samples);

    return samples;
  }

private:
  static int MaxConsecutiveRejectedAttempts(int requested)
  {
    // Rejection sampling is useful only while it keeps making progress; a
    // long consecutive-rejection streak usually means the predicate is broken.
    if (requested <= 0) return 0;
    return std::min(100000, std::max(1000, requested / 100));
  }

  static long long MaxTotalRejectedAttempts(int requested)
  {
    // Also cap total rejections so a very low but non-zero acceptance rate
    // cannot keep a test alive for minutes while occasionally resetting the
    // consecutive-rejection streak. Successful samples must not consume this
    // budget, otherwise high-statistics tests can be silently downsampled.
    if (requested <= 0) return 0;
    return std::max(1000LL, static_cast<long long>(requested) * 20);
  }

  static void ReportPartialSampleGeneration(const char *category, int generated, int requested, long long attempts)
  {
    std::cerr << "ShapeSampler generated only " << generated << "/" << requested << " " << category << " samples after "
              << attempts << " attempts." << std::endl;
  }

  template <typename Type>
  Type RandomRange(Type min, Type max)
  {
    return min + (max - min) * fRNG.uniform();
  }

  Vec_t RandomDirection()
  {
    Precision phi   = 2. * kPi * fRNG.uniform();
    Precision theta = vecgeom::ACos(1. - 2. * fRNG.uniform());
    Precision vx    = std::sin(theta) * std::cos(phi);
    Precision vy    = std::sin(theta) * std::sin(phi);
    Precision vz    = std::cos(theta);
    Vec_t vec(vx, vy, vz);
    vec.Normalize();
    return vec;
  }

  template <typename ImplT>
  int CreateSurfaceSamples(ImplT const *volume, int requested, ShapeSampleSet &samples)
  {
    int generated                = 0;
    long long attempts           = 0;
    int consecutive_rejections   = 0;
    const int max_rejections     = MaxConsecutiveRejectedAttempts(requested);
    const long long max_rejected = MaxTotalRejectedAttempts(requested);

    while (generated < requested && (attempts - generated) < max_rejected && consecutive_rejections < max_rejections) {
      ++attempts;
      Vec_t pointU;
      pointU = volume->GetUnplacedVolume()->SamplePointOnSurface();
      if (volume->Inside(pointU) != vecgeom::EInside::kSurface) {
        ++consecutive_rejections;
        continue;
      }

      samples.points.emplace_back(pointU.x(), pointU.y(), pointU.z());
      samples.directions.push_back(RandomDirection());
      ++generated;
      consecutive_rejections = 0;
    }

    if (generated < requested) {
      ReportPartialSampleGeneration("surface", generated, requested, attempts);
    }
    return generated;
  }

  template <typename ImplT>
  int CreateOutsideSamples(ImplT const *volume, const ShapeSamplingConfig &config, int requested,
                           ShapeSampleSet &samples)
  {
    Vec_t minExtent, maxExtent;
    volume->Extent(minExtent, maxExtent);
    Precision maxX = std::max(std::fabs(maxExtent.x()), std::fabs(minExtent.x()));
    Precision maxY = std::max(std::fabs(maxExtent.y()), std::fabs(minExtent.y()));
    Precision maxZ = std::max(std::fabs(maxExtent.z()), std::fabs(minExtent.z()));
    Precision rOut = std::sqrt(maxX * maxX + maxY * maxY + maxZ * maxZ);

    int generated                = 0;
    long long attempts           = 0;
    int consecutive_rejections   = 0;
    const int max_rejections     = MaxConsecutiveRejectedAttempts(requested);
    const long long max_rejected = MaxTotalRejectedAttempts(requested);
    while (generated < requested && (attempts - generated) < max_rejected && consecutive_rejections < max_rejections) {
      ++attempts;
      Vec_t vec, point;
      point.x() = -1 + 2 * fRNG.uniform();
      point.y() = -1 + 2 * fRNG.uniform();
      point.z() = -1 + 2 * fRNG.uniform();
      point *= rOut * config.outside_max_radius_multiple;
      if (volume->Inside(point) != vecgeom::EInside::kOutside) {
        ++consecutive_rejections;
        continue;
      }

      Precision random = fRNG.uniform();
      if (random <= config.outside_random_direction_ratio / 100.) {
        vec = RandomDirection();
      } else {
        Vec_t pointSurface = volume->GetUnplacedVolume()->SamplePointOnSurface();
        vec                = pointSurface - point;
        vec.Normalize();
      }

      samples.points.push_back(point);
      samples.directions.push_back(vec);
      ++generated;
      consecutive_rejections = 0;
    }

    if (generated < requested) {
      ReportPartialSampleGeneration("outside", generated, requested, attempts);
    }
    return generated;
  }

  template <typename ImplT>
  int CreateInsideSamples(ImplT const *volume, int requested, ShapeSampleSet &samples)
  {
    Vec_t minExtent, maxExtent;
    volume->Extent(minExtent, maxExtent);
    int generated                = 0;
    long long attempts           = 0;
    int consecutive_rejections   = 0;
    const int max_rejections     = MaxConsecutiveRejectedAttempts(requested);
    const long long max_rejected = MaxTotalRejectedAttempts(requested);
    while (generated < requested && (attempts - generated) < max_rejected && consecutive_rejections < max_rejections) {
      ++attempts;
      Precision x = RandomRange(minExtent.x(), maxExtent.x());
      Precision y = RandomRange(minExtent.y(), maxExtent.y());
      if (minExtent.y() == maxExtent.y()) y = RandomRange(-1000., +1000.);
      Precision z = RandomRange(minExtent.z(), maxExtent.z());
      Vec_t point0(x, y, z);
      if (volume->Inside(point0) == vecgeom::EInside::kInside) {
        samples.points.emplace_back(x, y, z);
        samples.directions.push_back(RandomDirection());
        ++generated;
        consecutive_rejections = 0;
      } else {
        ++consecutive_rejections;
      }
    }

    if (generated < requested) {
      ReportPartialSampleGeneration("inside", generated, requested, attempts);
    }
    return generated;
  }

  vecgeom::RNG &fRNG;
};

} // namespace test
} // namespace vecgeom

#endif
