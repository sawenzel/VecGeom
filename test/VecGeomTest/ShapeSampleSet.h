// Reusable ShapeTester sampling helpers.

#ifndef VECGEOM_TEST_VECGEOMTEST_SHAPESAMPLESET_HH
#define VECGEOM_TEST_VECGEOMTEST_SHAPESAMPLESET_HH

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

    samples.max_points_inside  = static_cast<int>(config.max_points * (config.inside_percent / 100));
    samples.max_points_outside = static_cast<int>(config.max_points * (config.outside_percent / 100));
    samples.max_points_edge    = static_cast<int>(config.max_points * (config.edge_percent / 100));
    samples.max_points_surface =
        config.max_points - samples.max_points_inside - samples.max_points_outside - samples.max_points_edge;

    samples.offset_inside  = 0;
    samples.offset_surface = samples.max_points_inside;
    samples.offset_edge    = samples.offset_surface + samples.max_points_surface;
    samples.offset_outside = samples.offset_edge + samples.max_points_edge;

    samples.points.resize(config.max_points);
    samples.directions.resize(config.max_points);

    CreateOutsideSamples(volume, config, samples);
    CreateInsideSamples(volume, samples);
    CreateSurfaceSamples(volume, samples);

    return samples;
  }

private:
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
  void CreateSurfaceSamples(ImplT const *volume, ShapeSampleSet &samples)
  {
    Vec_t point;
    for (int i = 0; i < samples.max_points_surface; ++i) {
      Vec_t pointU;
      int retry = 100;
      do {
        pointU                                         = volume->GetUnplacedVolume()->SamplePointOnSurface();
        samples.directions[i + samples.offset_surface] = RandomDirection();
        point.Set(pointU.x(), pointU.y(), pointU.z());
        samples.points[i + samples.offset_surface] = point;
        if (retry-- == 0) {
          std::cout << "Couldn't find point on surface in 100 trials, so skipping this point." << std::endl;
          break;
        }
      } while (volume->Inside(pointU) != vecgeom::EInside::kSurface);
    }
  }

  template <typename ImplT>
  void CreateOutsideSamples(ImplT const *volume, const ShapeSamplingConfig &config, ShapeSampleSet &samples)
  {
    Vec_t minExtent, maxExtent;
    volume->Extent(minExtent, maxExtent);
    Precision maxX = std::max(std::fabs(maxExtent.x()), std::fabs(minExtent.x()));
    Precision maxY = std::max(std::fabs(maxExtent.y()), std::fabs(minExtent.y()));
    Precision maxZ = std::max(std::fabs(maxExtent.z()), std::fabs(minExtent.z()));
    Precision rOut = std::sqrt(maxX * maxX + maxY * maxY + maxZ * maxZ);

    for (int i = 0; i < samples.max_points_outside; ++i) {
      Vec_t vec, point;
      do {
        point.x() = -1 + 2 * fRNG.uniform();
        point.y() = -1 + 2 * fRNG.uniform();
        point.z() = -1 + 2 * fRNG.uniform();
        point *= rOut * config.outside_max_radius_multiple;
      } while (volume->Inside(point) != vecgeom::EInside::kOutside);

      Precision random = fRNG.uniform();
      if (random <= config.outside_random_direction_ratio / 100.) {
        vec = RandomDirection();
      } else {
        Vec_t pointSurface = volume->GetUnplacedVolume()->SamplePointOnSurface();
        vec                = pointSurface - point;
        vec.Normalize();
      }

      samples.points[i + samples.offset_outside]     = point;
      samples.directions[i + samples.offset_outside] = vec;
    }
  }

  template <typename ImplT>
  void CreateInsideSamples(ImplT const *volume, ShapeSampleSet &samples)
  {
    Vec_t minExtent, maxExtent;
    volume->Extent(minExtent, maxExtent);
    int i = 0;
    while (i < samples.max_points_inside) {
      Precision x = RandomRange(minExtent.x(), maxExtent.x());
      Precision y = RandomRange(minExtent.y(), maxExtent.y());
      if (minExtent.y() == maxExtent.y()) y = RandomRange(-1000., +1000.);
      Precision z = RandomRange(minExtent.z(), maxExtent.z());
      Vec_t point0(x, y, z);
      if (volume->Inside(point0) == vecgeom::EInside::kInside) {
        Vec_t point(x, y, z);
        samples.points[i + samples.offset_inside]     = point;
        samples.directions[i + samples.offset_inside] = RandomDirection();
        ++i;
      }
    }
  }

  vecgeom::RNG &fRNG;
};

} // namespace test
} // namespace vecgeom

#endif
