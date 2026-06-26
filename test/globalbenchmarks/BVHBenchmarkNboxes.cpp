//===-- test/globalbenchmarks/BVHBenchmarkNboxes.cpp --------------*- C++ -*-===//
//
// BVHBenchmarkNboxes — compare BVH implementations on a random box cloud
// ============================================================================
//
// What it measures
// ----------------
// This is a sibling of BVHBenchmark.cpp. Instead of taking the daughter solids
// of a real geometry volume and subdividing them into slab boxes, it builds the
// BVH directly from a synthetic set of `N` axis-aligned boxes of *varying size*
// scattered at *random positions* inside a cubic "room". This isolates the BVH
// tree layout/build and traversal cost from any geometry- or subdivision-related
// effect, and lets the box count, box-size distribution and room size be dialed
// freely from the command line to probe how each implementation scales.
//
// For each random ray (started inside the room) it finds the distance to the
// nearest box (a box's "DistanceToIn": the ray/AABB entry distance, clamped to 0
// when the ray starts inside the box). The same query is answered three ways and
// compared, exactly as in BVHBenchmark.cpp:
//
//   - plain   : production BVH (base/BVH.h, implicit-heap layout).
//   - (v2)    : BVH_V2 (base/BVH_V2.h, explicit-index layout, cost-based sweep SAH).
//   - (root)  : the vendored madmann91/bvh "v2" library that ROOT's
//               TGeoTessellated.cxx uses internally (geom/geom/inc/bvh/v2/).
//   - brute   : loop over all boxes, no acceleration structure. Reference.
//
// All BVHs consume the exact same boxes, so the comparison isolates the tree
// layout/build from everything else.
//
// How to read the output
// -----------------------
//   checksum  : sum of nearest-hit distances. Every method MUST equal the brute
//               reference; mismatches are flagged FAIL and the program exits
//               non-zero.
//   intersect : number of leaf-box intersection tests — the efficiency metric
//               (lower is better).
//
// Usage
// -----
//   BVHBenchmarkNboxes [-nboxes N] [-npoints N] [-room R] [-minsize a] [-maxsize b]
//                      [-seed S] [-bvh_depth D] [-nrep N]
//

#include "../benchmark/ArgParser.h"
#include "ScopedPerfCounters.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/AABB.h"
#include "VecGeom/base/BVH.h"
#include "VecGeom/base/BVH_V2.h"
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/base/Vector3D.h"

// Vendored madmann91/bvh "v2" header-only library that ROOT's TGeoTessellated.cxx builds against
// (geom/geom/inc/bvh/v2/, wrapped by bvh2_third_party.h, which also silences its third-party
// warnings). Exported as a public ROOT header, so it resolves through the same ROOT_INCLUDE_DIRS
// already required to build this benchmark -- no extra include path needed.
#include <bvh2_third_party.h>

using namespace vecgeom;

namespace {

// One row of the final comparison table.
struct Result {
  std::string name;
  size_t nboxes    = 0;  // BVH leaves (== nboxes for this benchmark)
  size_t intersect = 0;  // leaf-box intersection tests
  size_t innernodes = 0; // inner-node expansions (per ray-pass, one repetition)
  size_t leafnodes  = 0; // leaf-node visits     (per ray-pass, one repetition)
  double checksum  = 0.; // sum of nearest-hit distances
  double build     = 0.; // BVH build time [s]
  double query     = 0.; // fastest query pass over all rays [s]
};

// "DistanceToIn" for a single box: the ray/AABB entry distance, clamped to 0 when the
// ray origin is already inside the box, and +inf when the box is missed. Every method
// (brute and all three BVHs) routes its exact leaf test through this one function, so the
// reported checksums are directly comparable regardless of internal float/double handling.
inline double BoxDistanceToIn(AABB<double> const &box, Vector3D<double> const &point, Vector3D<double> const &dir)
{
  const double d = box.Distance(point, dir);
  if (d >= kInfLength) return kInfLength;
  return d < 0. ? 0. : d; // ray starts inside the box
}

// Generate `n` axis-aligned boxes of random size at random positions inside a cubic room
// centered at the origin with half-extent `room`. Box centers are uniform in the room; each
// box's full edge length per axis is uniform in [minSize, maxSize] (chosen independently per
// axis, so boxes are not just cubes). `boxes` is filled in the (min,max) corner-pair layout the
// VecGeom BVH constructors expect; `aabbs` holds the same boxes as AABB<double> for the exact
// leaf distance test.
void GenerateBoxes(int n, double room, double minSize, double maxSize, unsigned seed,
                   std::vector<Vector3D<double>> &boxes, std::vector<AABB<double>> &aabbs)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> posD(-room, room);
  std::uniform_real_distribution<double> sizeD(minSize, maxSize);

  boxes.clear();
  aabbs.clear();
  boxes.reserve(2 * n);
  aabbs.reserve(n);

  for (int i = 0; i < n; ++i) {
    const Vector3D<double> center(posD(rng), posD(rng), posD(rng));
    const Vector3D<double> half(0.5 * sizeD(rng), 0.5 * sizeD(rng), 0.5 * sizeD(rng));
    const Vector3D<double> lo = center - half;
    const Vector3D<double> hi = center + half;
    boxes.push_back(lo);
    boxes.push_back(hi);
    aabbs.emplace_back(lo, hi);
  }
}

// Random rays: origins uniform inside the cubic room, directions uniform on the unit sphere.
void GenerateRays(int n, double room, unsigned seed, SOA3D<double> &points, SOA3D<double> &directions)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> posD(-room, room);
  std::uniform_real_distribution<double> cosD(-1., 1.);
  std::uniform_real_distribution<double> phiD(0., 2. * M_PI);

  points.reserve(n);
  directions.reserve(n);
  for (int i = 0; i < n; ++i) {
    points.push_back(Vector3D<double>(posD(rng), posD(rng), posD(rng)));
    const double cosTheta = cosD(rng);
    const double sinTheta = std::sqrt(std::max(0., 1. - cosTheta * cosTheta));
    const double phi      = phiD(rng);
    directions.push_back(Vector3D<double>(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta));
  }
}

// Run the nearest-box query for every ray against a VecGeom BVH (production BVH or BVH_V2; both
// share the Intersect signature). The whole ray loop is repeated `nrep` times and the fastest pass
// is reported, which suppresses one-off noise (counts and checksum are identical each pass).
// check_leaf_bb=false: the hook does its own exact box test, so the BVH's per-primitive box check
// would be redundant.
template <typename BVHType>
Result BenchmarkBVH(std::string name, BVHType const &bvh, std::vector<AABB<double>> const &aabbs,
                    SOA3D<double> const &points, SOA3D<double> const &directions, int nrep)
{
  Result r;
  r.name   = std::move(name);
  r.nboxes = aabbs.size();
  r.query  = kInfLength;

  ScopedPerfCounters pc(r.name + " query");
  pc.Start();
  for (int rep = 0; rep < nrep; ++rep) {
    r.intersect  = 0;
    r.innernodes = 0;
    r.leafnodes  = 0;
    r.checksum   = 0.;
    auto innerHook = [&]() { ++r.innernodes; };
    auto countHook = [&]() { ++r.leafnodes; };
    Stopwatch timer;
    timer.Start();
    for (size_t i = 0; i < points.size(); ++i) {
      const Vector3D<double> ray_point = points[i];
      const Vector3D<double> ray_dir   = directions[i];
      double hit_distance              = kInfLength;
      auto hook                        = [&](BVHIntersectContext<float> &ctx) {
        ++r.intersect;
        const double dist = BoxDistanceToIn(aabbs[ctx.primID], ray_point, ray_dir);
        if (dist < hit_distance) {
          hit_distance = dist;
          ctx.step_max = dist; // shrink the search so the BVH can prune
        }
        return false;
      };
      bvh.template Intersect<false>(ray_point, ray_dir, kInfLength, hook, innerHook, countHook);
      if (hit_distance < kInfLength) r.checksum += hit_distance;
    }
    timer.Stop();
    r.query = std::min(r.query, timer.Elapsed());
  }
  pc.Stop();
  pc.Report();
  return r;
}

// Build and time one VecGeom BVHType from the (min,max) corner-pair boxes, then benchmark it.
// Both BVH<float> and BVH_V2<float> share the constructor signature (rootId, AABB corners, nChild,
// depth).
template <typename BVHType>
Result RunVecGeomBVH(std::string name, std::vector<Vector3D<double>> &boxes, std::vector<AABB<double>> const &aabbs,
                     int bvh_depth, SOA3D<double> const &points, SOA3D<double> const &directions, int nrep)
{
  Stopwatch build_timer;
  build_timer.Start();
  BVHType bvh(0, &boxes[0], boxes.size() / 2, bvh_depth);
  build_timer.Stop();
  auto r  = BenchmarkBVH(std::move(name), bvh, aabbs, points, directions, nrep);
  r.build = build_timer.Elapsed();
  return r;
}

// Same query as BenchmarkBVH above, but against a bvh::v2::Bvh built the same way
// TGeoTessellated::BuildBVH builds its facet BVH (DefaultBuilder, Quality::High) -- here over the
// random box cloud. `boxes` is the (min,max)-pair layout produced by GenerateBoxes.
Result BenchmarkRootBVH(std::string name, std::vector<Vector3D<double>> const &boxes,
                        std::vector<AABB<double>> const &aabbs, SOA3D<double> const &points,
                        SOA3D<double> const &directions, int nrep)
{
  using Scalar = float;
  using BBox   = bvh::v2::BBox<Scalar, 3>;
  using Vec3   = bvh::v2::Vec<Scalar, 3>;
  using Node   = bvh::v2::Node<Scalar, 3>;
  using Bvh    = bvh::v2::Bvh<Node>;
  using Ray    = bvh::v2::Ray<Scalar, 3>;

  const size_t nprim = aabbs.size();

  Result r;
  r.name   = std::move(name);
  r.nboxes = nprim;
  r.query  = kInfLength;

  Stopwatch build_timer;
  build_timer.Start();
  std::vector<BBox> bboxes;
  std::vector<Vec3> centers;
  bboxes.reserve(nprim);
  centers.reserve(nprim);
  for (size_t i = 0; i < nprim; ++i) {
    const Vec3 lo(static_cast<Scalar>(boxes[2 * i].x()), static_cast<Scalar>(boxes[2 * i].y()),
                  static_cast<Scalar>(boxes[2 * i].z()));
    const Vec3 hi(static_cast<Scalar>(boxes[2 * i + 1].x()), static_cast<Scalar>(boxes[2 * i + 1].y()),
                  static_cast<Scalar>(boxes[2 * i + 1].z()));
    bboxes.emplace_back(lo, hi);
    centers.push_back(bboxes.back().get_center());
  }
  typename bvh::v2::DefaultBuilder<Node>::Config config;
  config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High; // matches TGeoTessellated::BuildBVH
  Bvh bvh        = bvh::v2::DefaultBuilder<Node>::build(bboxes, centers, config);
  build_timer.Stop();
  r.build = build_timer.Elapsed();

  bvh::v2::GrowingStack<Bvh::Index> stack;

  ScopedPerfCounters pc(r.name + " query");
  pc.Start();
  for (int rep = 0; rep < nrep; ++rep) {
    r.intersect  = 0;
    r.innernodes = 0;
    r.leafnodes  = 0;
    r.checksum   = 0.;
    Stopwatch timer;
    timer.Start();
    for (size_t i = 0; i < points.size(); ++i) {
      const Vector3D<double> ray_point = points[i];
      const Vector3D<double> ray_dir   = directions[i];
      double hit_distance              = kInfLength;

      Ray ray(
          Vec3(static_cast<Scalar>(ray_point.x()), static_cast<Scalar>(ray_point.y()),
               static_cast<Scalar>(ray_point.z())),
          Vec3(static_cast<Scalar>(ray_dir.x()), static_cast<Scalar>(ray_dir.y()), static_cast<Scalar>(ray_dir.z())),
          Scalar(0.), std::numeric_limits<Scalar>::max());

      stack.clear();
      bvh.intersect<false, true>(
          ray, bvh.get_root().index, stack,
          [&](size_t begin, size_t end) {
            ++r.leafnodes;
            for (size_t slot = begin; slot < end; ++slot) {
              ++r.intersect;
              const int primid  = bvh.prim_ids[slot];
              const double dist = BoxDistanceToIn(aabbs[primid], ray_point, ray_dir);
              if (dist < hit_distance) {
                hit_distance = dist;
                ray.tmax     = static_cast<Scalar>(dist); // shrink the search so the BVH can prune
              }
            }
            return false;
          },
          [&](const Node &, const Node &) { ++r.innernodes; });
      if (hit_distance < kInfLength) r.checksum += hit_distance;
    }
    timer.Stop();
    r.query = std::min(r.query, timer.Elapsed());
  }
  pc.Stop();
  pc.Report();
  return r;
}

} // namespace

int main(int argc, char *argv[])
{
  OPTION_INT(nboxes, 1000);
  OPTION_INT(npoints, 10240);
  // Half-extent of the cubic room the boxes and ray origins live in.
  OPTION_DOUBLE(room, 100.);
  // Range of box edge lengths (full size, per axis, drawn independently).
  OPTION_DOUBLE(minsize, 1.);
  OPTION_DOUBLE(maxsize, 10.);
  OPTION_INT(seed, 12345);
  // Fixed BVH tree depth; 0 = pick automatically from the number of boxes.
  OPTION_INT(bvh_depth, 0);
  OPTION_INT(nrep, 1);
  // Isolate one implementation for clean perf-stat hardware-counter comparison: 0=all, 2=v2 only, 3=root only.
  // When set, brute-force reference + checksum validation are skipped so the process does only the selected traversal.
  OPTION_INT(only, 0);

  if (nboxes < 1) {
    std::cerr << "nboxes must be >= 1\n";
    return 1;
  }
  if (nrep < 1) nrep = 1;
  if (maxsize < minsize) std::swap(minsize, maxsize);

  std::cout << "BVH benchmark over " << nboxes << " random boxes in a room of half-extent " << room << " (box size in ["
            << minsize << ", " << maxsize << "]), " << npoints << " rays\n";

  // Generate the random box cloud (shared by every BVH) and the random rays.
  std::vector<Vector3D<double>> boxes;
  std::vector<AABB<double>> aabbs;
  GenerateBoxes(nboxes, room, minsize, maxsize, (unsigned)seed, boxes, aabbs);

  SOA3D<double> points, directions;
  GenerateRays(npoints, room, (unsigned)seed + 1u, points, directions);

  std::vector<Result> results;

  // Reference: brute-force loop over all boxes (fastest of nrep passes). Skipped under -only so the
  // process under perf stat does only the one selected traversal.
  if (only == 0) {
    Result r;
    r.name   = "brute";
    r.nboxes = aabbs.size();
    r.query  = kInfLength;
    for (int rep = 0; rep < nrep; ++rep) {
      r.intersect = 0;
      r.checksum  = 0.;
      Stopwatch timer;
      timer.Start();
      for (size_t i = 0; i < points.size(); ++i) {
        double hit_distance = kInfLength;
        for (size_t d = 0; d < aabbs.size(); ++d) {
          const double dist = BoxDistanceToIn(aabbs[d], points[i], directions[i]);
          if (dist < hit_distance) hit_distance = dist;
        }
        r.intersect += aabbs.size();
        if (hit_distance < kInfLength) r.checksum += hit_distance;
      }
      timer.Stop();
      r.query = std::min(r.query, timer.Elapsed());
    }
    results.push_back(r);
  }

  // The three BVH implementations, all built from the exact same boxes.
  if (only == 0) results.push_back(RunVecGeomBVH<BVH<float>>("plain", boxes, aabbs, bvh_depth, points, directions, nrep));
  if (only == 0 || only == 2)
    results.push_back(RunVecGeomBVH<BVH_V2<float>>("plain (v2)", boxes, aabbs, bvh_depth, points, directions, nrep));
  if (only == 0 || only == 3)
    results.push_back(BenchmarkRootBVH("plain (root)", boxes, aabbs, points, directions, nrep));

  // Report. Under -only there is no brute reference, so checksum validation is skipped.
  const double reference = only == 0 ? results.front().checksum : results.front().checksum;
  const double tolerance = only == 0 ? 1e-6 * (reference != 0. ? std::abs(reference) : 1.) : kInfLength;
  bool all_ok            = true;

  std::printf("\n(times are the fastest of %d pass%s)\n", nrep, nrep == 1 ? "" : "es");
  std::printf("%-16s %10s %12s %12s %12s %14s %10s %10s   %s\n", "method", "boxes", "intersect", "innernodes",
              "leafnodes", "checksum", "build[s]", "query[s]", "status");

  for (auto const &r : results) {
    const bool ok = std::abs(r.checksum - reference) <= tolerance;
    all_ok &= ok;
    std::printf("%-16s %10zu %12zu %12zu %12zu %14.4f %10.5f %10.5f   %s\n", r.name.c_str(), r.nboxes, r.intersect,
                r.innernodes, r.leafnodes, r.checksum, r.build, r.query, ok ? "OK" : "FAIL");
  }

  if (!all_ok) {
    std::cerr << "\nFAIL: a BVH disagrees with the brute-force reference\n";
    return 1;
  }
  return 0;
}
