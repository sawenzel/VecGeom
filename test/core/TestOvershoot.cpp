//
// File:    TestOvershoot.cpp
// Purpose: Unit tests for the ULP function implementation to prevent overshooting
//          ULP (unit in the last place) is used to estimate the maximum
//          rounding error made for approaching the bounding box of a volume

//-- ensure asserts are compiled in
#undef NDEBUG

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/volumes/utilities/VolumeUtilities.h>

int test(int n)
{
  using namespace vecgeom;
  using Vec3D  = Vector3D<Precision>;
  using Vec3Df = Vector3D<float>;
  Vec3D *points_near, *points_far, *dirs;
  Precision *dist_small = new Precision[n];
  Precision *dist_large = new Precision[n];
  points_near           = new Vec3D[n];
  points_far            = new Vec3D[n];
  dirs                  = new Vec3D[n];

  Vec3D amin_small(-10., -10., -10.), amax_small(10., 10., 10.);
  Vec3D amin_large(-100000., -100000., -100000.), amax_large(100000., 100000., 100000.);
  vecgeom::volumeUtilities::FillRandomPoints(amin_small, amax_small, points_near, n);
  vecgeom::volumeUtilities::FillRandomPoints(amin_large, amax_large, points_far, n);
  vecgeom::volumeUtilities::FillRandomDirections(dirs, n);
  for (auto i = 0; i < n; ++i) {
    dist_small[i] = 10. * RNG::Instance().uniform();
    dist_large[i] = 100000. * RNG::Instance().uniform();
  }

  auto undershoot_distance = [](Vec3Df const &point, float distance) {
    float approach = 0.f;

    // Overestimate error to 10 ULP (20 roundings)
    float err = 10.f * ULP<float>(vecCore::math::Max(point.Abs().Max(), distance));
    if (distance > 1.f) {
      approach = vecCore::math::Max(distance - err, 0.f);
    }

    // approach = std::nextafter(distance * (1.0f - 2.0f * kEpsilonT<float>), 0.0f);
    return approach;
  };

  auto propagation_error_near_large = [&](int i, double &overshoot) {
    // Shoot from near points with large distance to target
    Vec3D reach        = points_near[i] + dirs[i] * dist_large[i];
    Vec3Df reach_float = Vec3Df(points_near[i]) + Vec3Df(dirs[i]) * float(dist_large[i]);
    double error       = (reach - Vec3D(reach_float)).Mag();
    // Try to undershoot and check if this is really the case
    float approach = undershoot_distance(Vec3Df(points_near[i]), float(dist_large[i]));
    // Aproach target in double
    Vec3D approached_point = points_near[i] + dirs[i] * double(approach);
    // Calculate the overshoot, which must be negative
    overshoot = (approached_point - reach).Dot(dirs[i]);
    VECGEOM_ASSERT(overshoot <= 0. && "near_large overshoots");
    if (approach == 0.f) overshoot = -1.;
    return error;
  };

  auto propagation_error_far_small = [&](int i, double &overshoot) {
    // Shoot from far points with small distance to target
    Vec3D reach        = points_far[i] + dirs[i] * dist_small[i];
    Vec3Df reach_float = Vec3Df(points_far[i]) + Vec3Df(dirs[i]) * float(dist_small[i]);
    double error       = (reach - Vec3D(reach_float)).Mag();
    // Try to undershoot and check if this is really the case
    float approach = undershoot_distance(Vec3Df(points_far[i]), float(dist_small[i]));
    // Aproach target in double
    Vec3D approached_point = points_far[i] + dirs[i] * double(approach);
    // Calculate the overshoot, which must be negative
    overshoot = (approached_point - reach).Dot(dirs[i]);
    VECGEOM_ASSERT(overshoot <= 0. && "far_small overshoots");
    if (approach == 0.f) overshoot = -1.;
    return error;
  };

  auto propagation_error_far_large = [&](int i, double &overshoot) {
    // Shoot from far points with large distance to target
    Vec3D reach        = points_far[i] + dirs[i] * dist_large[i];
    Vec3Df reach_float = Vec3Df(points_far[i]) + Vec3Df(dirs[i]) * float(dist_large[i]);
    double error       = (reach - Vec3D(reach_float)).Mag();
    // Try to undershoot and check if this is really the case
    float approach = undershoot_distance(Vec3Df(points_far[i]), float(dist_large[i]));
    // Aproach target in double
    Vec3D approached_point = points_far[i] + dirs[i] * double(approach);
    // Calculate the overshoot, which must be negative
    overshoot = (approached_point - reach).Dot(dirs[i]);
    VECGEOM_ASSERT(overshoot <= 0. && "far_large overshoots");
    if (approach == 0.f) overshoot = -1.;
    return error;
  };

  double err_max_near_large = 0.;
  double err_max_far_small  = 0.;
  double err_max_far_large  = 0.;
  double largest_overshoot  = -1.;
  double overshoot;

  for (auto i = 0; i < n; ++i) {
    auto error         = propagation_error_near_large(i, overshoot);
    err_max_near_large = vecCore::math::Max(err_max_near_large, error);
    largest_overshoot  = vecCore::math::Max(largest_overshoot, overshoot);
    error              = propagation_error_far_small(i, overshoot);
    err_max_far_small  = vecCore::math::Max(err_max_far_small, error);
    largest_overshoot  = vecCore::math::Max(largest_overshoot, overshoot);
    error              = propagation_error_far_large(i, overshoot);
    err_max_far_large  = vecCore::math::Max(err_max_far_large, error);
    largest_overshoot  = vecCore::math::Max(largest_overshoot, overshoot);
  }
  printf("Max error for near points with large distance to target: %g\n", err_max_near_large);
  printf("Max error for far points with small distance to target : %g\n", err_max_far_small);
  printf("Max error for far points with large distance to target : %g\n", err_max_far_large);
  printf("Largest overshoot: %g\n", largest_overshoot);
  return 0;
}

int main(int, char **) { return test(1000000); }
