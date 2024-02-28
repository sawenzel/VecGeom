#ifndef VECGEOM_SURFACE_TORUSCONVERTER_H_
#define VECGEOM_SURFACE_TORUSCONVERTER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/Torus2.h>

namespace vgbrep {
namespace conv {

/// @brief Converter for Torus
/// @tparam Real_t Precision type
/// @param torus Torus solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateTorusSurfaces(vecgeom::UnplacedTorus2 const &torus, int logical_id)
{
  using RingMask_t = RingMask<Real_t>;

  LogicExpressionCPU logic; // top & bottom & [rmin] & rmax & (dphi < 180) ? sphi * ephi : sphi | ephi
  auto rmin = torus.rmin();
  auto rmax = torus.rmax();
  auto rtor = torus.rtor();
  auto sphi = torus.sphi();
  auto dphi = torus.dphi();
  auto ephi = dphi + sphi;

  assert(rtor - rmin > -vecgeom::kTolerance);

  bool fullCirc        = ApproxEqual(dphi, vecgeom::kTwoPi);
  bool smallerPi       = dphi < (vecgeom::kPi - vecgeom::kTolerance);
  bool use_surf_safety = true;

  int isurf;
  Real_t surfdata[4];

  // We need angles in degrees for transformations
  auto sphid = vecgeom::kRadToDeg * sphi;
  auto ephid = vecgeom::kRadToDeg * ephi;

  // inner torus
  // Note that the torus surface needs to be fully checked already in the surface check.
  // Therefore, it does not need a frame and the frames should not be checked, hence the `never_check` flag.
  surfdata[0] = rtor;
  surfdata[1] = rmin;
  surfdata[2] = sphi;
  surfdata[3] = ephi;
  isurf       = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kTorus, surfdata, /*flipped=*/true),
      builder::CreateFrame<Real_t>(FrameType::kRing, RingMask_t{rtor - rmin, rtor + rmin, fullCirc, sphi, ephi}),
      /*identity transformation*/ 0, use_surf_safety, /*never_check=*/true);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(isurf);

  // outer torus
  surfdata[0] = rtor;
  surfdata[1] = rmax;
  surfdata[2] = sphi;
  surfdata[3] = ephi;
  isurf       = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kTorus, surfdata),
      builder::CreateFrame<Real_t>(FrameType::kRing, RingMask_t{rtor - rmax, rtor + rmax, fullCirc, sphi, ephi}),
      /*identity transformation*/ 0, use_surf_safety, /*never_check=*/true);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(land);
  logic.push_back(isurf);

  if (ApproxEqual(dphi, vecgeom::kTwoPi)) {
    builder::AddLogicToShell<Real_t>(logical_id, logic);
    return true;
  }

  // ring cap at Sphi
  isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
      builder::CreateFrame<Real_t>(FrameType::kRing, RingMask_t{rmin, rmax, /*fullcircle=*/true, 0, 360}),
      builder::CreateLocalTransformation<Real_t>({rtor * std::cos(sphi), rtor * std::sin(sphi), 0, sphid, 90, 0}),
      use_surf_safety && smallerPi);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(land);
  logic.push_back(lplus); // '('
  logic.push_back(isurf);

  // ring cap at Sphi+Dphi
  isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
      builder::CreateFrame<Real_t>(FrameType::kRing, RingMask_t{rmin, rmax, /*fullcircle=*/true, 0, 360}),
      builder::CreateLocalTransformation<Real_t>({rtor * std::cos(ephi), rtor * std::sin(ephi), 0, ephid, -90, 0}),
      use_surf_safety && smallerPi);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(smallerPi ? land : lor);
  logic.push_back(isurf);
  logic.push_back(lminus); // ')'
  builder::AddLogicToShell<Real_t>(logical_id, logic);
  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
