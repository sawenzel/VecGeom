#ifndef VECGEOM_SURFACE_ELLIPTICALTUBECONVERTER_H_
#define VECGEOM_SURFACE_ELLIPTICALTUBECONVERTER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/EllipticalTube.h>

namespace vgbrep {
namespace conv {

/// @brief Converter for elliptical Tube
/// @tparam Real_t Precision type
/// @param tube Tube solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateEllipticalTubeSurfaces(vecgeom::UnplacedEllipticalTube const &tube, int logical_id,
                                  bool intersection = false)
{
  using ZPhiMask_t   = ZPhiMask<Real_t>;
  using WindowMask_t = WindowMask<Real_t>;

  LogicExpressionCPU logic; // top & bottom & [rmin] & rmax & (dphi < 180) ? sphi * ephi : sphi | ephi

  int isurf;
  vecgeom::Precision surfdata[3];

  auto &cpudata = CPUsurfData<Real_t>::Instance();

  // surface at +dz
  // due to the cut the top and bottom caps are elliptical, which is handled by using a rectangle frame
  // that fully contains the ellipse and making the surface logical such that the full boolean expression
  // must be evaluated to decide whether it is a hit or not.
  // As a consequence, all surfaces of this volume must be logical surfaces.
  isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
      builder::CreateFrame<Real_t>(FrameType::kWindow, WindowMask_t{tube.GetDx(), tube.GetDy()}),
      builder::CreateLocalTransformation<Real_t, vecgeom::Precision>({0, 0, tube.GetDz(), 0, 0, 0}));
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  // Make the surface "logical"
  assert(isurf >= 0);
  cpudata.fLocalSurfaces[isurf].fLogicId = isurf;
  logic.push_back(isurf);
  logic.push_back(land);

  // surface at -dz
  isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
      builder::CreateFrame<Real_t>(FrameType::kWindow, WindowMask_t{tube.GetDx(), tube.GetDy()}),
      builder::CreateLocalTransformation<Real_t, vecgeom::Precision>({0, 0, -tube.GetDz(), 0, 180, 0}));
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  // Make the surface "logical"
  assert(isurf >= 0);
  cpudata.fLocalSurfaces[isurf].fLogicId = isurf;
  logic.push_back(isurf);
  logic.push_back(land);

  // outer cylinder
  surfdata[0] = tube.GetDx();
  surfdata[1] = tube.GetDy();
  surfdata[2] = tube.GetDz();
  auto rmax   = vecCore::math::Max(tube.GetDx(), tube.GetDy());
  isurf       = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kElliptical, surfdata),
      builder::CreateFrame<Real_t>(FrameType::kZPhi, ZPhiMask_t{-surfdata[2], surfdata[2], /* full_circle=*/1, rmax,
                                                                rmax, static_cast<vecgeom::Precision>(0),
                                                                static_cast<vecgeom::Precision>(360)}),
      /*identity transformation*/ 0);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  // Make the surface "logical"
  assert(isurf >= 0);
  cpudata.fLocalSurfaces[isurf].fLogicId = isurf;
  logic.push_back(isurf);

  builder::AddLogicToShell<Real_t>(logical_id, logic);

  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
