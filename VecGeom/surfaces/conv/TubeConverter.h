#ifndef VECGEOM_SURFACE_TUBECONVERTER_H_
#define VECGEOM_SURFACE_TUBECONVERTER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/Tube.h>

namespace vgbrep {
namespace conv {

/// @brief Converter for Tube
/// @tparam Real_t Precision type
/// @param tube Tube solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateTubeSurfaces(vecgeom::UnplacedTube const &tube, int logical_id)
{
  using RingMask_t   = RingMask<Real_t>;
  using ZPhiMask_t   = ZPhiMask<Real_t>;
  using WindowMask_t = WindowMask<Real_t>;

  LogicExpressionCPU logic; // top & bottom & [rmin] & rmax & (dphi < 180) ? sphi * ephi : sphi | ephi
  auto sphi = tube.sphi();
  auto dphi = tube.dphi();
  auto ephi = tube.sphi() + tube.dphi();

  assert(dphi > vecgeom::kTolerance);

  auto Rmean = (tube.rmin() + tube.rmax()) / 2;
  auto Rdiff = (tube.rmax() - tube.rmin()) / 2;

  assert(Rdiff > 0);

  bool fullCirc  = ApproxEqual(dphi, vecgeom::kTwoPi);
  bool smallerPi = dphi < (vecgeom::kPi - vecgeom::kTolerance);

  int isurf;
  vecgeom::Precision surfdata[2];

  // We need angles in degrees for transformations
  auto sphid = vecgeom::kRadToDeg * sphi;
  auto ephid = vecgeom::kRadToDeg * ephi;

  // surface at +dz
  isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
      builder::CreateFrame<Real_t>(FrameType::kRing, RingMask_t{tube.rmin(), tube.rmax(), fullCirc, sphi, ephi}),
      builder::CreateLocalTransformation<Real_t, vecgeom::Precision>({0, 0, tube.z(), 0, 0, 0}));
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(isurf);
  // surface at -dz
  isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
      builder::CreateFrame<Real_t>(FrameType::kRing, RingMask_t{tube.rmin(), tube.rmax(), fullCirc, sphi, ephi}),
      builder::CreateLocalTransformation<Real_t, vecgeom::Precision>({0, 0, -tube.z(), 0, 180, -sphid - ephid}));
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(land);
  logic.push_back(isurf);
  // inner cylinder
  if (tube.rmin() > vecgeom::kTolerance) {
    surfdata[0] = tube.rmin();
    isurf       = builder::CreateLocalSurface<Real_t>(
        builder::CreateUnplacedSurface<Real_t>(SurfaceType::kCylindrical, surfdata, /*flipped=*/true),
        builder::CreateFrame<Real_t>(FrameType::kZPhi,
                                     ZPhiMask_t{-tube.z(), tube.z(), fullCirc, tube.rmin(), tube.rmin(), sphi, ephi}),
        /*identity transformation*/ 0);
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);
  }
  // outer cylinder
  surfdata[0] = tube.rmax();
  isurf       = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kCylindrical, surfdata),
      builder::CreateFrame<Real_t>(FrameType::kZPhi,
                                   ZPhiMask_t{-tube.z(), tube.z(), fullCirc, tube.rmax(), tube.rmax(), sphi, ephi}),
      /*identity transformation*/ 0);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(land);
  logic.push_back(isurf);

  if (ApproxEqual(dphi, vecgeom::kTwoPi)) {
    builder::AddLogicToShell<Real_t>(logical_id, logic);
    return true;
  }
  // plane cap at Sphi
  isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
      builder::CreateFrame<Real_t>(FrameType::kWindow, WindowMask_t{Rdiff, tube.z()}),
      builder::CreateLocalTransformation<Real_t, vecgeom::Precision>(
          {Rmean * std::cos(sphi), Rmean * std::sin(sphi), 0, sphid, 90, 0}));
  if (!smallerPi) builder::GetSurface<Real_t>(isurf).fEmbedding = false;
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(land);
  logic.push_back(lplus); // '('
  logic.push_back(isurf);

  // plane cap at Sphi+Dphi
  isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar),
      builder::CreateFrame<Real_t>(FrameType::kWindow, WindowMask_t{Rdiff, tube.z()}),
      builder::CreateLocalTransformation<Real_t, vecgeom::Precision>(
          {Rmean * std::cos(ephi), Rmean * std::sin(ephi), 0, ephid, -90, 0}));
  if (!smallerPi) builder::GetSurface<Real_t>(isurf).fEmbedding = false;
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
