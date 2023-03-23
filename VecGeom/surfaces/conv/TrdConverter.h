#ifndef VECGEOM_SURFACE_TRDCONVERTER_H_
#define VECGEOM_SURFACE_TRDCONVERTER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/Trd.h>

namespace vgbrep {
namespace conv {

/// @brief Converter for Trd
/// @tparam Real_t Precision type
/// @param trd Trd solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateTrdSurfaces(vecgeom::UnplacedTrd const &trd, int logical_id)
{
  using WindowMask_t = WindowMask<Real_t>;
  using QuadMask_t   = QuadrilateralMask<Real_t>;

  LogicExpressionCPU logic; // AND logic: 0 & 1 & 2 & 3 & 4 & 5
  bool use_surf_safety = true;
  auto dx              = trd.dx1() - trd.dx2();
  auto dy              = trd.dy1() - trd.dy2();
  auto dzx             = vecgeom::Sqrt(4 * trd.dz() * trd.dz() + dy * dy) * 0.5;
  auto dzy             = vecgeom::Sqrt(4 * trd.dz() * trd.dz() + dx * dx) * 0.5;

  auto phix = ApproxEqual(dy, 0.) ? 90 : vecgeom::ATan(2 * trd.dz() / dy) * vecgeom::kRadToDeg;
  auto phiy = ApproxEqual(dx, 0.) ? 90 : vecgeom::ATan(2 * trd.dz() / dx) * vecgeom::kRadToDeg;
  if (phix < 0) phix = 180 + phix;
  if (phiy < 0) phiy = 180 + phiy;

  auto movey = (trd.dy1() + trd.dy2()) * 0.5;
  auto movex = (trd.dx1() + trd.dx2()) * 0.5;

  // Bottom face
  int isurf = builder::CreateLocalSurface<Real_t>(
      builder::CreateUnplacedSurface<Real_t>(kPlanar),
      builder::CreateFrame<Real_t>(kWindow, WindowMask_t{trd.dx1(), trd.dy1()}),
      builder::CreateLocalTransformation<Real_t>({0, 0, -trd.dz(), 0, 180, 0}), use_surf_safety);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(isurf);

  // Top face
  isurf = builder::CreateLocalSurface<Real_t>(builder::CreateUnplacedSurface<Real_t>(kPlanar),
                                              builder::CreateFrame<Real_t>(kWindow, WindowMask_t{trd.dx2(), trd.dy2()}),
                                              builder::CreateLocalTransformation<Real_t>({0, 0, trd.dz()}),
                                              use_surf_safety);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(land);
  logic.push_back(isurf);

  // Sides parallel to x axis
  if (vecgeom::Abs(dx) > vecgeom::kTolerance) {
    // At -dy
    isurf = builder::CreateLocalSurface<Real_t>(
        builder::CreateUnplacedSurface<Real_t>(kPlanar),
        builder::CreateFrame<Real_t>(kQuadrilateral,
                                     QuadMask_t{-trd.dx1(), -dzx, trd.dx1(), -dzx, trd.dx2(), dzx, -trd.dx2(), dzx}),
        builder::CreateLocalTransformation<Real_t>({0, -movey, 0, 0, phix, 0}), use_surf_safety);
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);

    // At +dy
    isurf = builder::CreateLocalSurface<Real_t>(
        builder::CreateUnplacedSurface<Real_t>(kPlanar),
        builder::CreateFrame<Real_t>(kQuadrilateral,
                                     QuadMask_t{-trd.dx1(), -dzx, trd.dx1(), -dzx, trd.dx2(), dzx, -trd.dx2(), dzx}),
        builder::CreateLocalTransformation<Real_t>({0, movey, 0, 180, phix, 0}), use_surf_safety);
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);
  } else { // We have rectangles.
    isurf = builder::CreateLocalSurface<Real_t>(builder::CreateUnplacedSurface<Real_t>(kPlanar),
                                                builder::CreateFrame<Real_t>(kWindow, WindowMask_t{trd.dx1(), dzx}),
                                                builder::CreateLocalTransformation<Real_t>({0, -movey, 0, 0, phix, 0}),
                                                use_surf_safety);
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);

    // At +dy
    isurf = builder::CreateLocalSurface<Real_t>(builder::CreateUnplacedSurface<Real_t>(kPlanar),
                                                builder::CreateFrame<Real_t>(kWindow, WindowMask_t{trd.dx1(), dzx}),
                                                builder::CreateLocalTransformation<Real_t>({0, movey, 0, 180, phix, 0}),
                                                use_surf_safety);
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);
  }
  // Sides parallel to y axis
  if (vecgeom::Abs(dy) > vecgeom::kTolerance) {
    // At -dx
    isurf = builder::CreateLocalSurface<Real_t>(
        builder::CreateUnplacedSurface<Real_t>(kPlanar),
        builder::CreateFrame<Real_t>(kQuadrilateral,
                                     QuadMask_t{-trd.dy1(), -dzy, trd.dy1(), -dzy, trd.dy2(), dzy, -trd.dy2(), dzy}),
        builder::CreateLocalTransformation<Real_t>({-movex, 0, 0, -90, phiy, 0}), use_surf_safety);
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);

    // At +dx
    isurf = builder::CreateLocalSurface<Real_t>(
        builder::CreateUnplacedSurface<Real_t>(kPlanar),
        builder::CreateFrame<Real_t>(kQuadrilateral,
                                     QuadMask_t{-trd.dy1(), -dzy, trd.dy1(), -dzy, trd.dy2(), dzy, -trd.dy2(), dzy}),
        builder::CreateLocalTransformation<Real_t>({movex, 0, 0, 90, phiy, 0}), use_surf_safety);
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);
  } else { // We have rectangles.
    // At -dx
    isurf = builder::CreateLocalSurface<Real_t>(
        builder::CreateUnplacedSurface<Real_t>(kPlanar),
        builder::CreateFrame<Real_t>(kWindow, WindowMask_t{trd.dy1(), dzy}),
        builder::CreateLocalTransformation<Real_t>({-movex, 0, 0, -90, phiy, 0}), use_surf_safety);
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);

    // At +dx
    isurf = builder::CreateLocalSurface<Real_t>(builder::CreateUnplacedSurface<Real_t>(kPlanar),
                                                builder::CreateFrame<Real_t>(kWindow, WindowMask_t{trd.dy1(), dzy}),
                                                builder::CreateLocalTransformation<Real_t>({movex, 0, 0, 90, phiy, 0}),
                                                use_surf_safety);
    builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
    logic.push_back(land);
    logic.push_back(isurf);
  }
  builder::AddLogicToShell<Real_t>(logical_id, logic);
  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
