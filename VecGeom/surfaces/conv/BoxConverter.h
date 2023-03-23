#ifndef VECGEOM_SURFACE_BOXCONVERTER_H_
#define VECGEOM_SURFACE_BOXCONVERTER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

#include <VecGeom/volumes/Box.h>

namespace vgbrep {
namespace conv {

/// @brief Converter for Box
/// @tparam Real_t Precision type
/// @param box Box solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateBoxSurfaces(vecgeom::UnplacedBox const &box, int logical_id)
{
  using WindowMask_t         = WindowMask<Real_t>;
  const bool use_surf_safety = true;
  int isurf;
  LogicExpressionCPU logic; // AND logic: 0 & 1 & 2 & 3 & 4 & 5
  // surface at -dx:
  isurf = builder::CreateLocalSurface<Real_t>(builder::CreateUnplacedSurface<Real_t>(kPlanar),
                                              builder::CreateFrame<Real_t>(kWindow, WindowMask_t{box.y(), box.z()}),
                                              builder::CreateLocalTransformation<Real_t>({-box.x(), 0, 0, -90, 90, 0}),
                                              use_surf_safety);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(isurf);
  // surface at +dx:
  isurf = builder::CreateLocalSurface<Real_t>(builder::CreateUnplacedSurface<Real_t>(kPlanar),
                                              builder::CreateFrame<Real_t>(kWindow, WindowMask_t{box.y(), box.z()}),
                                              builder::CreateLocalTransformation<Real_t>({box.x(), 0, 0, 90, 90, 0}),
                                              use_surf_safety);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(land);
  logic.push_back(isurf);
  // surface at -dy:
  isurf = builder::CreateLocalSurface<Real_t>(builder::CreateUnplacedSurface<Real_t>(kPlanar),
                                              builder::CreateFrame<Real_t>(kWindow, WindowMask_t{box.x(), box.z()}),
                                              builder::CreateLocalTransformation<Real_t>({0, -box.y(), 0, 0, 90, 0}),
                                              use_surf_safety);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(land);
  logic.push_back(isurf);
  // surface at +dy:
  isurf = builder::CreateLocalSurface<Real_t>(builder::CreateUnplacedSurface<Real_t>(kPlanar),
                                              builder::CreateFrame<Real_t>(kWindow, WindowMask_t{box.x(), box.z()}),
                                              builder::CreateLocalTransformation<Real_t>({0, box.y(), 0, 0, -90, 0}),
                                              use_surf_safety);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(land);
  logic.push_back(isurf);
  // surface at -dz:
  isurf = builder::CreateLocalSurface<Real_t>(builder::CreateUnplacedSurface<Real_t>(kPlanar),
                                              builder::CreateFrame<Real_t>(kWindow, WindowMask_t{box.x(), box.y()}),
                                              builder::CreateLocalTransformation<Real_t>({0, 0, -box.z(), 0, 180, 0}),
                                              use_surf_safety);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(land);
  logic.push_back(isurf);
  // surface at +dz:
  isurf = builder::CreateLocalSurface<Real_t>(builder::CreateUnplacedSurface<Real_t>(kPlanar),
                                              builder::CreateFrame<Real_t>(kWindow, WindowMask_t{box.x(), box.y()}),
                                              builder::CreateLocalTransformation<Real_t>({0, 0, box.z(), 0, 0, 0}),
                                              use_surf_safety);
  builder::AddSurfaceToShell<Real_t>(logical_id, isurf);
  logic.push_back(land);
  logic.push_back(isurf);
  builder::AddLogicToShell<Real_t>(logical_id, logic);
  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
