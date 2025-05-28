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
bool CreateTrdSurfaces(vecgeom::UnplacedTrd const &trd, int logical_id, bool intersection = false)
{
  using Vector3 = vecgeom::Vector3D<vecgeom::Precision>;
  int isurf;
  LogicExpressionCPU logic; // AND logic: 0 & 1 & 2 & 3 & 4 & 5

  auto dx1 = trd.dx1();
  auto dx2 = trd.dx2();
  auto dy1 = trd.dy1();
  auto dy2 = trd.dy2();
  auto dz  = trd.dz();

  // corners represented as vectors (-x, -y, -z), (+x, -y, -z), (+x, +y, -z), (-x, +y, -z), same for +z
  std::vector<Vector3> corners = {{-dx1, -dy1, -dz}, {dx1, -dy1, -dz}, {dx1, dy1, -dz}, {-dx1, dy1, -dz},
                                  {-dx2, -dy2, dz},  {dx2, -dy2, dz},  {dx2, dy2, dz},  {-dx2, dy2, dz}};
  auto assertWindow            = [](int isurf) {
    VECGEOM_ASSERT(isurf >= 0 &&
                              CPUsurfData<Real_t>::Instance().fLocalSurfaces[isurf].fFrame.type == FrameType::kWindow);
  };
  std::vector<Vector3> vert;
  // surface at -dx:
  vert  = {corners[3], corners[0], corners[4], corners[7]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  logic.push_back(isurf);
  // surface at +dx:
  vert  = {corners[1], corners[2], corners[6], corners[5]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  logic.push_back(land);
  logic.push_back(isurf);
  // surface at -dy:
  vert  = {corners[0], corners[1], corners[5], corners[4]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  logic.push_back(land);
  logic.push_back(isurf);
  // surface at +dy:
  vert  = {corners[2], corners[3], corners[7], corners[6]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  logic.push_back(land);
  logic.push_back(isurf);
  // surface at -dz:
  vert  = {corners[0], corners[3], corners[2], corners[1]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  assertWindow(isurf);
  logic.push_back(land);
  logic.push_back(isurf);
  // surface at +dz:
  vert  = {corners[4], corners[5], corners[6], corners[7]};
  isurf = builder::CreateLocalSurfaceFromVertices<Real_t>(vert, logical_id);
  if (intersection) builder::GetSurface<Real_t>(isurf).fSkipConvexity = true;
  assertWindow(isurf);
  logic.push_back(land);
  logic.push_back(isurf);
  builder::AddLogicToShell<Real_t>(logical_id, logic);
  return true;
}

} // namespace conv
} // namespace vgbrep
#endif
