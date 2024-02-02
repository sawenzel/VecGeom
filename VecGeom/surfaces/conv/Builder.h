#ifndef VECGEOM_SURFACE_BUILDER_H
#define VECGEOM_SURFACE_BUILDER_H

#include <VecGeom/surfaces/base/CpuTypes.h>

namespace vgbrep {

namespace builder {

template <typename Real_t>
UnplacedSurface CreateUnplacedSurface(SurfaceType type, Real_t *data = nullptr, bool flip = false)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  switch (type) {
  case kPlanar:
    return UnplacedSurface(type);
  case kCylindrical:
  case kSpherical:
    cpudata.fCylSphData.push_back({data[0], flip});
    return UnplacedSurface(type, cpudata.fCylSphData.size() - 1);
  case kConical:
    cpudata.fConeData.push_back({data[0], data[1], flip});
    return UnplacedSurface(type, cpudata.fConeData.size() - 1);
  case kTorus:
  case kGenSecondOrder:
    std::cout << "kTorus, kGenSecondOrder unhandled\n";
    return UnplacedSurface(type);
  };
  return UnplacedSurface(type);
}

// Creators for different types of frames.
template <typename Real_t>
Frame CreateFrame(FrameType type, WindowMask<Real_t> const &mask)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  int id        = cpudata.fWindowMasks.size();
  cpudata.fWindowMasks.push_back(mask);
  return Frame(type, id);
}

template <typename Real_t>
Frame CreateFrame(FrameType type, RingMask<Real_t> const &mask)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  int id        = cpudata.fRingMasks.size();
  cpudata.fRingMasks.push_back(mask);
  return Frame(type, id);
}

template <typename Real_t>
Frame CreateFrame(FrameType type, ZPhiMask<Real_t> const &mask)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  int id        = cpudata.fZPhiMasks.size();
  cpudata.fZPhiMasks.push_back(mask);
  return Frame(type, id);
}

template <typename Real_t>
Frame CreateFrame(FrameType type, QuadrilateralMask<Real_t> const &mask)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  int id        = cpudata.fQuadMasks.size();
  cpudata.fQuadMasks.push_back(mask);
  return Frame(type, id);
}

template <typename Real_t>
Frame CreateFrame(FrameType type, TriangleMask<Real_t> const &mask)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  int id        = cpudata.fTriangleMasks.size();
  cpudata.fTriangleMasks.push_back(mask);
  return Frame(type, id);
}

template <typename Real_t>
int CreateLocalTransformation(Transformation const &trans)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  int id        = cpudata.fLocalTrans.size();
  cpudata.fLocalTrans.push_back(trans);
  return id;
}

template <typename Real_t>
int CreateLocalSurface(UnplacedSurface const &unplaced, Frame const &frame, int trans, bool use_surf_safety)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  int id        = cpudata.fLocalSurfaces.size();
  cpudata.fLocalSurfaces.push_back({unplaced, frame, trans, use_surf_safety});
  return id;
}

template <typename Real_t>
int AddSurfaceToShell(int logical_id, int isurf)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  if (cpudata.fShells.size() == 0) {
    std::cout << "BrepHelper::AddSurfaceToShell: need to call SetNvolumes first\n";
    return -1;
  }
  assert(logical_id < (int)cpudata.fShells.size() && "surface shell id exceeding number of volumes");
  int id = cpudata.fShells[logical_id].fSurfaces.size();
  cpudata.fShells[logical_id].fSurfaces.push_back(isurf);
  cpudata.fSceneShells[logical_id].fSurfaces.push_back(isurf);
  cpudata.fLocalSurfaces[isurf].fSurfIndex = id;
  return id;
}

template <typename Real_t>
void AddLogicToShell(int logical_id, LogicExpressionCPU &logic)
{
  // Add solid logic to existing shell logic
  auto &cpudata  = CPUsurfData<Real_t>::Instance();
  auto &crtlogic = cpudata.fShells[logical_id].fLogic;
  crtlogic.insert(crtlogic.end(), logic.begin(), logic.end());
}

/// @brief Creates a quadrilateral frame, triangular frame of no frame based on a vector of vertices (using only XY coordinates)
/// @tparam Real_t Precision type
/// @tparam Container Container type
/// @param points Vector of points
/// @return Created frame. If this has the type kNoFrame, the user must abort framed surface creation
template <typename Real_t, typename Container>
Frame CreateFrameFromVertices(Container &points)
{
  if (points.size() == 3)
    return CreateFrame<Real_t>(kTriangle, TriangleMask<Real_t>{points[0].x(), points[0].y(), points[1].x(),
                                                               points[1].y(), points[2].x(), points[2].y()});
  if (points.size() == 4)
    return CreateFrame<Real_t>(kQuadrilateral,
                               QuadrilateralMask<Real_t>{points[0].x(), points[0].y(), points[1].x(), points[1].y(),
                                                         points[2].x(), points[2].y(), points[3].x(), points[3].y()});
  return Frame(kNoFrame);
}

/// @brief Compute transformation (rotation + translation) for a surface defined by a set of co-planar points.
/// @details The points must be ordered such that the cross product of any two consecutive segments has the same
/// direction as the normal. All points must be different and not all colinear.
/// @param points Input container holding the ordered Vector3D<Real_t> points, defined in the solid frame.
//  The container will return the points in the local surface reference frame
/// @return Transformation moving a surface from the (XOY) plane to the final position. The container will hold the
/// input points transformed with the inverse transformation, lying in the (XOY) plane
template <typename Real_t, typename Container>
vecgeom::Transformation3D TransformationFromPlanarPoints(Container &points)
{
  using Vector3 = vecgeom::Vector3D<Real_t>;

  int npoints = points.size();
  assert(npoints > 2 && "TransformationFromPlanarPoints takes at least three points");
  int istart            = 0;
  Real_t cross_mag2_max = 0.;
  Vector3 normal;
  Vector3 center;
  for (int i = 0; i < npoints; ++i) {
    center += points[i];
    auto a          = points[(i + 1) % npoints] - points[i];
    auto b          = points[(i + 2) % npoints] - points[(i + 1) % npoints];
    auto a_cross_b  = a.Cross(b);
    auto cross_mag2 = a_cross_b.Mag2();
    if (cross_mag2 > cross_mag2_max) {
      normal         = a_cross_b.Unit();
      istart         = i;
      cross_mag2_max = cross_mag2;
    }
  }
  assert(cross_mag2_max > vecgeom::kToleranceSquared && "TransformationFromPlanarPoints: degenerated polygon");
  center *= 1. / npoints;

  Vector3 zref = normal;
  Vector3 xref = (points[(istart + 1) % npoints] - points[istart]).Unit();
  Vector3 yref = zref.Cross(xref);
  vecgeom::Transformation3D transformation(center[0], center[1], center[2], xref[0], yref[0], zref[0], xref[1], yref[1],
                                           zref[1], xref[2], yref[2], zref[2]);
  // Convert points to the local frame
  for (int i = 0; i < npoints; ++i) {
    Vector3 local = transformation.Transform(points[i]);
    points[i]     = local;
  }
  return transformation;
}

/// @brief Create local surface starting from a vector of maximum four vertices
/// @tparam Real_t Precision type
/// @tparam Container Container type
/// @param points Vertices vector
/// @return Index of created local surface
template <typename Real_t, typename Container>
int CreateLocalSurfaceFromVertices(Container &points, int logical_id, bool use_surf_safety)
{
  // copy container because the content may get changed due to degenerated vertices
  Container vertices(points);
  // Remove duplicated vertices
  size_t i = 0;
  for (i = 0; i < vertices.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (ApproxEqualVector(vertices[i], vertices[j])) {
        // remove duplicate vertex
        vertices.erase(vertices.begin() + i--);
        break;
      }
    }
  }
  if (vertices.size() < 3) return -1;

  auto transformation = TransformationFromPlanarPoints<Real_t>(vertices);
  auto itrans         = CreateLocalTransformation<Real_t>(transformation);
  auto frame          = CreateFrameFromVertices<Real_t>(vertices);
  // Create transformation
  int isurf =
      builder::CreateLocalSurface<Real_t>(CreateUnplacedSurface<Real_t>(kPlanar), frame, itrans, use_surf_safety);
  AddSurfaceToShell<Real_t>(logical_id, isurf);
  return isurf;
}

} // namespace builder

} // namespace vgbrep

#endif
