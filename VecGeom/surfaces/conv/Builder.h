#ifndef VECGEOM_SURFACE_BUILDER_H
#define VECGEOM_SURFACE_BUILDER_H

#include <VecGeom/surfaces/base/CpuTypes.h>

#include <initializer_list>

namespace vgbrep {

namespace builder {

template <typename Real_t>
UnplacedSurface<Real_t> CreateUnplacedSurface(SurfaceType type, vecgeom::Precision *data = nullptr, bool flip = false)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  switch (type) {
  case SurfaceType::kPlanar:
    return UnplacedSurface<Real_t>(type);
  case SurfaceType::kCylindrical:
  case SurfaceType::kSpherical:
    return UnplacedSurface<Real_t>(type, -1, data[0], 0., flip);
  case SurfaceType::kElliptical:
    cpudata.fEllipData.push_back({data[0], data[1], data[2]});
    return UnplacedSurface<Real_t>(type, cpudata.fEllipData.size() - 1);
  case SurfaceType::kConical:
    return UnplacedSurface<Real_t>(type, -1, data[0], data[1], flip);
  case SurfaceType::kTorus:
    cpudata.fTorusData.push_back({data[0], data[1], data[2], data[3], flip});
    return UnplacedSurface<Real_t>(type, cpudata.fTorusData.size() - 1);
  case SurfaceType::kArb4:
    cpudata.fArb4Data.push_back(
        {data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]});
    return UnplacedSurface<Real_t>(type, cpudata.fArb4Data.size() - 1);
  default:
    return UnplacedSurface<Real_t>(type);
  };
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
int CreateLocalSurface(UnplacedSurface<Real_t> const &unplaced, Frame const &frame, TransformationMP<Real_t> trans,
                       bool never_check = false)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  int id        = cpudata.fLocalSurfaces.size();
  cpudata.fLocalSurfaces.push_back({unplaced, frame, trans, /*NavIndex=*/0, never_check});
  return id;
}

template <typename Real_t>
FramedSurface<Real_t, TransformationMP<Real_t>> &GetSurface(int isurf)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  VECGEOM_ASSERT(size_t(isurf) < cpudata.fLocalSurfaces.size());
  return cpudata.fLocalSurfaces[isurf];
}

template <typename Real_t>
int AddSurfaceToShell(int logical_id, int isurf)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  if (cpudata.fShells.size() == 0) {
    std::cout << "BrepHelper::AddSurfaceToShell: need to call SetNvolumes first\n";
    return -1;
  }
  VECGEOM_VALIDATE(logical_id < (int)cpudata.fShells.size(), << "surface shell id exceeding number of volumes");
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
Frame CreateFrameFromVertices(Container &points, TransformationMP<Real_t> &trans)
{
  if (points.size() == 3)
    return CreateFrame<Real_t>(FrameType::kTriangle, TriangleMask<Real_t>{points[0].x(), points[0].y(), points[1].x(),
                                                                          points[1].y(), points[2].x(), points[2].y()});
  if (points.size() == 4) {
    // Check for rectangular frame
    // The 0->1 vector is aligned with the local Ox
    bool rectangle = ApproxEqualVector(points[1] - points[0], points[2] - points[3]);
    rectangle &=
        ApproxEqual(static_cast<Real_t>((points[1] - points[0]).Dot(points[3] - points[0])), static_cast<Real_t>(0.));
    if (rectangle) {
      auto dx = 0.5 * (points[1] - points[0]).Mag();
      auto dy = 0.5 * (points[3] - points[0]).Mag();
      return CreateFrame<Real_t>(FrameType::kWindow, WindowMask<Real_t>{dx, dy});
    } else {
      return CreateFrame<Real_t>(FrameType::kQuadrilateral,
                                 QuadrilateralMask<Real_t>{points[0].x(), points[0].y(), points[1].x(), points[1].y(),
                                                           points[2].x(), points[2].y(), points[3].x(), points[3].y()});
    }
  }
  return Frame(FrameType::kNoFrame);
}

/// @brief Compute transformation (rotation + translation) for a surface defined by a set of co-planar points.
/// @details The points must be ordered such that the cross product of any two consecutive segments has the same
/// direction as the normal. All points must be different and not all colinear.
/// @param points Input container holding the ordered Vector3D<Real_t> points, defined in the solid frame.
//  The container will return the points in the local surface reference frame
/// @return Transformation moving a surface from the (XOY) plane to the final position. The container will hold the
/// input points transformed with the inverse transformation, lying in the (XOY) plane
template <typename Real_t, typename Container>
vecgeom::Transformation3DMP<Real_t> TransformationFromPlanarPoints(Container &points)
{
  using Vector3 = vecgeom::Vector3D<Real_t>;

  int npoints = points.size();
  VECGEOM_VALIDATE(npoints > 2, << "TransformationFromPlanarPoints takes at least three points");
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
    if (cross_mag2 > cross_mag2_max + cross_mag2_max * vecgeom::kToleranceDistSquared<Real_t>) {
      normal         = a_cross_b.Unit();
      istart         = i;
      cross_mag2_max = cross_mag2;
    }
  }
  VECGEOM_ASSERT(cross_mag2_max > vecgeom::kToleranceDistSquared<Real_t> &&
                 "TransformationFromPlanarPoints: degenerated polygon");
  center *= 1. / npoints;

  Vector3 zref = normal;
  Vector3 xref = (points[(istart + 1) % npoints] - points[istart]).Unit();
  Vector3 yref = zref.Cross(xref);
  vecgeom::Transformation3DMP<Real_t> transformation(center[0], center[1], center[2], xref[0], yref[0], zref[0],
                                                     xref[1], yref[1], zref[1], xref[2], yref[2], zref[2]);
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
int CreateLocalSurfaceFromVertices(Container &points, int logical_id)
{
  // helper function to find non-coplanar surfaces (Arb4)
  using Vector3      = vecgeom::Vector3D<Real_t>;
  auto PointsOnPlane = [](Container &points) {
    // this method of checking the co-planarity was taken from the UnplacedTrapezoid
    Vector3 normalVector = (points[2] - points[0]).Cross(points[3] - points[1]);
    normalVector.Normalize();
    Vector3 centr = 0.25 * (points[0] + points[1] + points[2] + points[3]);

    Real_t dist_scale = Max(Max((points[0] - centr).Length(), (points[1] - centr).Length()),
                            Max((points[2] - centr).Length(), (points[3] - centr).Length()));
    // check co-planarity
    Real_t resid1 = normalVector.Dot(points[0] - centr);
    Real_t resid2 = normalVector.Dot(points[1] - centr);
    Real_t resid3 = normalVector.Dot(points[2] - centr);
    Real_t resid4 = normalVector.Dot(points[3] - centr);
    Real_t resid  = Max(Max(fabs(resid1), fabs(resid2)), Max(fabs(resid3), fabs(resid4)));

    // the residue should be small compared to the length scale of the quadrilateral
    return resid / dist_scale < 100 * vecgeom::kTolerance;
  };

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

  int isurf = 0;
  // Create transformation
  auto transformation = TransformationFromPlanarPoints<Real_t>(vertices);
  auto frame          = CreateFrameFromVertices<Real_t>(vertices, transformation);
  if (vertices.size() != 4 || PointsOnPlane(vertices)) {
    isurf =
        builder::CreateLocalSurface<Real_t>(CreateUnplacedSurface<Real_t>(SurfaceType::kPlanar), frame, transformation);
  } else { // creating Arb4 surface
    vecgeom::Precision surfdata[10];
    surfdata[0] = points[0].x();
    surfdata[1] = points[0].y();
    surfdata[2] = points[0].z();
    surfdata[3] = points[1].x();
    surfdata[4] = points[1].y();
    surfdata[5] = points[2].x();
    surfdata[6] = points[2].y();
    surfdata[7] = points[2].z();
    surfdata[8] = points[3].x();
    surfdata[9] = points[3].y();
    // we are using the frame above although it is never used for the Arb4
    vecgeom::Transformation3DMP<Real_t> identity;
    isurf = builder::CreateLocalSurface<Real_t>(CreateUnplacedSurface<Real_t>(SurfaceType::kArb4, surfdata), frame,
                                                /*identity transformation*/ identity, /*never_check=*/1);
  }
  AddSurfaceToShell<Real_t>(logical_id, isurf);
  return isurf;
}

} // namespace builder

} // namespace vgbrep

#endif
