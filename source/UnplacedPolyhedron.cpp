/// \file UnplacedPolyhedron.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/UnplacedPolyhedron.h"
#include "VecGeom/volumes/PlacedPolyhedron.h"
#include "VecGeom/volumes/SpecializedPolyhedron.h"
#include "VecGeom/volumes/utilities/GenerationUtilities.h"
#include "VecGeom/management/VolumeFactory.h"

#include <cmath>
#include <memory>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

using namespace vecgeom::Polyhedron;

UnplacedPolyhedron::UnplacedPolyhedron(const int sideCount, const int zPlaneCount, Precision const zPlanes[],
                                       Precision const rMin[], Precision const rMax[])
    : UnplacedPolyhedron(0., kTwoPi, sideCount, zPlaneCount, zPlanes, rMin, rMax)
{
  DetectConvexity();
  ComputeBBox();
}

VECCORE_ATT_HOST_DEVICE
UnplacedPolyhedron::UnplacedPolyhedron(Precision phiStart, Precision phiDelta, const int sideCount,
                                       const int zPlaneCount, Precision const zPlanes[], Precision const rMin[],
                                       Precision const rMax[])
{
  fPoly = new PolyhedronStruct<Precision>(phiStart, phiDelta, sideCount, zPlaneCount, zPlanes, rMin, rMax);
  DetectConvexity();
  ComputeBBox();
}

VECCORE_ATT_HOST_DEVICE
UnplacedPolyhedron::UnplacedPolyhedron(Precision phiStart, Precision phiDelta, const int sideCount,
                                       const int zPlaneCount, Precision const zPlanes[], Precision const rMin[],
                                       Precision const rMax[], size_t const buffer_size, void const *buffer)
{

  fAllocated = (buffer == nullptr);
  fBuffer    = fAllocated ? new char[buffer_size] : buffer;
  AlignedAllocator a((void *)fBuffer, buffer_size);
  fPoly = a.aligned_alloc<PolyhedronStruct<Precision>>(1, kAlignmentBoundary, phiStart, phiDelta, sideCount,
                                                       zPlaneCount, zPlanes, rMin, rMax, a);
  DetectConvexity();
  ComputeBBox();
}

UnplacedPolyhedron::UnplacedPolyhedron(Precision phiStart, Precision phiDelta, const int sideCount,
                                       const int verticesCount,
                                       Precision const r[], // 2*zPlaneCount elements
                                       Precision const z[]  // ditto
)
{
  fPoly = new PolyhedronStruct<Precision>(phiStart, phiDelta, sideCount, verticesCount, r, z);
  DetectConvexity();
  ComputeBBox();
}

VECCORE_ATT_HOST_DEVICE
int UnplacedPolyhedron::GetNQuadrilaterals() const
{
  int count = 0;
  for (int i = 0; i < GetZSegmentCount(); ++i) {
    // outer
    count += GetZSegment(i).outer.size();
    // inner
    count += GetZSegment(i).inner.size();
    // phi
    count += GetZSegment(i).phi.size();
  }
  return count;
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedPolyhedron::Create(LogicalVolume const *const logical_volume,
                                          Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                          const int id, const int copy_no, const int child_id,
#endif
                                          VPlacedVolume *const placement)
{
  UnplacedPolyhedron const *unplaced = static_cast<UnplacedPolyhedron const *>(logical_volume->GetUnplacedVolume());

  EInnerRadii innerRadii = unplaced->HasInnerRadii() ? EInnerRadii::kTrue : EInnerRadii::kFalse;

  EPhiCutout phiCutout = unplaced->HasPhiCutout()
                             ? (unplaced->HasLargePhiCutout() ? EPhiCutout::kLarge : EPhiCutout::kTrue)
                             : EPhiCutout::kFalse;

#ifndef VECCORE_CUDA
// for the moment we do not propagate placement specialization
// (We should in the future select a few important specializations here such as rotation or no rotation)
#define POLYHEDRON_CREATE_SPECIALIZATION(INNER, PHI) \
  return CreateSpecializedWithPlacement<SpecializedPolyhedron<INNER, PHI>>(logical_volume, transformation, placement)
#else
#define POLYHEDRON_CREATE_SPECIALIZATION(INNER, PHI)                                                           \
  return CreateSpecializedWithPlacement<SpecializedPolyhedron<INNER, PHI>>(logical_volume, transformation, id, \
                                                                           copy_no, child_id, placement)
#endif

  if (innerRadii == EInnerRadii::kTrue) {
    if (phiCutout == EPhiCutout::kFalse) POLYHEDRON_CREATE_SPECIALIZATION(EInnerRadii::kTrue, EPhiCutout::kFalse);
    if (phiCutout == EPhiCutout::kTrue) POLYHEDRON_CREATE_SPECIALIZATION(EInnerRadii::kTrue, EPhiCutout::kTrue);
    if (phiCutout == EPhiCutout::kLarge) POLYHEDRON_CREATE_SPECIALIZATION(EInnerRadii::kTrue, EPhiCutout::kLarge);

  } else {
    if (phiCutout == EPhiCutout::kFalse) POLYHEDRON_CREATE_SPECIALIZATION(EInnerRadii::kFalse, EPhiCutout::kFalse);
    if (phiCutout == EPhiCutout::kTrue) POLYHEDRON_CREATE_SPECIALIZATION(EInnerRadii::kFalse, EPhiCutout::kTrue);
    if (phiCutout == EPhiCutout::kLarge) POLYHEDRON_CREATE_SPECIALIZATION(EInnerRadii::kFalse, EPhiCutout::kLarge);
  }

  // Return value in case of NO_SPECIALIZATION
  if (placement) {
    new (placement)
        SpecializedPolyhedron<Polyhedron::EInnerRadii::kGeneric,
#ifdef VECCORE_CUDA
                              Polyhedron::EPhiCutout::kGeneric>(logical_volume, transformation, id, copy_no, child_id);
#else
                              Polyhedron::EPhiCutout::kGeneric>(logical_volume, transformation);
#endif
    return placement;
  }

  return new SpecializedPolyhedron<Polyhedron::EInnerRadii::kGeneric,
#ifdef VECCORE_CUDA
                                   Polyhedron::EPhiCutout::kGeneric>(logical_volume, transformation, id, copy_no,
                                                                     child_id);
#else
                                   Polyhedron::EPhiCutout::kGeneric>(logical_volume, transformation);
#endif

#undef POLYHEDRON_CREATE_SPECIALIZATION
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedPolyhedron::SpecializedVolume(LogicalVolume const *const volume,
                                                     Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                                     const int id, const int copy_no, const int child_id,
#endif
                                                     VPlacedVolume *const placement) const
{

  return VolumeFactory::CreateByTransformation<UnplacedPolyhedron>(volume, transformation,
#ifdef VECCORE_CUDA
                                                                   id, copy_no, child_id,
#endif
                                                                   placement);
}

VECCORE_ATT_HOST_DEVICE
Precision UnplacedPolyhedron::GetTriangleArea(Vector3D<Precision> const &v1, Vector3D<Precision> const &v2,
                                              Vector3D<Precision> const &v3) const
{
  Vector3D<Precision> vec1 = v1 - v2;
  Vector3D<Precision> vec2 = v1 - v3;
  return 0.5 * (vec1.Cross(vec2)).Mag();
}

// TODO: this functions seems to be neglecting the phi cut !!
Precision UnplacedPolyhedron::Capacity() const
{
  if (fPoly->fCapacity == 0.) {
    // Formula for section : V=h(f+F+sqrt(f*F))/3;
    // Fand f-areas of surfaces on +/-dz
    // h-heigh

    // a helper lambda for the volume calculation ( per quadrilaterial )
    auto VolumeHelperFunc = [&](Vector3D<Precision> const &a, Vector3D<Precision> const &b,
                                Vector3D<Precision> const &c, Vector3D<Precision> const &d) {
      Precision dz         = std::fabs(a.z() - c.z());
      Precision bottomArea = GetTriangleArea(a, b, Vector3D<Precision>(0., 0., a.z()));
      Precision topArea    = GetTriangleArea(c, d, Vector3D<Precision>(0., 0., c.z()));
      return dz * (bottomArea + topArea + std::sqrt(topArea * bottomArea));
    };

    for (int j = 0; j < GetZSegmentCount(); ++j) {
      // need to protect against empty segments because it could be
      // that the polyhedron makes a jump at this segment count
      if (GetZSegment(j).outer.size() > 0) {
        auto outercorners     = GetZSegment(j).outer.GetCorners();
        Vector3D<Precision> a = outercorners[0][0];
        Vector3D<Precision> b = outercorners[1][0];
        Vector3D<Precision> c = outercorners[2][0];
        Vector3D<Precision> d = outercorners[3][0];
        Precision volume      = VolumeHelperFunc(a, b, c, d); // outer volume

        if (GetZSegment(j).inner.size() > 0) {
          auto innercorners = GetZSegment(j).inner.GetCorners();

          a = innercorners[0][0];
          b = innercorners[1][0];
          c = innercorners[2][0];
          d = innercorners[3][0];
          volume -= VolumeHelperFunc(a, b, c, d); // subtract inner volume
        }
        fPoly->fCapacity += volume;
      }
    }
    fPoly->fCapacity *= GetSideCount() * (1. / 3.);
  }
  return fPoly->fCapacity;
}

// VECCORE_ATT_HOST_DEVICE
Precision UnplacedPolyhedron::SurfaceArea() const
{
  if (!fPoly->fAreaStruct) {
    signed int j;
    Precision totArea = 0., area, aTop = 0., aBottom = 0.;

    fPoly->fAreaStruct = new PolyhedronStruct<Precision>::AreaStruct(GetZSegmentCount());

    // Below we generate the areas relevant to our solid
    // We are starting with ZSegments(lateral parts)

    for (j = 0; j < GetZSegmentCount(); ++j) {
      fPoly->fAreaStruct->outer[j] = 0.;
      fPoly->fAreaStruct->inner[j] = 0.;
      fPoly->fAreaStruct->phi[j]   = 0.;

      if (GetZSegment(j).outer.size() > 0) {
        area                         = GetZSegment(j).outer.GetQuadrilateralArea(0) * GetSideCount();
        fPoly->fAreaStruct->outer[j] = area;
        totArea += area;
      }

      if (GetZSegment(j).inner.size() > 0) {
        area                         = GetZSegment(j).inner.GetQuadrilateralArea(0) * GetSideCount();
        fPoly->fAreaStruct->inner[j] = area;
        totArea += area;
      }

      if (HasPhiCutout() && GetZSegment(j).phi.size() > 0) {
        area                       = GetZSegment(j).phi.GetQuadrilateralArea(0) * 2.0;
        fPoly->fAreaStruct->phi[j] = area;
        totArea += area;
      }
    }

    // Must include top and bottom areas
    //

    Vector3D<Precision> point1 = GetZSegment(0).outer.GetCorners()[0][0];
    Vector3D<Precision> point2 = GetZSegment(0).outer.GetCorners()[1][0];
    Vector3D<Precision> point3, point4;
    if (GetZSegment(0).inner.size() > 0) {
      point3 = GetZSegment(0).inner.GetCorners()[0][0];
      point4 = GetZSegment(0).inner.GetCorners()[1][0];
      aTop   = GetSideCount() * (GetTriangleArea(point1, point2, point3) + GetTriangleArea(point3, point4, point2));

    } else {
      point3.Set(0.0, 0.0, GetZSegment(0).outer.GetCorners()[0][0].z());
      aTop = GetSideCount() * (GetTriangleArea(point1, point2, point3));
    }

    fPoly->fAreaStruct->top_area = aTop;
    totArea += aTop;

    point1 = GetZSegment(GetZSegmentCount() - 1).outer.GetCorners()[2][0];
    point2 = GetZSegment(GetZSegmentCount() - 1).outer.GetCorners()[3][0];

    if (GetZSegment(GetZSegmentCount() - 1).inner.size() > 0) {
      point3  = GetZSegment(GetZSegmentCount() - 1).inner.GetCorners()[2][0];
      point4  = GetZSegment(GetZSegmentCount() - 1).inner.GetCorners()[3][0];
      aBottom = GetSideCount() * (GetTriangleArea(point1, point2, point3) + GetTriangleArea(point3, point4, point2));
    } else {
      point3.Set(0.0, 0.0, GetZSegment(GetZSegmentCount() - 1).outer.GetCorners()[2][0].z());
      aBottom = GetSideCount() * GetTriangleArea(point1, point2, point3);
    }

    fPoly->fAreaStruct->bottom_area = aBottom;
    totArea += aBottom;
    fPoly->fAreaStruct->area = totArea;
  }
  return fPoly->fAreaStruct->area;
}

#ifndef VECCORE_CUDA
void UnplacedPolyhedron::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  aMin               = kInfLength;
  aMax               = -kInfLength;
  Precision phiStart = fPoly->fPhiStart;
  Precision phiDelta = fPoly->fPhiDelta;
  Precision sidePhi  = phiDelta / fPoly->fSideCount;
  // Specified radii are to the sides, not to the corners. Change these values,
  // as corners and not sides are used to compute the extent
  Precision conv = 1. / cos(0.5 * sidePhi);
  Vector3D<Precision> crt;
  // Loop all vertices and update min/max
  for (int iphi = 0; iphi <= fPoly->fSideCount; ++iphi) {
    Precision phi  = phiStart + iphi * sidePhi;
    Precision corx = conv * cos(phi);
    Precision cory = conv * sin(phi);
    for (int zPlaneCount = 0; zPlaneCount < fPoly->fZPlanes.size(); ++zPlaneCount) {
      // Do Rmin
      crt.Set(fPoly->fRMin[zPlaneCount] * corx, fPoly->fRMin[zPlaneCount] * cory, fPoly->fZPlanes[zPlaneCount]);
      for (int i = 0; i < 3; ++i) {
        aMin[i] = Min(aMin[i], crt[i]);
        aMax[i] = Max(aMax[i], crt[i]);
      }
      // Do Rmax
      crt.Set(fPoly->fRMax[zPlaneCount] * corx, fPoly->fRMax[zPlaneCount] * cory, fPoly->fZPlanes[zPlaneCount]);
      for (int i = 0; i < 3; ++i) {
        aMin[i] = Min(aMin[i], crt[i]);
        aMax[i] = Max(aMax[i], crt[i]);
      }
    }
  }
}

VECCORE_ATT_HOST_DEVICE
Precision UnplacedPolyhedron::DistanceSquarePointToSegment(Vector3D<Precision> &v1, Vector3D<Precision> &v2,
                                                           const Vector3D<Precision> &p) const
{

  Precision p1_p2_squareLength = (v1 - v2).Mag2();
  Precision dotProduct         = (p - v1).Dot(v1 - v2) / p1_p2_squareLength;
  if (dotProduct < 0) {
    return (p - v1).Mag2();
  } else if (dotProduct <= 1) {
    Precision p_p1_squareLength = (p - v1).Mag2();
    return p_p1_squareLength - dotProduct * dotProduct * p1_p2_squareLength;
  } else {
    return (p - v2).Mag2();
  }
}

VECCORE_ATT_HOST_DEVICE
bool UnplacedPolyhedron::InsideTriangle(Vector3D<Precision> &v1, Vector3D<Precision> &v2, Vector3D<Precision> &v3,
                                        const Vector3D<Precision> &p) const
{
  Precision epsilon_square = 0.00000001;
  Vector3D<Precision> vec1 = p - v1;
  Vector3D<Precision> vec2 = p - v2;
  Vector3D<Precision> vec3 = p - v3;

  bool sameSide1 = vec1.Dot(vec2) >= 0.;
  bool sameSide2 = vec1.Dot(vec3) >= 0.;
  bool sameSide3 = vec2.Dot(vec3) >= 0.;
  sameSide1      = sameSide1 && sameSide2 && sameSide3;

  if (sameSide1) return sameSide1;

  // If sameSide1 is false, point can be on the Surface or Outside
  // Use sqr of distance in order to check if point is on the Surface

  if (DistanceSquarePointToSegment(v1, v2, p) <= epsilon_square) return true;
  if (DistanceSquarePointToSegment(v1, v3, p) <= epsilon_square) return true;
  if (DistanceSquarePointToSegment(v2, v3, p) <= epsilon_square) return true;

  return false;
}

VECCORE_ATT_HOST_DEVICE
Vector3D<Precision> UnplacedPolyhedron::GetPointOnTriangle(Vector3D<Precision> const &v1, Vector3D<Precision> const &v2,
                                                           Vector3D<Precision> const &v3) const
{
  Precision r1 = RNG::Instance().uniform(0.0, 1.0);
  Precision r2 = RNG::Instance().uniform(0.0, 1.0);
  if (r1 + r2 > 1.) {
    r1 = 1. - r1;
    r2 = 1. - r2;
  }
  Vector3D<Precision> vec1 = v2 - v1;
  Vector3D<Precision> vec2 = v3 - v1;
  return v1 + r1 * vec1 + r2 * vec2;
}

Vector3D<Precision> UnplacedPolyhedron::SamplePointOnSurface() const
{
  int j;
  Precision chose, rnd, achose;
  Precision totArea = SurfaceArea();
  auto areaStruct   = fPoly->fAreaStruct;

  Vector3D<Precision> point1, point2, point3, point4, pReturn;

  // Chose area and Create Point on Surface
  chose = RNG::Instance().uniform(0.0, totArea);

  achose = areaStruct->top_area + areaStruct->bottom_area; // top or bottom

  // Point on Top or Bottom
  if (chose < achose) {

    chose     = RNG::Instance().uniform(0.0, achose);
    int iside = int(RNG::Instance().uniform(0.0, GetSideCount()));
    if (chose < areaStruct->top_area) {
      point1 = GetZSegment(GetZSegmentCount() - 1).outer.GetCorners()[2][iside];
      point2 = GetZSegment(GetZSegmentCount() - 1).outer.GetCorners()[3][iside];
      // Avoid generating points on degenerated triangles
      if (GetZSegment(GetZSegmentCount() - 1).inner.size() > 0) {
        point3 = GetZSegment(GetZSegmentCount() - 1).inner.GetCorners()[2][iside];
        point4 = GetZSegment(GetZSegmentCount() - 1).inner.GetCorners()[3][iside];
        if ((point4 - point3).Mag2() < kTolerance || RNG::Instance().uniform(0.0, 1.0) < 0.5)
          pReturn = GetPointOnTriangle(point3, point1, point2);
        else
          pReturn = GetPointOnTriangle(point4, point3, point2);
      } else {
        point3.Set(0.0, 0.0, GetZSegment(GetZSegmentCount() - 1).outer.GetCorners()[2][iside].z());
        pReturn = GetPointOnTriangle(point3, point1, point2);
      }
      return pReturn;
    } else {
      point1 = GetZSegment(0).outer.GetCorners()[0][iside];
      point2 = GetZSegment(0).outer.GetCorners()[1][iside];

      if (GetZSegment(0).inner.size() > 0) {
        point3 = GetZSegment(0).inner.GetCorners()[0][iside];
        point4 = GetZSegment(0).inner.GetCorners()[1][iside];
        // Avoid generating points on degenerated triangles
        if ((point4 - point3).Mag2() < kTolerance || RNG::Instance().uniform(0.0, 1.0) < 0.5)
          pReturn = GetPointOnTriangle(point3, point1, point2);
        else
          pReturn = GetPointOnTriangle(point4, point3, point2);
      } else {
        point3.Set(0.0, 0.0, GetZSegment(0).outer.GetCorners()[0][iside].z());
        pReturn = GetPointOnTriangle(point3, point1, point2);
      }
      return pReturn;
    }
  } else {
    // Point on Lateral segment or Phi segment
    for (j = 0; j < GetZSegmentCount(); j++) {
      achose += areaStruct->outer[j] + areaStruct->inner[j] + areaStruct->phi[j];
      if ((chose < achose) || (j == GetZSegmentCount() - 1)) {
        break;
      }
    }
  }

  // At this point we have chosen a subsection
  // between to adjacent plane cuts

  rnd = int(RNG::Instance().uniform(0.0, GetSideCount()));
  // area of the selected section
  totArea = areaStruct->outer[j] + areaStruct->inner[j] + areaStruct->phi[j];
  chose   = RNG::Instance().uniform(0., totArea);
  Vector3D<Precision> RandVec;
  if (chose <= areaStruct->outer[j]) {
    RandVec = (GetZSegment(j)).outer.GetPointOnFace(rnd);
    return RandVec;
  } else if (chose <= areaStruct->outer[j] + areaStruct->inner[j]) {
    return (GetZSegment(j)).inner.GetPointOnFace(rnd);
  }
  // Point on Phi segment
  rnd     = int(RNG::Instance().uniform(0.0, 1.999999));
  RandVec = (GetZSegment(j)).phi.GetPointOnFace(rnd);
  return RandVec;
}

#endif // !VECCORE_CUDA

VECCORE_ATT_HOST_DEVICE
bool UnplacedPolyhedron::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const
{
  // Compute normal vector to closest surface
  return (
      PolyhedronImplementation<Polyhedron::EInnerRadii::kGeneric, Polyhedron::EPhiCutout::kGeneric>::ScalarNormalKernel(
          *fPoly, point, normal));
}

VECCORE_ATT_HOST_DEVICE
void UnplacedPolyhedron::Print() const
{
  printf("UnplacedPolyhedron {%i sides, phi %f to %f, %i segments}", fPoly->fSideCount, GetPhiStart() * kRadToDeg,
         GetPhiEnd() * kRadToDeg, fPoly->fZSegments.size());
  printf("}");
}

VECCORE_ATT_HOST_DEVICE
void UnplacedPolyhedron::PrintSegments() const
{
  printf("Printing %i polyhedron segments: ", fPoly->fZSegments.size());
  for (int i = 0, iMax = fPoly->fZSegments.size(); i < iMax; ++i) {
    printf("  Outer: ");
    fPoly->fZSegments[i].outer.Print();
    printf("\n");
    if (fPoly->fHasPhiCutout) {
      printf("  Phi: ");
      fPoly->fZSegments[i].phi.Print();
      printf("\n");
    }
    if (fPoly->fZSegments[i].inner.size() > 0) {
      printf("  Inner: ");
      fPoly->fZSegments[i].inner.Print();
      printf("\n");
    }
  }
}

void UnplacedPolyhedron::Print(std::ostream &os) const
{
  int oldprc           = os.precision(16);
  int Nz               = fPoly->fZPlanes.size();
  const char *sbool[2] = {"false", "true"};

  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - polyhedron ***\n"
     << "     ===================================================\n"
     << " Parameters:\n"
     << " Phi start= " << fPoly->fPhiStart * vecgeom::kRadToDeg
     << " deg, Phi delta= " << fPoly->fPhiDelta * vecgeom::kRadToDeg << " deg\n"
     << "     Number of segments along phi: " << fPoly->fSideCount << "\n"
     << "     N = number of Z-planes: " << Nz << "\n"
     << "     z-coordinates (in cm):\n";

  for (int i = 0; i < Nz; ++i) {
    os << "       at Z=" << fPoly->fZPlanes[i] << "cm:" << " Rmin=" << fPoly->fRMin[i] << "cm,"
       << " Rmax=" << fPoly->fRMax[i] << "cm" << " sameZ=" << sbool[int(fPoly->fSameZ[i])] << std::endl;
  }
  os << "-----------------------------------------------------------\n";
  os.precision(oldprc);
}

VECCORE_ATT_HOST_DEVICE
void UnplacedPolyhedron::DetectConvexity()
{
  // Default safe convexity value
  fGlobalConvexity = false;

  if (fPoly->fConvexityPossible) {
    if (fPoly->fEqualRmax &&
        (fPoly->fPhiDelta <= kPi ||
         fPoly->fPhiDelta ==
             kTwoPi)) // In this case, Polycone become solid Cylinder, No need to check anything else, 100% convex
      fGlobalConvexity = true;
    else {
      if (fPoly->fPhiDelta <= kPi || fPoly->fPhiDelta == kTwoPi) {
        fGlobalConvexity = fPoly->fContinuousInSlope;
      }
    }
  }
}

#ifndef VECCORE_CUDA
SolidMesh *UnplacedPolyhedron::CreateMesh3D(Transformation3D const &trans, size_t nSegments) const
{

  typedef Vector3D<Precision> Vec_t;

  SolidMesh *sm = new SolidMesh();

  size_t n      = GetZSegmentCount();
  size_t nSides = GetSideCount();
  Precision k   = std::cos(0.5 * GetPhiDelta() / nSides); // divide R_side by k to get the R_corner

  Vec_t *vertices = new Vec_t[2 * (n + 1) * (nSides + 1)];
  size_t idx      = 0;
  size_t idx2     = ((n + 1) * (nSides + 1));
  for (size_t i = 0; i <= n; i++) {
    for (size_t j = 0; j <= nSides; j++) {
      vertices[idx++]  = GetPhiSection(j) * GetRMax()[i] / k + Vec_t(0, 0, GetZPlane(i));
      vertices[idx2++] = GetPhiSection(j) * GetRMin()[i] / k + Vec_t(0, 0, GetZPlane(i));
    }
  }

  sm->ResetMesh(2 * (n + 1) * (nSides + 1), 2 * n * nSides + 2 * nSides + 2 * n);

  sm->SetVertices(vertices, 2 * (n + 1) * (nSides + 1));
  delete[] vertices;
  sm->TransformVertices(trans);

  for (size_t i = 0, p = 0, k = nSides + 1, offset = (n + 1) * (nSides + 1); i < n; i++, p++, k++) {
    for (size_t j = 0; j < nSides; j++, p++, k++) {
      sm->AddPolygon(4, {p, p + 1, k + 1, k}, true);                                     // outer
      sm->AddPolygon(4, {k + offset, k + 1 + offset, p + offset + 1, p + offset}, true); // inner
    }
  }

  for (size_t i = 0, l = n * (nSides + 1), k = l + nSides + 1, p = k + l; i < nSides; i++, k++, l++, p++) {
    sm->AddPolygon(4, {k, k + 1, i + 1, i}, true); // lower
    sm->AddPolygon(4, {l, l + 1, p + 1, p}, true); // upper
  }

  if (GetPhiDelta() != kTwoPi) {
    for (size_t i = 0, j = 0, k = nSides + 1, offset = (n + 1) * (nSides + 1); i < n;
         i++, j += nSides + 1, k += nSides + 1) {
      sm->AddPolygon(4, {j, k, k + offset, j + offset}, true); // surface at sPhi
      sm->AddPolygon(4, {j + offset + nSides, k + offset + nSides, k + nSides, j + nSides},
                     true); // surface at sPhi + dPhi
    }
  }

  return sm;
}
#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedPolyhedron::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpuPtr) const
{

  // idea: reconstruct defining arrays: copy them to GPU; then construct the UnplacedPolycon object from scratch
  // on the GPU

  DevicePtr<Precision> zPlanesGpu;
  zPlanesGpu.Allocate(fPoly->fZPlanes.size());
  zPlanesGpu.ToDevice(&fPoly->fZPlanes[0], fPoly->fZPlanes.size());

  DevicePtr<Precision> rminGpu;
  rminGpu.Allocate(fPoly->fZPlanes.size());
  rminGpu.ToDevice(&fPoly->fRMin[0], fPoly->fZPlanes.size());

  DevicePtr<Precision> rmaxGpu;
  rmaxGpu.Allocate(fPoly->fZPlanes.size());
  rmaxGpu.ToDevice(&fPoly->fRMax[0], fPoly->fZPlanes.size());

  DevicePtr<cuda::VUnplacedVolume> gpupolyhedra =
      CopyToGpuImpl<UnplacedPolyhedron>(gpuPtr, fPoly->fPhiStart, fPoly->fPhiDelta, fPoly->fSideCount,
                                        fPoly->fZPlanes.size(), zPlanesGpu, rminGpu, rmaxGpu);

  zPlanesGpu.Deallocate();
  rminGpu.Deallocate();
  rmaxGpu.Deallocate();

  CudaAssertError();
  return gpupolyhedra;
}

DevicePtr<cuda::VUnplacedVolume> UnplacedPolyhedron::CopyToGpu() const { return CopyToGpuImpl<UnplacedPolyhedron>(); }

/**
 * Bulk-copy UnplacedPolyhedron instances to the device.
 * This function is significantly faster in constructing unplaced polyhedra on the GPU, because it bulk
 * copies the constructor arguments to the device, saving a lot of memory allocations and kernel invocations.
 * @param volumes Pointers to UnplacedPolyhedron instances.
 * @param devicePointer Locations where the unplaced polyhedra should be constructed.
 */
void UnplacedPolyhedron::CopyToGpu(std::vector<VUnplacedVolume const *> const &volumes,
                                   std::vector<DevicePtr<cuda::VUnplacedVolume>> const &devicePointers)
{
  const auto size = volumes.size();
  size_t size_buff{0};
  std::vector<size_t> sizes_buff(size, 0);
  std::vector<size_t> offsets_buff(size, 0);
  std::vector<Precision> floatData(2 * size, 0.);
  std::vector<int> intData(2 * size, 0);

  struct VarLengthData {
    std::vector<std::size_t> offsets[3];
    std::vector<Precision> data;
  } vld;
  struct RAIIDevPtr {
    DevicePtr<Precision> devPtr;
    DevicePtr<char> bufferPtr;
    RAIIDevPtr()                   = default;
    RAIIDevPtr(const RAIIDevPtr &) = delete;
    ~RAIIDevPtr()
    {
      devPtr.Deallocate();
      // must NOT deallocate bufferPtr, used to construct in place the PolyhedronStruct
      // We will need to register this device memory in a list of GPU cleanups, something like:
      // CudaManager::Instance().RegisterCleanup(bufferPtr.GetPtr());
    }
  } vldGPU;

  for (unsigned int i = 0; i < size; ++i) {
    UnplacedPolyhedron const &volume = static_cast<UnplacedPolyhedron const &>(*volumes[i]);
    floatData[0 * size + i]          = volume.fPoly->fPhiStart;
    floatData[1 * size + i]          = volume.fPoly->fPhiDelta;

    intData[0 * size + i] = volume.fPoly->fSideCount;
    intData[1 * size + i] = volume.fPoly->fZPlanes.size();

    sizes_buff[i]   = volume.fPoly->fSize;
    offsets_buff[i] = size_buff;
    size_buff += sizes_buff[i];

    int offsetCounter = 0;
    for (vecgeom::cxx::Array<Precision> const *array :
         {&volume.fPoly->fZPlanes, &volume.fPoly->fRMin, &volume.fPoly->fRMax}) {
      vld.offsets[offsetCounter++].push_back(vld.data.size());
      for (int j = 0; j < array->size(); ++j) {
        vld.data.push_back((*array)[j]);
      }
    }
  }

  vldGPU.devPtr.Allocate(vld.data.size());
  vldGPU.devPtr.ToDevice(vld.data.data(), vld.data.size());
  vldGPU.bufferPtr.Allocate(size_buff);

  std::vector<Precision const *> zPlanesGpuPtr, rMinGpuPtr, rMaxGpuPtr;
  std::vector<char const *> bufferGpuPtr;
  auto computeGpuPtr    = [&vldGPU](std::size_t offset) { return vldGPU.devPtr.GetPtr() + offset; };
  auto computeBufferPtr = [&vldGPU](std::size_t offset) { return vldGPU.bufferPtr.GetPtr() + offset; };
  std::transform(vld.offsets[0].begin(), vld.offsets[0].end(), std::back_inserter(zPlanesGpuPtr), computeGpuPtr);
  std::transform(vld.offsets[1].begin(), vld.offsets[1].end(), std::back_inserter(rMinGpuPtr), computeGpuPtr);
  std::transform(vld.offsets[2].begin(), vld.offsets[2].end(), std::back_inserter(rMaxGpuPtr), computeGpuPtr);
  std::transform(offsets_buff.begin(), offsets_buff.end(), std::back_inserter(bufferGpuPtr), computeBufferPtr);

  // Forwards all data to this constructor:
  // UnplacedPolyhedron(Precision phiStart, Precision phiDelta, const int sideCount, const int zPlaneCount,
  //                   Precision const zPlanes[], Precision const rMin[], Precision const rMax[], size_t const
  //                   buff_size, void const *buffer);
  ConstructManyOnGpu<vecgeom::cuda::UnplacedPolyhedron>(size, devicePointers.data(), floatData.data(),
                                                        floatData.data() + size, // phiStart, phiDelta
                                                        intData.data(),
                                                        intData.data() + size, // sideCount, zPlaneCount
                                                        zPlanesGpuPtr.data(), rMinGpuPtr.data(), rMaxGpuPtr.data(),
                                                        sizes_buff.data(), (void const *const *)bufferGpuPtr.data());
}

#endif

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedPolyhedron>::SizeOf();
template void DevicePtr<cuda::UnplacedPolyhedron>::Construct(Precision phiStart, Precision phiDelta, int sideCount,
                                                             int zPlaneCount, DevicePtr<Precision> zPlanes,
                                                             DevicePtr<Precision> rMin,
                                                             DevicePtr<Precision> rMax) const;
template void ConstructManyOnGpu<cuda::UnplacedPolyhedron>(
    std::size_t nElement, DevicePtr<cuda::VUnplacedVolume> const *gpu_ptrs, Precision const *phiStart,
    Precision const *phiDelta, int const *sideCount, int const *zPlaneCount, Precision const *const *zPlanes,
    Precision const *const *rMin, Precision const *const *rMax, size_t const *buff_size, void const *const *buffer);

} // namespace cxx

#endif

} // End namespace vecgeom
