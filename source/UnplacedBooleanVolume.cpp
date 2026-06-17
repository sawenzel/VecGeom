/*
 * UnplacedBooleanVolume.cpp
 *
 *  Created on: 07.11.2014
 *      Author: swenzel
 */

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/UnplacedBooleanVolume.h"
#include "VecGeom/volumes/SpecializedBooleanVolume.h"
#include "VecGeom/management/Logger.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/volumes/utilities/GenerationUtilities.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/RNG.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedHalfSpace.h"
#ifndef VECCORE_CUDA
#include "VecGeom/volumes/UnplacedAssembly.h"
#include "VecGeom/volumes/UnplacedMultiUnion.h"
#include "VecGeom/volumes/UnplacedScaledShape.h"
#endif

#ifdef VECGEOM_CUDA_INTERFACE
#include "VecGeom/management/CudaManager.h"
#endif

#include <cmath>
#include <map>
#include <vector>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA
VECCORE_ATT_HOST_DEVICE
BooleanStruct const *BooleanHelper::GetBooleanStruct(VUnplacedVolume const *unplaced)
{
  if (!unplaced) return nullptr;

  UnplacedBooleanVolume<kUnion> const *buni = dynamic_cast<UnplacedBooleanVolume<kUnion> const *>(unplaced);
  BooleanStruct const *bstruct              = (buni) ? &buni->GetStruct() : nullptr;

  if (!bstruct) {
    UnplacedBooleanVolume<kIntersection> const *bint =
        dynamic_cast<UnplacedBooleanVolume<kIntersection> const *>(unplaced);
    bstruct = (bint) ? &bint->GetStruct() : nullptr;
  }

  if (!bstruct) {
    UnplacedBooleanVolume<kSubtraction> const *bsub =
        dynamic_cast<UnplacedBooleanVolume<kSubtraction> const *>(unplaced);
    bstruct = (bsub) ? &bsub->GetStruct() : nullptr;
  }
  return bstruct;
}

bool BooleanHelper::ContainsHalfSpace(VUnplacedVolume const *unplaced)
{
  if (!unplaced) return false;
  if (unplaced->GetType() == ESolidType::halfspace) return true;

  auto const *scaled = dynamic_cast<UnplacedScaledShape const *>(unplaced);
  if (scaled) return ContainsHalfSpace(scaled->GetPlaced()->GetUnplacedVolume());

  auto const *multiUnion = dynamic_cast<UnplacedMultiUnion const *>(unplaced);
  if (multiUnion) {
    for (size_t i = 0; i < multiUnion->GetNumberOfSolids(); ++i) {
      if (ContainsHalfSpace(multiUnion->GetNode(i)->GetUnplacedVolume())) return true;
    }
    return false;
  }

  auto const *assembly = dynamic_cast<UnplacedAssembly const *>(unplaced);
  if (assembly) {
    if (!assembly->GetLogicalVolume()) return false;
    for (auto const *daughter : assembly->GetLogicalVolume()->GetDaughters()) {
      if (ContainsHalfSpace(daughter->GetUnplacedVolume())) return true;
    }
    return false;
  }

  BooleanStruct const *bstruct = BooleanHelper::GetBooleanStruct(unplaced);
  if (!bstruct) return false;

  return ContainsHalfSpace(bstruct->fLeftVolume->GetUnplacedVolume()) ||
         ContainsHalfSpace(bstruct->fRightVolume->GetUnplacedVolume());
}

bool BooleanHelper::HasRootUnsupportedHalfSpaceUnion(VUnplacedVolume const *unplaced)
{
  if (!unplaced) return false;

  auto const *scaled = dynamic_cast<UnplacedScaledShape const *>(unplaced);
  if (scaled) return HasRootUnsupportedHalfSpaceUnion(scaled->GetPlaced()->GetUnplacedVolume());

  auto const *multiUnion = dynamic_cast<UnplacedMultiUnion const *>(unplaced);
  if (multiUnion) {
    for (size_t i = 0; i < multiUnion->GetNumberOfSolids(); ++i) {
      if (HasRootUnsupportedHalfSpaceUnion(multiUnion->GetNode(i)->GetUnplacedVolume())) return true;
    }
    return false;
  }

  auto const *assembly = dynamic_cast<UnplacedAssembly const *>(unplaced);
  if (assembly) {
    if (!assembly->GetLogicalVolume()) return false;
    for (auto const *daughter : assembly->GetLogicalVolume()->GetDaughters()) {
      if (HasRootUnsupportedHalfSpaceUnion(daughter->GetUnplacedVolume())) return true;
    }
    return false;
  }

  BooleanStruct const *bstruct = BooleanHelper::GetBooleanStruct(unplaced);
  if (!bstruct) return false;

  VUnplacedVolume const *leftUnplaced  = bstruct->fLeftVolume->GetUnplacedVolume();
  VUnplacedVolume const *rightUnplaced = bstruct->fRightVolume->GetUnplacedVolume();
  if (bstruct->fOp == kUnion && (ContainsHalfSpace(leftUnplaced) || ContainsHalfSpace(rightUnplaced))) return true;

  return HasRootUnsupportedHalfSpaceUnion(leftUnplaced) || HasRootUnsupportedHalfSpaceUnion(rightUnplaced);
}

bool BooleanHelper::IsFiniteBody(VPlacedVolume const *placed)
{
  return placed && IsFiniteBody(placed->GetUnplacedVolume());
}

bool BooleanHelper::IsFiniteBody(VUnplacedVolume const *unplaced)
{
  if (!unplaced) return false;
  if (unplaced->GetType() == ESolidType::halfspace) return false;

  auto const *scaled = dynamic_cast<UnplacedScaledShape const *>(unplaced);
  if (scaled) return IsFiniteBody(scaled->GetPlaced());

  auto const *multiUnion = dynamic_cast<UnplacedMultiUnion const *>(unplaced);
  if (multiUnion) {
    for (size_t i = 0; i < multiUnion->GetNumberOfSolids(); ++i) {
      if (!IsFiniteBody(multiUnion->GetNode(i))) return false;
    }
    return true;
  }

  auto const *assembly = dynamic_cast<UnplacedAssembly const *>(unplaced);
  if (assembly) {
    if (!assembly->GetLogicalVolume()) return true;
    for (auto const *daughter : assembly->GetLogicalVolume()->GetDaughters()) {
      if (!IsFiniteBody(daughter)) return false;
    }
    return true;
  }

  BooleanStruct const *bstruct = BooleanHelper::GetBooleanStruct(unplaced);
  if (!bstruct) return true;

  if (bstruct->fOp == kSubtraction) return IsFiniteBody(bstruct->fLeftVolume);
  if (bstruct->fOp == kUnion) return IsFiniteBody(bstruct->fLeftVolume) && IsFiniteBody(bstruct->fRightVolume);
  return IsFiniteBody(bstruct->fLeftVolume) || IsFiniteBody(bstruct->fRightVolume);
}

namespace {

struct HalfSpacePlane {
  Vector3D<Precision> fPoint;
  Vector3D<Precision> fNormal;
};

struct SurfaceOperand {
  VPlacedVolume const *fPlaced = nullptr;
  std::vector<Transformation3D const *> fTransformsToRoot;
  Precision fArea = 0.;
};

constexpr int kBooleanSurfaceSampleAttempts = 1000;

void TransformPlaneToParent(Transformation3D const *transform, HalfSpacePlane &plane)
{
  plane.fPoint  = transform->InverseTransform(plane.fPoint);
  plane.fNormal = transform->InverseTransformDirection(plane.fNormal);
  if (plane.fNormal.Mag2() > 0.) plane.fNormal.Normalize();
}

void CollectHalfSpacePlanes(VPlacedVolume const *placed, std::vector<HalfSpacePlane> &planes)
{
  if (!placed) return;

  VUnplacedVolume const *unplaced = placed->GetUnplacedVolume();
  if (unplaced->GetType() == ESolidType::halfspace) {
    auto const *halfspace = static_cast<UnplacedHalfSpace const *>(unplaced);
    HalfSpacePlane plane{halfspace->GetPoint(), halfspace->GetNormal()};
    TransformPlaneToParent(placed->GetTransformation(), plane);
    planes.push_back(plane);
    return;
  }

  BooleanStruct const *bstruct = BooleanHelper::GetBooleanStruct(unplaced);
  if (!bstruct) return;

  std::vector<HalfSpacePlane> localPlanes;
  CollectHalfSpacePlanes(bstruct->fLeftVolume, localPlanes);
  CollectHalfSpacePlanes(bstruct->fRightVolume, localPlanes);
  for (auto &plane : localPlanes) {
    TransformPlaneToParent(placed->GetTransformation(), plane);
    planes.push_back(plane);
  }
}

bool ExtentIsFinite(Vector3D<Precision> const &lower, Vector3D<Precision> const &upper)
{
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(lower[i]) || !std::isfinite(upper[i])) return false;
    if (lower[i] <= -0.5 * kInfLength || upper[i] >= 0.5 * kInfLength) return false;
    if (!(lower[i] < upper[i])) return false;
  }
  return true;
}

bool SamplePointOnPlaneInBBox(HalfSpacePlane const &plane, Vector3D<Precision> const &lower,
                              Vector3D<Precision> const &upper, Vector3D<Precision> &point)
{
  const Precision nx = std::fabs(plane.fNormal.x());
  const Precision ny = std::fabs(plane.fNormal.y());
  const Precision nz = std::fabs(plane.fNormal.z());
  int solveAxis      = 0;
  if (ny > nx && ny >= nz) {
    solveAxis = 1;
  } else if (nz > nx && nz > ny) {
    solveAxis = 2;
  }
  if (std::fabs(plane.fNormal[solveAxis]) <= 0.) return false;

  const int u      = (solveAxis + 1) % 3;
  const int v      = (solveAxis + 2) % 3;
  point[u]         = RNG::Instance().uniform(lower[u], upper[u]);
  point[v]         = RNG::Instance().uniform(lower[v], upper[v]);
  point[solveAxis] = plane.fPoint[solveAxis] - (plane.fNormal[u] * (point[u] - plane.fPoint[u]) +
                                                plane.fNormal[v] * (point[v] - plane.fPoint[v])) /
                                                   plane.fNormal[solveAxis];

  for (int i = 0; i < 3; ++i) {
    if (point[i] < lower[i] - kTolerance || point[i] > upper[i] + kTolerance) return false;
  }
  return true;
}

Precision CachedSurfaceArea(VPlacedVolume const *placed)
{
  // Calculating surface area can be expensive. Keep the legacy per-logical
  // volume cache for finite non-half-space operands.
  static std::map<size_t, Precision> idtoareamap;
  auto id = placed->GetLogicalVolume()->id();
  auto it = idtoareamap.find(id);
  if (it != idtoareamap.end()) return it->second;

  Precision area = placed->SurfaceArea();
  idtoareamap.insert(std::pair<size_t, Precision>(id, area));
  return area;
}

void CollectFiniteSurfaceOperands(VPlacedVolume const *placed, std::vector<Transformation3D const *> const &ancestors,
                                  std::vector<SurfaceOperand> &operands)
{
  if (!placed) return;

  VUnplacedVolume const *unplaced = placed->GetUnplacedVolume();
  if (!BooleanHelper::ContainsHalfSpace(unplaced)) {
    const Precision area = CachedSurfaceArea(placed);
    if (std::isfinite(area) && area > 0.) {
      SurfaceOperand operand;
      operand.fPlaced = placed;
      operand.fArea   = area;
      operand.fTransformsToRoot.push_back(placed->GetTransformation());
      operand.fTransformsToRoot.insert(operand.fTransformsToRoot.end(), ancestors.begin(), ancestors.end());
      operands.push_back(operand);
    }
    return;
  }

  BooleanStruct const *bstruct = BooleanHelper::GetBooleanStruct(unplaced);
  if (!bstruct) return;

  // Descend through half-space-containing Boolean nodes instead of calling
  // their sampled SurfaceArea() just to build selection weights.
  std::vector<Transformation3D const *> childAncestors;
  childAncestors.push_back(placed->GetTransformation());
  childAncestors.insert(childAncestors.end(), ancestors.begin(), ancestors.end());
  CollectFiniteSurfaceOperands(bstruct->fLeftVolume, childAncestors, operands);
  CollectFiniteSurfaceOperands(bstruct->fRightVolume, childAncestors, operands);
}

template <BooleanOperation Op>
bool SampleConstituentSurface(UnplacedBooleanVolume<Op> const &volume, std::vector<SurfaceOperand> const &operands,
                              Vector3D<Precision> &point)
{
  Precision totalArea = 0.;
  for (auto const &operand : operands)
    totalArea += operand.fArea;
  if (totalArea <= 0.) return false;

  Precision select     = RNG::Instance().uniform(0., totalArea);
  auto const *selected = &operands.back();
  for (auto const &operand : operands) {
    select -= operand.fArea;
    if (select <= 0.) {
      selected = &operand;
      break;
    }
  }

  point = selected->fPlaced->GetUnplacedVolume()->SamplePointOnSurface();
  for (auto const *transform : selected->fTransformsToRoot)
    point = transform->InverseTransform(point);
  return volume.Inside(point) == vecgeom::kSurface;
}

template <BooleanOperation Op>
bool PrepareHalfSpacePlaneSampling(UnplacedBooleanVolume<Op> const &volume, Vector3D<Precision> &lower,
                                   Vector3D<Precision> &upper, std::vector<HalfSpacePlane> &planes)
{
  volume.Extent(lower, upper);
  if (!ExtentIsFinite(lower, upper)) return false;

  CollectHalfSpacePlanes(volume.GetLeft(), planes);
  CollectHalfSpacePlanes(volume.GetRight(), planes);
  return !planes.empty();
}

template <BooleanOperation Op>
bool TrySampleHalfSpacePlaneSurface(UnplacedBooleanVolume<Op> const &volume, Vector3D<Precision> const &lower,
                                    Vector3D<Precision> const &upper, std::vector<HalfSpacePlane> const &planes,
                                    Vector3D<Precision> &point)
{
  if (planes.empty()) return false;

  size_t planeIndex = static_cast<size_t>(RNG::Instance().uniform(0., static_cast<Precision>(planes.size())));
  if (planeIndex >= planes.size()) planeIndex = planes.size() - 1;

  Vector3D<Precision> candidate;
  if (!SamplePointOnPlaneInBBox(planes[planeIndex], lower, upper, candidate)) return false;
  if (volume.Inside(candidate) == vecgeom::kSurface) {
    point = candidate;
    return true;
  }
  return false;
}

template <BooleanOperation Op>
Vector3D<Precision> SampleBooleanSurface(UnplacedBooleanVolume<Op> const &volume)
{
  Vector3D<Precision> point;
  const bool hasHalfSpace = BooleanHelper::ContainsHalfSpace(&volume);
  std::vector<SurfaceOperand> operands;
  CollectFiniteSurfaceOperands(volume.GetLeft(), {}, operands);
  CollectFiniteSurfaceOperands(volume.GetRight(), {}, operands);

  Vector3D<Precision> lower, upper;
  std::vector<HalfSpacePlane> planes;
  const bool canSampleHalfSpacePlane = hasHalfSpace && PrepareHalfSpacePlaneSampling(volume, lower, upper, planes);

  for (int counter = 0; counter < kBooleanSurfaceSampleAttempts; ++counter) {
    if (canSampleHalfSpacePlane && RNG::Instance().uniform() < 0.5 &&
        TrySampleHalfSpacePlaneSurface(volume, lower, upper, planes, point))
      return point;
    if (SampleConstituentSurface(volume, operands, point)) return point;
    if (canSampleHalfSpacePlane && TrySampleHalfSpacePlaneSurface(volume, lower, upper, planes, point)) return point;
  }

  VECGEOM_LOG(error) << "Could not generate point on surface for boolean";
  return point;
}

} // namespace

template <>
Vector3D<Precision> UnplacedBooleanVolume<kUnion>::SamplePointOnSurface() const
{
  return SampleBooleanSurface(*this);
}

template <>
Vector3D<Precision> UnplacedBooleanVolume<kIntersection>::SamplePointOnSurface() const
{
  return SampleBooleanSurface(*this);
}

template <>
Vector3D<Precision> UnplacedBooleanVolume<kSubtraction>::SamplePointOnSurface() const
{
  return SampleBooleanSurface(*this);
}

VECCORE_ATT_HOST_DEVICE
size_t BooleanHelper::CountBooleanNodes(VUnplacedVolume const *unplaced, size_t &nunion, size_t &nintersection,
                                        size_t &nsubtraction)
{
  BooleanStruct const *bstruct = BooleanHelper::GetBooleanStruct(unplaced);
  if (!bstruct) return 0;

  nunion += bstruct->fOp == kUnion;
  nintersection += bstruct->fOp == kIntersection;
  nsubtraction += bstruct->fOp == kSubtraction;
  CountBooleanNodes(bstruct->fLeftVolume->GetUnplacedVolume(), nunion, nintersection, nsubtraction);
  CountBooleanNodes(bstruct->fRightVolume->GetUnplacedVolume(), nunion, nintersection, nsubtraction);
  return (nunion + nintersection + nsubtraction);
}

UnplacedMultiUnion *BooleanHelper::Flatten(VUnplacedVolume const *unplaced, size_t min_unions,
                                           Transformation3D const *trbase, UnplacedMultiUnion *munion)
{
  size_t nunion{0}, nintersection{0}, nsubtraction{0};
  CountBooleanNodes(unplaced, nunion, nintersection, nsubtraction);
  if (nunion < min_unions) {
    // If a multi-union is being built-up, add this volume
    if (munion) munion->AddNode(unplaced, *trbase);
    return nullptr;
  }
  VUnplacedVolume const *vol;
  BooleanStruct *bstruct = (BooleanStruct *)BooleanHelper::GetBooleanStruct(unplaced);
  if (bstruct->fOp == kUnion) {
    bool creator = munion == nullptr;
    if (!munion) munion = new UnplacedMultiUnion();
    Transformation3D transform;

    // Compute left transformation
    transform = (trbase) ? *trbase : Transformation3D();
    transform.MultiplyFromRight(*bstruct->fLeftVolume->GetTransformation());
    Flatten(bstruct->fLeftVolume->GetUnplacedVolume(), min_unions, &transform, munion);

    // Compute right transformation
    transform = (trbase) ? *trbase : Transformation3D();
    transform.MultiplyFromRight(*bstruct->fRightVolume->GetTransformation());
    Flatten(bstruct->fRightVolume->GetUnplacedVolume(), min_unions, &transform, munion);

    if (creator) {
      munion->Close();
      return munion;
    }
    return nullptr;
  }

  // Analyze branches in case of subtraction or intersection
  vol            = (VUnplacedVolume *)bstruct->fLeftVolume->GetUnplacedVolume();
  auto left_bool = Flatten(vol, min_unions);
  if (left_bool) {
    // Replace existing left volume with the new one
    auto lvol            = new LogicalVolume(left_bool);
    auto pvol            = lvol->Place(bstruct->fLeftVolume->GetTransformation());
    bstruct->fLeftVolume = pvol;
  }
  vol             = (VUnplacedVolume *)bstruct->fRightVolume->GetUnplacedVolume();
  auto right_bool = Flatten(vol, min_unions);
  if (right_bool) {
    // Replace existing right volume with the new one
    auto lvol             = new LogicalVolume(right_bool);
    auto pvol             = lvol->Place(bstruct->fRightVolume->GetTransformation());
    bstruct->fRightVolume = pvol;
  }
  if (munion) munion->AddNode(unplaced, *trbase);
  return nullptr;
}
#endif

template <>
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedBooleanVolume<kSubtraction>::Create(
    LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
    const int id, const int copy_no, const int child_id,
#endif
    VPlacedVolume *const placement)
{
  return CreateSpecializedWithPlacement<SpecializedBooleanVolume<kSubtraction>>(logical_volume, transformation,
#ifdef VECCORE_CUDA
                                                                                id, copy_no, child_id,
#endif
                                                                                placement); // TODO: add bounding box?
}

template <>
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedBooleanVolume<kUnion>::Create(LogicalVolume const *const logical_volume,
                                                                        Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                                                        const int id, const int copy_no,
                                                                        const int child_id,
#endif
                                                                        VPlacedVolume *const placement)
{
  return CreateSpecializedWithPlacement<SpecializedBooleanVolume<kUnion>>(logical_volume, transformation,
#ifdef VECCORE_CUDA
                                                                          id, copy_no, child_id,
#endif
                                                                          placement); // TODO: add bounding box?
}

template <>
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedBooleanVolume<kIntersection>::Create(
    LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
    const int id, const int copy_no, const int child_id,
#endif
    VPlacedVolume *const placement)
{
  return CreateSpecializedWithPlacement<SpecializedBooleanVolume<kIntersection>>(logical_volume, transformation,
#ifdef VECCORE_CUDA
                                                                                 id, copy_no, child_id,
#endif
                                                                                 placement); // TODO: add bounding box?
}

template <>
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedBooleanVolume<kSubtraction>::SpecializedVolume(
    LogicalVolume const *const volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
    const int id, const int copy_no, const int child_id,
#endif
    VPlacedVolume *const placement) const
{
#ifndef VECCORE_CUDA

  return VolumeFactory::CreateByTransformation<UnplacedBooleanVolume<kSubtraction>>(volume, transformation,
#ifdef VECCORE_CUDA
                                                                                    id, copy_no, child_id,
#endif
                                                                                    placement);

#else
  // Compiling the above code with nvcc 6.5 fails with the error:
  // nvcc error   : 'ptxas' died due to signal 11 (Invalid memory reference)
  // at least when optimized.
  return nullptr;
#endif
}

template <>
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedBooleanVolume<kUnion>::SpecializedVolume(
    LogicalVolume const *const volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
    const int id, const int copy_no, const int child_id,
#endif
    VPlacedVolume *const placement) const
{
#ifndef VECCORE_CUDA

  return VolumeFactory::CreateByTransformation<UnplacedBooleanVolume<kUnion>>(volume, transformation,
#ifdef VECCORE_CUDA
                                                                              id, copy_no, child_id,
#endif
                                                                              placement);

#else
  // Compiling the above code with nvcc 6.5 fails with the error:
  // nvcc error   : 'ptxas' died due to signal 11 (Invalid memory reference)
  // at least when optimized.
  return nullptr;
#endif
}

template <>
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedBooleanVolume<kIntersection>::SpecializedVolume(
    LogicalVolume const *const volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
    const int id, const int copy_no, const int child_id,
#endif
    VPlacedVolume *const placement) const
{
#ifndef VECCORE_CUDA

  return VolumeFactory::CreateByTransformation<UnplacedBooleanVolume<kIntersection>>(volume, transformation,
#ifdef VECCORE_CUDA
                                                                                     id, copy_no, child_id,
#endif
                                                                                     placement);

#else
  // Compiling the above code with nvcc 6.5 fails with the error:
  // nvcc error   : 'ptxas' died due to signal 11 (Invalid memory reference)
  // at least when optimized.
  return nullptr;
#endif
}

VECCORE_ATT_HOST_DEVICE
void TransformedExtent(VPlacedVolume const *pvol, Vector3D<Precision> &aMin, Vector3D<Precision> &aMax)
{
// CUDA does not have min and max in std:: namespace
#ifndef VECCORE_CUDA
  using std::max;
  using std::min;
#endif
  Vector3D<Precision> lower, upper;
  pvol->Extent(lower, upper);
  Vector3D<Precision> delta = upper - lower;
  Precision minx, miny, minz, maxx, maxy, maxz;
  minx         = kInfLength;
  miny         = kInfLength;
  minz         = kInfLength;
  maxx         = -kInfLength;
  maxy         = -kInfLength;
  maxz         = -kInfLength;
  auto *transf = pvol->GetTransformation();
  for (int x = 0; x <= 1; ++x)
    for (int y = 0; y <= 1; ++y)
      for (int z = 0; z <= 1; ++z) {
        Vector3D<Precision> corner;
        corner.x()                            = lower.x() + x * delta.x();
        corner.y()                            = lower.y() + y * delta.y();
        corner.z()                            = lower.z() + z * delta.z();
        Vector3D<Precision> transformedcorner = transf->InverseTransform(corner);
        minx                                  = min(minx, transformedcorner.x());
        miny                                  = min(miny, transformedcorner.y());
        minz                                  = min(minz, transformedcorner.z());
        maxx                                  = max(maxx, transformedcorner.x());
        maxy                                  = max(maxy, transformedcorner.y());
        maxz                                  = max(maxz, transformedcorner.z());
      }
  aMin.x() = minx;
  aMin.y() = miny;
  aMin.z() = minz;
  aMax.x() = maxx;
  aMax.y() = maxy;
  aMax.z() = maxz;
}

template <>
VECCORE_ATT_HOST_DEVICE void UnplacedBooleanVolume<kSubtraction>::Extent(Vector3D<Precision> &aMin,
                                                                         Vector3D<Precision> &aMax) const
{
  Vector3D<Precision> minLeft, maxLeft;
  // NOTE: VPlacedVolume::Extent returns now the placed volume extent
  // We ignore the subtracted volume extent since we miss the extent cutoff functionality
  fBoolean.fLeftVolume->Extent(aMin, aMax);
}

template <>
VECCORE_ATT_HOST_DEVICE void UnplacedBooleanVolume<kUnion>::Extent(Vector3D<Precision> &aMin,
                                                                   Vector3D<Precision> &aMax) const
{
  Vector3D<Precision> minLeft, maxLeft, minRight, maxRight;
  // NOTE: VPlacedVolume::Extent returns now the placed volume extent
  fBoolean.fLeftVolume->Extent(minLeft, maxLeft);
  fBoolean.fRightVolume->Extent(minRight, maxRight);
  aMin = Vector3D<Precision>(Min(minLeft.x(), minRight.x()), Min(minLeft.y(), minRight.y()),
                             Min(minLeft.z(), minRight.z()));
  aMax = Vector3D<Precision>(Max(maxLeft.x(), maxRight.x()), Max(maxLeft.y(), maxRight.y()),
                             Max(maxLeft.z(), maxRight.z()));
}

template <>
VECCORE_ATT_HOST_DEVICE void UnplacedBooleanVolume<kIntersection>::Extent(Vector3D<Precision> &aMin,
                                                                          Vector3D<Precision> &aMax) const
{
  Vector3D<Precision> minLeft, maxLeft, minRight, maxRight;
  // NOTE: VPlacedVolume::Extent returns now the placed volume extent
  fBoolean.fLeftVolume->Extent(minLeft, maxLeft);
  fBoolean.fRightVolume->Extent(minRight, maxRight);
  aMin = Vector3D<Precision>(Max(minLeft.x(), minRight.x()), Max(minLeft.y(), minRight.y()),
                             Max(minLeft.z(), minRight.z()));
  aMax = Vector3D<Precision>(Min(maxLeft.x(), maxRight.x()), Min(maxLeft.y(), maxRight.y()),
                             Min(maxLeft.z(), maxRight.z()));
}

template <>
VECCORE_ATT_HOST_DEVICE bool UnplacedBooleanVolume<kSubtraction>::Normal(Vector3D<Precision> const &point,
                                                                         Vector3D<Precision> &normal) const
{
  // Compute normal vector to closest surface
  bool valid = false;
  BooleanImplementation<kSubtraction>::NormalKernel(GetStruct(), point, normal, valid);
  return valid;
}

template <>
VECCORE_ATT_HOST_DEVICE bool UnplacedBooleanVolume<kUnion>::Normal(Vector3D<Precision> const &point,
                                                                   Vector3D<Precision> &normal) const
{
  // Compute normal vector to closest surface
  bool valid = false;
  BooleanImplementation<kUnion>::NormalKernel(GetStruct(), point, normal, valid);
  return valid;
}

template <>
VECCORE_ATT_HOST_DEVICE bool UnplacedBooleanVolume<kIntersection>::Normal(Vector3D<Precision> const &point,
                                                                          Vector3D<Precision> &normal) const
{
  // Compute normal vector to closest surface
  bool valid = false;
  BooleanImplementation<kIntersection>::NormalKernel(GetStruct(), point, normal, valid);
  return valid;
}

#ifdef VECGEOM_CUDA_INTERFACE

// functions to copy data structures to GPU
template <>
DevicePtr<cuda::VUnplacedVolume> UnplacedBooleanVolume<kUnion>::CopyToGpu(
    DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  // here we have our recursion:
  // since UnplacedBooleanVolume has pointer members we need to copy/construct those members too
  // very brute force; because this might have been copied already
  // TODO: integrate this into CUDA MGR?

  // use CUDA Manager to lookup GPU pointer
  DevicePtr<cuda::VPlacedVolume> leftgpuptr  = CudaManager::Instance().LookupPlaced(GetLeft());
  DevicePtr<cuda::VPlacedVolume> rightgpuptr = CudaManager::Instance().LookupPlaced(GetRight());

  return CopyToGpuImpl<UnplacedBooleanVolume<kUnion>>(in_gpu_ptr, GetOp(), leftgpuptr, rightgpuptr);
}
template <>
DevicePtr<cuda::VUnplacedVolume> UnplacedBooleanVolume<kUnion>::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedBooleanVolume<kUnion>>();
}

// functions to copy data structures to GPU
template <>
DevicePtr<cuda::VUnplacedVolume> UnplacedBooleanVolume<kIntersection>::CopyToGpu(
    DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  // here we have our recursion:
  // since UnplacedBooleanVolume has pointer members we need to copy/construct those members too
  // very brute force; because this might have been copied already
  // TODO: integrate this into CUDA MGR?

  // use CUDA Manager to lookup GPU pointer
  DevicePtr<cuda::VPlacedVolume> leftgpuptr  = CudaManager::Instance().LookupPlaced(GetLeft());
  DevicePtr<cuda::VPlacedVolume> rightgpuptr = CudaManager::Instance().LookupPlaced(GetRight());

  return CopyToGpuImpl<UnplacedBooleanVolume<kIntersection>>(in_gpu_ptr, GetOp(), leftgpuptr, rightgpuptr);
}

template <>
DevicePtr<cuda::VUnplacedVolume> UnplacedBooleanVolume<kIntersection>::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedBooleanVolume<kIntersection>>();
}

template <>
// functions to copy data structures to GPU
DevicePtr<cuda::VUnplacedVolume> UnplacedBooleanVolume<kSubtraction>::CopyToGpu(
    DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  // here we have our recursion:
  // since UnplacedBooleanVolume has pointer members we need to copy/construct those members too
  // very brute force; because this might have been copied already
  // TODO: integrate this into CUDA MGR?

  // use CUDA Manager to lookup GPU pointer
  DevicePtr<cuda::VPlacedVolume> leftgpuptr  = CudaManager::Instance().LookupPlaced(GetLeft());
  DevicePtr<cuda::VPlacedVolume> rightgpuptr = CudaManager::Instance().LookupPlaced(GetRight());

  return CopyToGpuImpl<UnplacedBooleanVolume<kSubtraction>>(in_gpu_ptr, GetOp(), leftgpuptr, rightgpuptr);
}

template <>
DevicePtr<cuda::VUnplacedVolume> UnplacedBooleanVolume<kSubtraction>::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedBooleanVolume<kSubtraction>>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedBooleanVolume<kUnion>>::SizeOf();
template void DevicePtr<cuda::UnplacedBooleanVolume<kUnion>>::Construct(BooleanOperation op,
                                                                        DevicePtr<cuda::VPlacedVolume> left,
                                                                        DevicePtr<cuda::VPlacedVolume> right) const;
template size_t DevicePtr<cuda::UnplacedBooleanVolume<kIntersection>>::SizeOf();
template void DevicePtr<cuda::UnplacedBooleanVolume<kIntersection>>::Construct(
    BooleanOperation op, DevicePtr<cuda::VPlacedVolume> left, DevicePtr<cuda::VPlacedVolume> right) const;
template size_t DevicePtr<cuda::UnplacedBooleanVolume<kSubtraction>>::SizeOf();
template void DevicePtr<cuda::UnplacedBooleanVolume<kSubtraction>>::Construct(
    BooleanOperation op, DevicePtr<cuda::VPlacedVolume> left, DevicePtr<cuda::VPlacedVolume> right) const;

} // namespace cxx

#endif

} // End namespace vecgeom
