// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \file PlacedVolume.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/RNG.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/SOA3D.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

unsigned int VPlacedVolume::g_id_count = 0;

#ifndef VECCORE_CUDA
VPlacedVolume::VPlacedVolume(char const *const label, LogicalVolume const *const logical_volume,
                             Transformation3D const *const transformation)
    : id_(), label_(NULL), logical_volume_(logical_volume),
#ifdef VECGEOM_INPLACE_TRANSFORMATIONS
      fTransformation(*transformation)
#else
      fTransformation(transformation)
#endif
{
  id_ = g_id_count++;
  GeoManager::Instance().RegisterPlacedVolume(this);
  label_ = new std::string(label);
}

VECCORE_ATT_HOST_DEVICE
VPlacedVolume::VPlacedVolume(VPlacedVolume const &other) : id_(), label_(NULL), logical_volume_(), fTransformation()
{
  VECGEOM_VALIDATE(0, << "COPY CONSTRUCTOR FOR PlacedVolumes NOT IMPLEMENTED");
}

VECCORE_ATT_HOST_DEVICE
VPlacedVolume *VPlacedVolume::operator=(VPlacedVolume const &other)
{
  // deliberaty copy using memcpy to also copy the virtual table
  if (this != &other) {
    // overriding the vtable is exactly what I want
    // so I silence a compier warning via the void* cast
    std::memcpy((void *)this, (void *)&other, sizeof(VPlacedVolume));
  }
  return this;
  //    if (this != &other) // protect against invalid self-assignment
  //    {
  //        id_ = other.id_;
  //        label_ = other.label_;
  //        logical_volume_ = other.logical_volume_;
  //        transformation_ = other.transformation_;
  //    }
  //    return this;
}
#endif

VECCORE_ATT_HOST_DEVICE
VPlacedVolume::~VPlacedVolume()
{
#ifndef VECCORE_CUDA
  GeoManager::Instance().DeregisterPlacedVolume(id_);
  delete label_;
#endif
}

VECCORE_ATT_HOST_DEVICE
void VPlacedVolume::Print(const int indent) const
{
  for (int i = 0; i < indent; ++i)
    printf("  ");
  PrintType();
  printf(" [%i]", id_);
#ifndef VECCORE_CUDA
  if (label_->size()) {
    printf(" \"%s\"", label_->c_str());
  }
#endif
  printf(": \n");
  for (int i = 0; i <= indent; ++i)
    printf("  ");
  GetTransformation()->Print();
  printf("\n");
  logical_volume_->Print(indent + 1);
}

VECCORE_ATT_HOST_DEVICE
void VPlacedVolume::PrintContent(const int indent) const
{
  Print(indent);
  if (GetDaughters().size() > 0) {
    printf(":");
    for (VPlacedVolume const **vol = GetDaughters().begin(), **volEnd = GetDaughters().end(); vol != volEnd; ++vol) {
      printf("\n");
      (*vol)->PrintContent(indent + 3);
    }
  }
}

VECCORE_ATT_HOST
std::ostream &operator<<(std::ostream &os, VPlacedVolume const &vol)
{
  os << "(" << (*vol.GetUnplacedVolume()) << ", " << (*vol.GetTransformation()) << ")";
  return os;
}

// implement a default function for SamplePointOnSurface
// based on contains + DistanceToOut

Precision VPlacedVolume::Capacity()
{
#ifndef VECCORE_CUDA
  return GetUnplacedVolume()->Capacity();
#else
  return 0;
#endif
}

VECCORE_ATT_HOST_DEVICE
bool VPlacedVolume::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const
{
  // transform point to local space 
  const Transformation3D *tr = GetTransformation();
  Vector3D<Precision> lp = tr->Transform(point);
  // get normal
  Vector3D<Precision> ln;
  bool valid = GetUnplacedVolume()->Normal(lp, ln);
  // transform normal to master space
  GetTransformation()->InverseTransformDirection(ln, normal);
  return valid;
}

VECCORE_ATT_HOST_DEVICE
void VPlacedVolume::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
#ifndef VECCORE_CUDA
  Vector3D<Precision> pmin, pmax;
  GetUnplacedVolume()->Extent(pmin, pmax);
  // transform bounding box to master space and recalculate it
  const Transformation3D *tr = GetTransformation();
  if (tr->HasRotation()) {
    Vector3D<Precision> bbox[8];
    tr->InverseTransform(Vector3D<Precision>(pmin.x(), pmin.y(), pmin.z()), bbox[0]);
    tr->InverseTransform(Vector3D<Precision>(pmax.x(), pmin.y(), pmin.z()), bbox[1]);
    tr->InverseTransform(Vector3D<Precision>(pmin.x(), pmax.y(), pmin.z()), bbox[2]);
    tr->InverseTransform(Vector3D<Precision>(pmax.x(), pmax.y(), pmin.z()), bbox[3]);
    tr->InverseTransform(Vector3D<Precision>(pmin.x(), pmin.y(), pmax.z()), bbox[4]);
    tr->InverseTransform(Vector3D<Precision>(pmax.x(), pmin.y(), pmax.z()), bbox[5]);
    tr->InverseTransform(Vector3D<Precision>(pmin.x(), pmax.y(), pmax.z()), bbox[6]);
    tr->InverseTransform(Vector3D<Precision>(pmax.x(), pmax.y(), pmax.z()), bbox[7]);
    Precision xmin = bbox[0].x();
    Precision ymin = bbox[0].y();
    Precision zmin = bbox[0].z();
    Precision xmax = bbox[0].x();
    Precision ymax = bbox[0].y();
    Precision zmax = bbox[0].z();
    for (int i = 1; i < 8; ++i) {
      xmin = vecCore::math::Min(xmin, bbox[i].x());
      ymin = vecCore::math::Min(ymin, bbox[i].y());
      zmin = vecCore::math::Min(zmin, bbox[i].z());
      xmax = vecCore::math::Max(xmax, bbox[i].x());
      ymax = vecCore::math::Max(ymax, bbox[i].y());
      zmax = vecCore::math::Max(zmax, bbox[i].z());
    }
    aMin.Set(xmin, ymin, zmin);
    aMax.Set(xmax, ymax, zmax);
  } else {
    tr->InverseTransform(pmin, aMin);
    tr->InverseTransform(pmax, aMax);
  }
#endif
}

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::VPlacedVolume const *>::SizeOf();
template size_t DevicePtr<char>::SizeOf();
template size_t DevicePtr<float>::SizeOf();
template size_t DevicePtr<double>::SizeOf();
// template void DevicePtr<cuda::PlacedBox>::Construct(
//    DevicePtr<cuda::LogicalVolume> const logical_volume,
//    DevicePtr<cuda::Transformation3D> const transform,
//    const int id) const;

} // namespace cxx

#endif // VECCORE_CUDA

} // namespace vecgeom
