#ifndef VECGEOM_BASE_TYPES_H_
#define VECGEOM_BASE_TYPES_H_

namespace vecgeom {

enum ImplType { kVc, kCuda, kScalar, kCilk };

template <ImplType it>
struct Impl;

template <typename Type>
class Vector3D;

template <typename Type>
class SOA3D;

class VLogicalVolume;

class VPlacedVolume;

typedef VPlacedVolume VUSolid;

class VUnplacedVolume;

class TransformationMatrix;

class GeoManager;

} // End namespace vecgeom

#endif // VECGEOM_BASE_TYPES_H_