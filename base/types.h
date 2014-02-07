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

template <typename Precision>
class VLogicalVolume;

template <typename Precision>
class VPlacedVolume;

#ifdef VECGEOM_STD_CXX11
template<typename Precision>
using VUSolid = VPlacedVolume<Precision>;
#endif

template <typename Precision>
class VUnplacedVolume;

template <typename Type>
class TransformationMatrix;

template <typename Precision>
class GeoManager;

} // End namespace vecgeom

#endif // VECGEOM_BASE_TYPES_H_