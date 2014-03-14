/**
 * \author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_TYPES_H_
#define VECGEOM_BASE_TYPES_H_

namespace vecgeom {

#ifdef VECGEOM_FLOAT_PRECISION
typedef float Precision;
#else
typedef double Precision;
#endif

template <typename Type>
class Vector3D;

template <typename Type>
class SOA3D;

template <typename Type>
class AOS3D;

template <typename Type>
class Container;

template <typename Type>
class Vector;

template <typename Type>
class Array;

class LogicalVolume;

class VPlacedVolume;

typedef VPlacedVolume VUSolid;

class VUnplacedVolume;

class UnplacedBox;

class PlacedBox;

class TransformationMatrix;

class GeoManager;

#ifdef VECGEOM_CUDA
class CudaManager;
#endif

} // End namespace vecgeom

#ifdef VECGEOM_ROOT
class TGeoShape;
class TGeoBBox;
class TGeoNode;
class TGeoMatrix;
class TGeoVolume;
#endif

#ifdef VECGEOM_USOLIDS
class VUSolid;
class UBox;
#endif

#endif // VECGEOM_BASE_TYPES_H_