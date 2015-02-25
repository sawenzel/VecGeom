/**
 * @file TubeTypes.h
 * @author Georgios Bitzes (georgios.bitzes@cern.ch)
 *
 * Contains all possible tube types
 **/

#ifndef VECGEOM_VOLUMES_KERNEL_SHAPETYPES_TUBETYPES_H_
#define VECGEOM_VOLUMES_KERNEL_SHAPETYPES_TUBETYPES_H_

#include <string>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_NS_CONV(TubeTypes,UniversalTube,UniversalTube)

#ifndef VECGEOM_NO_SPECIALIZATION

VECGEOM_DEVICE_DECLARE_NS_CONV(TubeTypes,NonHollowTube,UniversalTube)
VECGEOM_DEVICE_DECLARE_NS_CONV(TubeTypes,NonHollowTubeWithSmallerThanPiSector,UniversalTube)
VECGEOM_DEVICE_DECLARE_NS_CONV(TubeTypes,NonHollowTubeWithBiggerThanPiSector,UniversalTube)
VECGEOM_DEVICE_DECLARE_NS_CONV(TubeTypes,NonHollowTubeWithPiSector,UniversalTube)

VECGEOM_DEVICE_DECLARE_NS_CONV(TubeTypes,HollowTube,UniversalTube)
VECGEOM_DEVICE_DECLARE_NS_CONV(TubeTypes,HollowTubeWithSmallerThanPiSector,UniversalTube)
VECGEOM_DEVICE_DECLARE_NS_CONV(TubeTypes,HollowTubeWithBiggerThanPiSector,UniversalTube)
VECGEOM_DEVICE_DECLARE_NS_CONV(TubeTypes,HollowTubeWithPiSector,UniversalTube)

#endif // VECGEOM_NO_SPECIALIZATION

inline namespace VECGEOM_IMPL_NAMESPACE { namespace TubeTypes {

#define DEFINE_TUBE_TYPE(name) \
    struct name { \
      VECGEOM_CUDA_HEADER_BOTH \
      static char const* toString() { \
        return #name; \
      } \
    } \

// A tube that encompasses all cases - not specialized and
// will do extra checks at runtime
DEFINE_TUBE_TYPE(UniversalTube);


//#ifndef VECGEOM_NO_SPECIALIZATION

// A tube not having rmin or phi sector
DEFINE_TUBE_TYPE(NonHollowTube);
// A tube without rmin but with a phi sector smaller than pi
DEFINE_TUBE_TYPE(NonHollowTubeWithSmallerThanPiSector);
// A tube without rmin but with a phi sector greater than pi
DEFINE_TUBE_TYPE(NonHollowTubeWithBiggerThanPiSector);
// A tube without rmin but with a phi sector equal to pi
DEFINE_TUBE_TYPE(NonHollowTubeWithPiSector);

// A tube with rmin and no phi sector
DEFINE_TUBE_TYPE(HollowTube);
// A tube with rmin and a phi sector smaller than pi
DEFINE_TUBE_TYPE(HollowTubeWithSmallerThanPiSector);
// A tube with rmin and a phi sector greater than pi
DEFINE_TUBE_TYPE(HollowTubeWithBiggerThanPiSector);
// A tube with rmin and a phi sector equal to pi
DEFINE_TUBE_TYPE(HollowTubeWithPiSector);

//#endif // VECGEOM_NO_SPECIALIZATION

#undef DEFINE_TUBE_TYPE

// Mapping of tube types to certain characteristics
enum ETreatmentType {
  kYes = 0,
  kNo,
  kUnknown
};

// asking for phi treatment
template <typename T>
struct NeedsPhiTreatment {
#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL ETreatmentType value=kYes;
#else
  static const ETreatmentType value=kYes;
#endif
};

#ifndef VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsPhiTreatment<NonHollowTube> {
#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL ETreatmentType value=kNo;
#else
  static const ETreatmentType value=kNo;
#endif
};
template <>
struct NeedsPhiTreatment<HollowTube> {
#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL ETreatmentType value=kNo;
#else
  static const ETreatmentType value=kNo;
#endif
};

#endif // VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsPhiTreatment<UniversalTube> {
#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL ETreatmentType value=kUnknown;
#else
  static const ETreatmentType value=kUnknown;
#endif
};

template<typename T>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool checkPhiTreatment(const UnplacedTube& tube) {
  if(NeedsPhiTreatment<T>::value != kUnknown)
    return NeedsPhiTreatment<T>::value == kYes;
  else
    return tube.dphi() < 2*M_PI;
}

// asking for rmin treatment
template <typename T>
struct NeedsRminTreatment
{
#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL ETreatmentType value=kYes;
#else
  static const ETreatmentType value=kYes;
#endif
};

#ifndef VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsRminTreatment<NonHollowTube>
{
#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL ETreatmentType value=kNo;
#else
  static const ETreatmentType value=kNo;
#endif
};
template <>
struct NeedsRminTreatment<NonHollowTubeWithSmallerThanPiSector>
{
  static const ETreatmentType value=kNo;
};
template <>
struct NeedsRminTreatment<NonHollowTubeWithBiggerThanPiSector>
{
  static const ETreatmentType value=kNo;
};
template <>
struct NeedsRminTreatment<NonHollowTubeWithPiSector>
{
  static const ETreatmentType value=kNo;
};

#endif // VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsRminTreatment<UniversalTube>
{
#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL ETreatmentType value=kUnknown;
#else
  static const ETreatmentType value=kUnknown;
#endif
};


template<typename T>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool checkRminTreatment(const UnplacedTube& tube) {
  if(NeedsRminTreatment<T>::value != kUnknown)
    return NeedsRminTreatment<T>::value == kYes;
  else
    return tube.rmin() > 0;
}

// sector size
enum EAngleType
{
  kNoAngle = 0,
  kSmallerThanPi,
  kOnePi,
  kBiggerThanPi,
  kUnknownAngle
};

template<typename T>
struct SectorType {
#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL EAngleType value=kNoAngle;
#else
  static const EAngleType value=kNoAngle;
#endif
};

template<>
struct SectorType<UniversalTube> {
#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL EAngleType value=kUnknownAngle;
#else
  static const EAngleType value=kUnknownAngle;
#endif
};

#ifndef VECGEOM_NO_SPECIALIZATION

template<>
struct SectorType<NonHollowTubeWithSmallerThanPiSector> {
  static const EAngleType value=kSmallerThanPi;
};

template<>
struct SectorType<NonHollowTubeWithPiSector> {
  static const EAngleType value=kOnePi;
};

template<>
struct SectorType<NonHollowTubeWithBiggerThanPiSector> {
  static const EAngleType value=kBiggerThanPi;
};
template<>
struct SectorType<HollowTubeWithSmallerThanPiSector> {
  static const EAngleType value=kSmallerThanPi;
};

template<>
struct SectorType<HollowTubeWithPiSector> {
  static const EAngleType value=kOnePi;
};

template<>
struct SectorType<HollowTubeWithBiggerThanPiSector> {
  static const EAngleType value=kBiggerThanPi;
};

#endif // VECGEOM_NO_SPECIALIZATION

} // End of TubeTypes

} } // End global namespace


#endif // VECGEOM_VOLUMES_KERNEL_SHAPETYPES_TUBETYPES_H_
