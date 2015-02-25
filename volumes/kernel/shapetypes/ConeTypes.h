/*
 * ConeTypes.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_KERNEL_SHAPETYPES_CONETYPES_H_
#define VECGEOM_VOLUMES_KERNEL_SHAPETYPES_CONETYPES_H_

#include <string>
#include "volumes/UnplacedCone.h"

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes,UniversalCone,UniversalCone)

#ifndef VECGEOM_NO_SPECIALIZATION

VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes,NonHollowCone,UniversalCone)
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes,NonHollowConeWithSmallerThanPiSector,UniversalCone)
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes,NonHollowConeWithBiggerThanPiSector,UniversalCone)
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes,NonHollowConeWithPiSector,UniversalCone)

VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes,HollowCone,UniversalCone)
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes,HollowConeWithSmallerThanPiSector,UniversalCone)
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes,HollowConeWithBiggerThanPiSector,UniversalCone)
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes,HollowConeWithPiSector,UniversalCone)

#endif // VECGEOM_NO_SPECIALIZATION

inline namespace VECGEOM_IMPL_NAMESPACE {
namespace ConeTypes {


#define DEFINE_TRAIT_TYPE(name) \
    struct name { \
      static std::string toString() { \
        return #name; \
      } \
    } \

// A cone that encompasses all cases - not specialized and
// will do extra checks at runtime
DEFINE_TRAIT_TYPE(UniversalCone);

#ifndef VECGEOM_NO_SPECIALIZATION

// A cone not having rmin or phi sector
DEFINE_TRAIT_TYPE(NonHollowCone);
// A cone without rmin but with a phi sector smaller than pi
DEFINE_TRAIT_TYPE(NonHollowConeWithSmallerThanPiSector);
// A cone without rmin but with a phi sector greater than pi
DEFINE_TRAIT_TYPE(NonHollowConeWithBiggerThanPiSector);
// A cone without rmin but with a phi sector equal to pi
DEFINE_TRAIT_TYPE(NonHollowConeWithPiSector);

// A cone with rmin and no phi sector
DEFINE_TRAIT_TYPE(HollowCone);
// A cone with rmin and a phi sector smaller than pi
DEFINE_TRAIT_TYPE(HollowConeWithSmallerThanPiSector);
// A cone with rmin and a phi sector greater than pi
DEFINE_TRAIT_TYPE(HollowConeWithBiggerThanPiSector);
// A cone with rmin and a phi sector equal to pi
DEFINE_TRAIT_TYPE(HollowConeWithPiSector);

#endif // VECGEOM_NO_SPECIALIZATION

#undef DEFINE_TRAIT_TYPE

// Mapping of cone types to certain characteristics
enum TreatmentType {
  YES = 0,
  NO,
  UNKNOWN
};


// asking for phi treatment
template <typename T>
struct NeedsPhiTreatment {
  static const TreatmentType value=YES;
};

#ifndef VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsPhiTreatment<NonHollowCone> {
  static const TreatmentType value=NO;
};
template <>
struct NeedsPhiTreatment<HollowCone> {
  static const TreatmentType value=NO;
};

#endif // VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsPhiTreatment<UniversalCone> {
#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL TreatmentType value=UNKNOWN;
#else
  static const TreatmentType value=UNKNOWN;
#endif
};


template<typename T>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
bool checkPhiTreatment(const UnplacedCone& cone) {
  if(NeedsPhiTreatment<T>::value != UNKNOWN)
    return NeedsPhiTreatment<T>::value == YES;
  else
    // could use a direct constant for 2*M_PI here
    return cone.GetDPhi() < 2.*M_PI;
}

// asking for rmin treatment
template <typename T>
struct NeedsRminTreatment
{
#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL TreatmentType value=YES;
#else
  static const TreatmentType value=YES;
#endif
};

#ifndef VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsRminTreatment<NonHollowCone>
{
  static const TreatmentType value=NO;
};
template <>
struct NeedsRminTreatment<NonHollowConeWithSmallerThanPiSector>
{
  static const TreatmentType value=NO;
};
template <>
struct NeedsRminTreatment<NonHollowConeWithBiggerThanPiSector>
{
  static const TreatmentType value=NO;
};
template <>
struct NeedsRminTreatment<NonHollowConeWithPiSector>
{
  static const TreatmentType value=NO;
};

#endif // VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsRminTreatment<UniversalCone>
{
#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL TreatmentType value=UNKNOWN;
#else
  static const TreatmentType value=UNKNOWN;
#endif
};


template<typename T>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
bool checkRminTreatment(const UnplacedCone& cone) {
  if(NeedsRminTreatment<T>::value != UNKNOWN)
    return NeedsRminTreatment<T>::value == YES;
  else
    return cone.GetRmin1() > 0 || cone.GetRmin2() >0;
}


// sector size
enum AngleType
{
  NOANGLE = 0,
  SMALLER_THAN_PI,
  ONE_PI,
  BIGGER_THAN_PI,
  UNKNOWN_AT_COMPILE_TIME
};

template<typename T>
struct SectorType {
  static const AngleType value=NOANGLE;
};

template<>
struct SectorType<UniversalCone> {
  static const AngleType value=UNKNOWN_AT_COMPILE_TIME;
};

#ifndef VECGEOM_NO_SPECIALIZATION

template<>
struct SectorType<NonHollowConeWithSmallerThanPiSector> {
  static const AngleType value=SMALLER_THAN_PI;
};

template<>
struct SectorType<NonHollowConeWithPiSector> {
  static const AngleType value=ONE_PI;
};

template<>
struct SectorType<NonHollowConeWithBiggerThanPiSector> {
  static const AngleType value=BIGGER_THAN_PI;
};
template<>
struct SectorType<HollowConeWithSmallerThanPiSector> {
  static const AngleType value=SMALLER_THAN_PI;
};

template<>
struct SectorType<HollowConeWithPiSector> {
  static const AngleType value=ONE_PI;
};

template<>
struct SectorType<HollowConeWithBiggerThanPiSector> {
  static const AngleType value=BIGGER_THAN_PI;
};

#endif // VECGEOM_NO_SPECIALIZATION

} // end CONETYPES namespace

} } // End global namespace

#endif /* VECGEOM_VOLUMES_KERNEL_SHAPETYPES_CONETYPES_H_ */
