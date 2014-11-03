/**
 * @file tube_traits.h
 * @author Georgios Bitzes (georgios.bitzes@cern.ch)
 *
 * Contains all possible ways a tube can be specialized
 **/

#ifndef VECGEOM_VOLUMES_SPECIALIZATIONS_TUBETRAITS_H_
#define VECGEOM_VOLUMES_SPECIALIZATIONS_TUBETRAITS_H_

#include <string>
#include <csignal>


namespace VECGEOM_NAMESPACE { 

namespace TubeTraits {

#define DEFINE_TRAIT_TYPE(name) \
    struct name { \
      static std::string toString() { \
        return #name; \
      } \
    } \

// A tube that encompasses all cases - not specialized and
// will do extra checks at runtime
DEFINE_TRAIT_TYPE(UniversalTube);

// A tube not having rmin or phi sector
DEFINE_TRAIT_TYPE(NonHollowTube);
// A tube without rmin but with a phi sector smaller than pi
DEFINE_TRAIT_TYPE(NonHollowTubeWithSmallerThanPiSector);
// A tube without rmin but with a phi sector greater than pi
DEFINE_TRAIT_TYPE(NonHollowTubeWithBiggerThanPiSector);
// A tube without rmin but with a phi sector equal to pi
DEFINE_TRAIT_TYPE(NonHollowTubeWithPiSector);

// A tube with rmin and no phi sector
DEFINE_TRAIT_TYPE(HollowTube);
// A tube with rmin and a phi sector smaller than pi
DEFINE_TRAIT_TYPE(HollowTubeWithSmallerThanPiSector);
// A tube with rmin and a phi sector greater than pi
DEFINE_TRAIT_TYPE(HollowTubeWithBiggerThanPiSector);
// A tube with rmin and a phi sector equal to pi
DEFINE_TRAIT_TYPE(HollowTubeWithPiSector);

#undef DEFINE_TRAIT_TYPE

// Mapping of tube types to certain characteristics
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
template <>
struct NeedsPhiTreatment<NonHollowTube> {
  static const TreatmentType value=NO;
};
template <>
struct NeedsPhiTreatment<HollowTube> {
  static const TreatmentType value=NO;
};
template <>
struct NeedsPhiTreatment<UniversalTube> {
  static const TreatmentType value=UNKNOWN;
};


template<typename T>
VECGEOM_INLINE
bool checkPhiTreatment(const UnplacedTube& tube) {
  if(NeedsPhiTreatment<T>::value != UNKNOWN)
    return NeedsPhiTreatment<T>::value == YES;
  else
    return tube.dphi() < 2*M_PI;
}

// asking for rmin treatment
template <typename T>
struct NeedsRminTreatment
{
  static const TreatmentType value=YES;
};
template <>
struct NeedsRminTreatment<NonHollowTube>
{
  static const TreatmentType value=NO;
};
template <>
struct NeedsRminTreatment<NonHollowTubeWithSmallerThanPiSector>
{
  static const TreatmentType value=NO;
};
template <>
struct NeedsRminTreatment<NonHollowTubeWithBiggerThanPiSector>
{
  static const TreatmentType value=NO;
};
template <>
struct NeedsRminTreatment<NonHollowTubeWithPiSector>
{
  static const TreatmentType value=NO;
};
template <>
struct NeedsRminTreatment<UniversalTube>
{
  static const TreatmentType value=UNKNOWN;
};


template<typename T>
VECGEOM_INLINE
bool checkRminTreatment(const UnplacedTube& tube) {
  if(NeedsRminTreatment<T>::value != UNKNOWN)
    return NeedsRminTreatment<T>::value == YES;
  else
    return tube.rmin() > 0;
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
struct SectorType<UniversalTube> {
  static const AngleType value=UNKNOWN_AT_COMPILE_TIME;
};

template<>
struct SectorType<NonHollowTubeWithSmallerThanPiSector> {
  static const AngleType value=SMALLER_THAN_PI;
};

template<>
struct SectorType<NonHollowTubeWithPiSector> {
  static const AngleType value=ONE_PI;
};

template<>
struct SectorType<NonHollowTubeWithBiggerThanPiSector> {
  static const AngleType value=BIGGER_THAN_PI;
};
template<>
struct SectorType<HollowTubeWithSmallerThanPiSector> {
  static const AngleType value=SMALLER_THAN_PI;
};

template<>
struct SectorType<HollowTubeWithPiSector> {
  static const AngleType value=ONE_PI;
};

template<>
struct SectorType<HollowTubeWithBiggerThanPiSector> {
  static const AngleType value=BIGGER_THAN_PI;
};

}

} // End global namespace

#endif

