/// \file HypeTypes.h
// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)
//
// Contains all possible hype types
///

#ifndef VECGEOM_VOLUMES_KERNEL_SHAPETYPES_HYPETYPES_H_
#define VECGEOM_VOLUMES_KERNEL_SHAPETYPES_HYPETYPES_H_

#include <string>

namespace VECGEOM_NAMESPACE { namespace HypeTypes {

#define DEFINE_HYPE_TYPE(name) \
    struct name { \
      VECGEOM_CUDA_HEADER_BOTH \
      static char const* toString() { \
        return #name; \
      } \
    } \

// A Hype that encompasses all cases - not specialized and
// will do extra checks at runtime
DEFINE_HYPE_TYPE(UniversalHype);

// A Hype with rmin=0
DEFINE_HYPE_TYPE(NonHollowHype);

// A Hype with rmin!=0
DEFINE_HYPE_TYPE(HollowHype);

#undef DEFINE_HYPE_TYPE

// Mapping of Hype types to certain characteristics
enum ETreatmentType {
  kYes = 0,
  kNo,
  kUnknown
};


// asking for rmin treatment
template <typename T>
struct NeedsRminTreatment
{
  static const ETreatmentType value=kYes;
};
template <>
struct NeedsRminTreatment<NonHollowHype>
{
  static const ETreatmentType value=kNo;
};

template <>
struct NeedsRminTreatment<UniversalHype>
{
  static const ETreatmentType value=kUnknown;
};


template<typename T>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool checkRminTreatment(const UnplacedHype& hype) {
  if(NeedsRminTreatment<T>::value != kUnknown)
    return NeedsRminTreatment<T>::value == kYes;
  else
    return hype.GetRmin() > 0;
}




} }



#endif // VECGEOM_VOLUMES_KERNEL_SHAPETYPES_HYPETYPES_H_
