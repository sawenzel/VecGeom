/// @file TrdTypes.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_SHAPETYPES_TRDTYPES_H_
#define VECGEOM_VOLUMES_KERNEL_SHAPETYPES_TRDTYPES_H_

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_NS_CONV(TrdTypes,UniversalTrd)
VECGEOM_DEVICE_DECLARE_NS_CONV(TrdTypes,Trd1)
VECGEOM_DEVICE_DECLARE_NS_CONV(TrdTypes,Trd2)

inline namespace VECGEOM_IMPL_NAMESPACE { namespace TrdTypes {

#define DEFINE_TRD_TYPE(name) \
    struct name { \
      VECGEOM_CUDA_HEADER_BOTH \
      static char const* toString() { \
        return #name; \
      } \
    } \

// A Trd that includes all cases but does runtime checks
DEFINE_TRD_TYPE(UniversalTrd);
// A special case for which dy1 == dy2
DEFINE_TRD_TYPE(Trd1);
// A general case without runtime checks
DEFINE_TRD_TYPE(Trd2);

#undef DEFINE_TRD_TYPE

enum ETreatmentType {
  kYes = 0,
  kNo,
  kUnknown
};

template <typename T>
struct HasVaryingY {
  static const ETreatmentType value=kUnknown;
};
template <>
struct HasVaryingY<Trd1> {
  static const ETreatmentType value=kNo;
};
template <>
struct HasVaryingY<Trd2> {
  static const ETreatmentType value=kYes;
};

template<typename T>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool checkVaryingY(const UnplacedTrd& trd) {
  if(HasVaryingY<T>::value != kUnknown)
    return HasVaryingY<T>::value == kYes;
  else
    return trd.dy1() != trd.dy2();
}

} // end of TrdTypes

} } // End global namespace



#endif // VECGEOM_VOLUMES_KERNEL_SHAPETYPES_TRDTYPES_H_
