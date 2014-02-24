#ifndef VECGEOM_MANAGEMENT_VOLUMEFACTORY_H_
#define VECGEOM_MANAGEMENT_VOLUMEFACTORY_H_

#include "base/transformation_matrix.h"
#include "volumes/logical_volume.h"
#include "volumes/placed_volume.h"

namespace vecgeom {

class VolumeFactory {

public:

  static VolumeFactory& Instance() {
    static VolumeFactory instance;
    return instance;
  }

  VPlacedVolume* CreateSpecializedVolume(
    LogicalVolume const &logical_volume,
    TransformationMatrix const &matrix) const;

private:

  VolumeFactory() {}
  VolumeFactory(VolumeFactory const&);
  VolumeFactory& operator=(VolumeFactory const&);

  /**
   * Middle templated function call which dispatches specialization based on
   * transformation. Arguments will be determined by CreateSpecializedVolume().
   * \sa CreateSpecializedVolume
   * \sa Create
   */
  template<typename VolumeType>
  VPlacedVolume* CreateByTransformation(
      LogicalVolume const &logical_volume, TransformationMatrix const &matrix,
      const TranslationCode trans_code, const RotationCode rot_code) const;

  /**
   * Final templated function call. Arguments will be determined by
   * CreateSpecializedVolume() and CreateByTransformation().
   * \sa CreateSpecializedVolume
   * \sa CreateByTransformation
   */
  template<typename VolumeType, TranslationCode trans_code,
           RotationCode rot_code>
  VPlacedVolume* Create(LogicalVolume const &logical_volume,
                       TransformationMatrix const &matrix) {
    typedef VolumeType<trans_code, rot_code> SpecializedVolume;
    return new SpecializedVolume(logical_volume, matrix);
  }

};

} // End namespace vecgeom

#endif // VECGEOM_MANAGEMENT_VOLUMEFACTORY_H_