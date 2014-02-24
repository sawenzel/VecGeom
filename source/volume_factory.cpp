#include "volumes/box.h"
#include "management/volume_factory.h"

namespace vecgeom {

template <typename VolumeType>
class ShapeFactory;

template <>
class ShapeFactory<PlacedBox> {

public:

  template<TranslationCode trans_code, RotationCode rot_code>
  static VPlacedVolume* Create(LogicalVolume const &logical_volume,
                               TransformationMatrix const &matrix) {
    return new SpecializedBox<trans_code, rot_code>(logical_volume, matrix);
  }

};

VPlacedVolume* VolumeFactory::CreateSpecializedVolume(
    LogicalVolume const &logical_volume,
    TransformationMatrix const &matrix) const {

  const TranslationCode trans_code = matrix.GenerateTranslationCode();
  const RotationCode rot_code = matrix.GenerateRotationCode();

  // All shapes must be implemented here. Better solution?

  VPlacedVolume *placed = NULL;

  if (UnplacedBox const *const box =
      dynamic_cast<UnplacedBox const *const>(
        &logical_volume.unplaced_volume()
      )) {
    placed = CreateByTransformation<PlacedBox>(logical_volume, matrix,
                                                    trans_code, rot_code);
  }

  // Will return null if the passed shape isn't implemented here. Maybe throw an
  // exception instead?

  return placed;

}

template<typename VolumeType>
VPlacedVolume* VolumeFactory::CreateByTransformation(
    LogicalVolume const &logical_volume, TransformationMatrix const &matrix,
    const TranslationCode trans_code, const RotationCode rot_code) const {

  if (trans_code == 0 && rot_code == 0x1b1) {
    return ShapeFactory<VolumeType>::template Create<0, 0x1b1>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x1b1) {
    return ShapeFactory<VolumeType>::template Create<1, 0x1b1>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x18e) {
    return ShapeFactory<VolumeType>::template Create<0, 0x18e>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x18e) {
    return ShapeFactory<VolumeType>::template Create<1, 0x18e>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x076) {
    return ShapeFactory<VolumeType>::template Create<0, 0x076>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x076) {
    return ShapeFactory<VolumeType>::template Create<1, 0x076>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x16a) {
    return ShapeFactory<VolumeType>::template Create<0, 0x16a>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x16a) {
    return ShapeFactory<VolumeType>::template Create<1, 0x16a>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x155) {
    return ShapeFactory<VolumeType>::template Create<0, 0x155>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x155) {
    return ShapeFactory<VolumeType>::template Create<1, 0x155>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x0ad) {
    return ShapeFactory<VolumeType>::template Create<0, 0x0ad>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x0ad) {
    return ShapeFactory<VolumeType>::template Create<1, 0x0ad>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x0dc) {
    return ShapeFactory<VolumeType>::template Create<0, 0x0dc>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x0dc) {
    return ShapeFactory<VolumeType>::template Create<1, 0x0dc>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x0e3) {
    return ShapeFactory<VolumeType>::template Create<0, 0x0e3>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x0e3) {
    return ShapeFactory<VolumeType>::template Create<1, 0x0e3>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x11b) {
    return ShapeFactory<VolumeType>::template Create<0, 0x11b>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x11b) {
    return ShapeFactory<VolumeType>::template Create<1, 0x11b>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x0a1) {
    return ShapeFactory<VolumeType>::template Create<0, 0x0a1>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x0a1) {
    return ShapeFactory<VolumeType>::template Create<1, 0x0a1>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x10a) {
    return ShapeFactory<VolumeType>::template Create<0, 0x10a>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x10a) {
    return ShapeFactory<VolumeType>::template Create<1, 0x10a>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x046) {
    return ShapeFactory<VolumeType>::template Create<0, 0x046>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x046) {
    return ShapeFactory<VolumeType>::template Create<1, 0x046>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x062) {
    return ShapeFactory<VolumeType>::template Create<0, 0x062>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x062) {
    return ShapeFactory<VolumeType>::template Create<1, 0x062>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x054) {
    return ShapeFactory<VolumeType>::template Create<0, 0x054>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x054) {
    return ShapeFactory<VolumeType>::template Create<1, 0x054>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x111) {
    return ShapeFactory<VolumeType>::template Create<0, 0x111>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x111) {
    return ShapeFactory<VolumeType>::template Create<1, 0x111>(
             logical_volume, matrix
           );
  }
  if (trans_code == 0 && rot_code == 0x200) {
    return ShapeFactory<VolumeType>::template Create<0, 0x200>(
             logical_volume, matrix
           );
  }
  if (trans_code == 1 && rot_code == 0x200) {
    return ShapeFactory<VolumeType>::template Create<1, 0x200>(
             logical_volume, matrix
           );
  }

  // No specialization
  return ShapeFactory<VolumeType>::template Create<1, 0>(
           logical_volume, matrix
         );

}

} // End namespace vecgeom