#ifndef VECGEOM_VOLUMES_UTILITIES_GENERATION_UTILITIES_H
#define VECGEOM_VOLUMES_UTILITIES_GENERATION_UTILITIES_H


namespace vecgeom { 
inline namespace VECGEOM_IMPL_NAMESPACE {

template<typename SpecializationT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* CreateSpecializedWithPlacement(
	 LogicalVolume const *const logical_volume,
	 Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {

   if(placement) {
      return new(placement) SpecializationT(
        logical_volume, transformation
#ifdef VECGEOM_NVCC
        ,(PlacedBox const*)nullptr, id
#endif
                                            ); // TODO: add bounding box?
   }

   return new SpecializationT(
#ifdef VECGEOM_NVCC
      logical_volume, transformation, (PlacedBox const *)nullptr, id); // TODO: add bounding box?
#else
      logical_volume, transformation);
#endif

}

} } // End global namespace


#endif // VECGEOM_VOLUMES_UTILITIES_GENERATION_UTILITIES_H
