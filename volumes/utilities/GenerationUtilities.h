
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
#ifdef VECGEOM_NVCC
        logical_volume, transformation, NULL, id); // TODO: add bounding box?
#else
        logical_volume, transformation);
#endif
	}

	return new SpecializationT(
#ifdef VECGEOM_NVCC
      logical_volume, transformation, NULL, id); // TODO: add bounding box?
#else
      logical_volume, transformation);
#endif

}

} } // End global namespace


