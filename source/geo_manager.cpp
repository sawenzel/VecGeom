#include "management/geo_manager.h"

namespace VECGEOM_NAMESPACE {

//int GeoManager::RegisterVolume(VPlacedVolume const *const volume) {
//  return volume_count++;
//}



int GeoManager::getMaxDepth( ) const
{
	// walk all the volume hierarchy and insert
	// placed volumes if not already in the container
	GetMaxDepthVisitor depthvisitor;
	visitAllPlacedVolumes( world(), &depthvisitor, 1 );
	return depthvisitor.getMaxDepth();
}


} // End global namespace
