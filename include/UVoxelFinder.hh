#ifndef USOLIDS_UVoxelFinder
#define USOLIDS_UVoxelFinder

////////////////////////////////////////////////////////////////////////////////
//
//  Finder class handling voxels
//
////////////////////////////////////////////////////////////////////////////////

#ifndef USOLIDS_VUSolid
#include "VUSolid.hh"
#endif

#ifndef USOLIDS_UUtils
#include "UUtils.hh"
#endif

#ifndef USOLIDS_UTRANSFORM3D
#include "UTransform3D.hh"
#endif

#ifndef USOLIDS_UMULTIUNION
#include "UMultiUnion.hh"
#endif

#include <vector>
using namespace std;

class UVoxelFinder
{
public:
   void Voxelize() {}
   void SortBoundaries();
   void CreateBoundaries();  
   void BuildVoxelLimits();
   void DisplayVoxelLimits();
   void DisplayBoundaries();
   
   UVoxelFinder();
   UVoxelFinder(UMultiUnion* multi_union);

private:
   UMultiUnion *fMultiUnion; // Solid to be voxelized (it is a union of several sub-solids)
   double *fBoxes;
   double *fBoundaries;
   double *fXBoundaries;
   int fXNumBound;
   double *fYBoundaries;
   int fYNumBound;   
   double *fZBoundaries;      
   int fZNumBound;   
};
#endif
