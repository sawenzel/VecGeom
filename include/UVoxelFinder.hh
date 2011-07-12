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

#define DIST_INTER_BOUND 1E-10

using namespace std;

class UVoxelFinder
{
public:
   void Voxelize() {}

   void BuildVoxelLimits();
   void DisplayVoxelLimits();

   void CreateBoundaries();    
   void SortBoundaries(); 
   void DisplayBoundaries();

   void BuildListNodes();
   void DisplayListNodes();
   
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
   int* fNsliceX;
   int* fNsliceY;
   int* fNsliceZ;
   char*	fIndcX;   
   char*	fIndcY;
   char*	fIndcZ;      
   int fNx, fNy, fNz;
};
#endif
