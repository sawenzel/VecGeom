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

#ifndef USOLIDS_UTransform3D
#include "UTransform3D.hh"
#endif

#ifndef USOLIDS_UMultiUnion
#include "UMultiUnion.hh"
#endif

#include <vector>
#include <string>

using namespace std;

class UVoxelFinder
{
public:
   void Voxelize();

   void BuildVoxelLimits();
   void DisplayVoxelLimits();

   void CreateBoundaries();    
   void SortBoundaries(); 
   void DisplayBoundaries();

   void BuildListNodes();
   void DisplayListNodes(); 
   
   UVoxelFinder();
   UVoxelFinder(UMultiUnion* multi_union);
   ~UVoxelFinder();

   // Method displaying the nodes located in a voxel characterized by its three indexes:
   void               GetCandidatesVoxel(int indexX, int indexY, int indexZ);
   // Method returning in a vector container the nodes located in a voxel characterized by its three indexes:
   vector<int>        GetCandidatesVoxelArray(int indexX, int indexY, int indexZ);
   vector<int>        GetCandidatesVoxelArray2(UVector3 point);
   // Method determining in which voxel(s) is located the passed point:
   vector<UVector3>   ConvertPointToIndexes(UVector3 point);

private:
   void               GetCandidatesAsString(const char* mask, std::string &result);

private:
   UMultiUnion       *fMultiUnion;           // Solid to be voxelized (it is a union of several sub-solids)
   double            *fBoxes;                // Array of box limits on the 3 cartesian axis
   double            *fBoundaries;           // Array of boundaries induced by the bounding boxes contained
                                             // in "fBoxes"
   double            *fXSortedBoundaries;    // Sorted and, if need be, skimmed boundaries along X axis
   int                fXNumBound;            // Total number of boundaries for X axis
   double            *fYSortedBoundaries;    // Sorted and, if need be, skimmed boundaries along Y axis
   int                fYNumBound;            // Total number of boundaries for Y axis
   double            *fZSortedBoundaries;    // Sorted and, if need be, skimmed boundaries along Z axis 
   int                fZNumBound;            // Total number of boundaries for Z axis
   int               *fNumNodesSliceX;       // Number of nodes in the considered slice along X axis
   int               *fNumNodesSliceY;       // Number of nodes in the considered slice along Y axis
   int               *fNumNodesSliceZ;       // Number of nodes in the considered slice along Z axis
   char              *fMemoryX;              // Each character of "fmemory" contains the nodes present in the
   char              *fMemoryY;              // considered slice
   char              *fMemoryZ;      
   int                fNx, fNy, fNz;         // Number of bytes stored in "fmemory" for each axis
   double             fTolerance;            // Minimal distance to discrminate two boundaries.
};
#endif
