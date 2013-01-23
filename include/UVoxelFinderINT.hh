#ifndef USOLIDS_UVoxelizerNT
#define USOLIDS_UVoxelizerNT

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

#include "UBits.hh"

class UVoxelizerNT
{
public:
   void Voxelize();

   void DisplayVoxelLimits();
   void DisplayBoundaries();
   void DisplayListNodes();
   
   UVoxelizerNT();
   UVoxelizerNT(UMultiUnion* multi_union);
   ~UVoxelizerNT();

   // Method displaying the nodes located in a voxel characterized by its three indexes:
   void               GetCandidatesVoxel(int indexX, int indexY, int indexZ);
   // Method returning in a vector container the nodes located in a voxel characterized by its three indexes:
   int GetCandidatesVoxelArray(const UVector3 &point, std::vector<int> *list=NULL);

   int GetCandidatesVoxelBits(const UVector3 &point, UBits &bits);

   int GetCandidatesVoxelArrayWithLocation(const UVector3 &point, VUSolid::EnumInside location, std::vector<int> *list=NULL);

   // deprecated
   std::vector<int>        GetCandidatesVoxelArrayOld(const UVector3 point);
   std::vector<int> Intersect(unsigned char* mask);

   // Method returning the pointer to the array containing the characteristics of each box:
   double*            GetBoxes();
   
   double*            GetBoundariesX();
   double*            GetBoundariesY();
   double*            GetBoundariesZ();

   bool Contains(const UVector3 &point);
   
   int                BinarySearch(double position, VUSolid::EAxisType axis);
   int                BinarySearch(const UVector3 &vector, int axis);
   int                GetSlicesCount(VUSolid::EAxisType axis);

   double             DistanceToNextVoxel(const UVector3 &point, const UVector3 &direction, bool findFirst);

private:
   void               GetCandidatesAsString(const unsigned int* mask, std::string &result);

   double *CreateBoundaries();    
   double *SortBoundaries(double *fBoundaries, int index, int &number);

   void BuildVoxelLimits();

   void BuildListNodes();

   void DisplayBoundaries(double *boundaries, int boundariesCount);

   unsigned int *BuildListNodes(double *boundaries, int boundariesCount, int offset);

   UBits *BuildListNodesBits(double *boundaries, int boundariesCount, int offset);

   void DisplayListNodes(double *boundaries, int boundariesCount, unsigned int *bitmask);

private:

   UMultiUnion       *multiUnion;           // Solid to be voxelized (it is a union of several sub-solids)
   double            *boxes;                // Array of box limits on the 3 cartesian axis
//   double            *fBoundaries;           // Array of boundaries induced by the bounding boxes contained in "boxes"
   double            *boundariesX;    // Sorted and, if need be, skimmed boundaries along X axis
   int                slicesCountX;            // Total number of boundaries for X axis
   double            *boundariesY;    // Sorted and, if need be, skimmed boundaries along Y axis
   int                slicesCountY;            // Total number of boundaries for Y axis
   double            *boundariesZ;    // Sorted and, if need be, skimmed boundaries along Z axis
   int                slicesCountZ;            // Total number of boundaries for Z axis

//   int               *fNumNodesSliceX;       // Number of nodes in the considered slice along X axis
   // int               *fNumNodesSliceY;       // Number of nodes in the considered slice along Y axis
   // int               *fNumNodesSliceZ;       // Number of nodes in the considered slice along Z axis
   unsigned int *bitmaskX, *bitmaskY, *bitmaskZ;              // Each character of "memory" contains the nodes present in the considered slice

//   int                fNx, fNy, fNz;         // Number of bytes stored in "memory" for each axis
//   double             fTolerance;            // Minimal distance to discrminate two boundaries.

//   int nPer;

   UBits *bitsX, *bitsY, *bitsZ;

};
#endif
