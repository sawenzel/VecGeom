#ifndef USOLIDS_UVoxelFinder
#define USOLIDS_UVoxelFinder

////////////////////////////////////////////////////////////////////////////////
//
//  Finder class handling voxels
//
////////////////////////////////////////////////////////////////////////////////

#include "VUSolid.hh"

#include "UUtils.hh"

#include "UTransform3D.hh"

//#include "UMultiUnion.hh"

#include <vector>
#include <string>

#include "UBits.h"

#include "UBox.hh"

struct UVoxelBox
{
   UVector3 hlen; // half length of the box
   UVector3 pos; // position of the box
};

class UVoxelFinder
{
	friend class UVoxelCandidatesIterator;

public:
   void Voxelize(std::vector<VUSolid *> &solids, std::vector<UTransform3D *> &transforms);

   void DisplayVoxelLimits();
   void DisplayBoundaries();
   void DisplayListNodes();
   
   UVoxelFinder();
   ~UVoxelFinder();

   // Method displaying the nodes located in a voxel characterized by its three indexes:
   void               GetCandidatesVoxel(int indexX, int indexY, int indexZ);
   // Method returning in a vector container the nodes located in a voxel characterized by its three indexes:
   int GetCandidatesVoxelArray(const UVector3 &point, std::vector<int> &list, UBits *crossed=NULL) const;

   int GetCandidatesVoxelBits(const UVector3 &point, UBits &bits) const;

   // Method returning the pointer to the array containing the characteristics of each box:
   const std::vector<UVoxelBox> &GetBoxes() const;
  
   inline int GetBitsPerSlice () const { return nPerSlice*8*sizeof(unsigned int); }

   bool Contains(const UVector3 &point) const;
   
   double             DistanceToNext(UVector3 &point, const UVector3 &direction) const;

   double             DistanceToFirst(UVector3 &point, const UVector3 &direction) const;

private:
   void               GetCandidatesAsString(const UBits &bits, std::string &result);

   void CreateSortedBoundary(std::vector<double> &boundaryRaw, int axis);
   
   void BuildOptimizedBoundaries();

   void BuildVoxelLimits(std::vector<VUSolid *> &solids, std::vector<UTransform3D *> &transforms);

   void DisplayBoundaries(std::vector<double> &boundaries);

   void BuildListNodes();

   void DisplayListNodes(std::vector<double> &boundaries, UBits &bitmask);

private:

   int nPerSlice;

   std::vector<UVoxelBox> boxes;                // Array of box limits on the 3 cartesian axis

   std::vector<double> boundaries[3]; // Sorted and, if need be, skimmed boundaries along X,Y,Z axis
   int boundariesCounts[3];            // Total number of boundaries for X,Y,Z axis
   // although, we could use instead calling to boundaries[3].size(), we have
   // found that it would very much affect performance in negative way (up to 15%)
   // therefore we keep this variable

   UBits bitmasks[3];

   UBox *boundingBox;
   UVector3 boundingBoxCenter;
};



class UVoxelCandidatesIterator
{
	private:
		unsigned int mask;
		int curInt, curBit, carNodes, n, sliceX, sliceY, sliceZ;
		unsigned int *maskX, *maskY, *maskZ;
		unsigned int *maskXLeft, *maskYLeft, *maskZLeft;
		bool nextAvailable;

	public:
		UVoxelCandidatesIterator(const UVoxelFinder &f, const UVector3 &point);

		int Next();
};

#endif

