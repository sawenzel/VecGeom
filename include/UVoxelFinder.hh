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

#include "UFacet.hh"
#include "UTriangularFacet.hh"
#include "UQuadrangularFacet.hh"

struct UVoxelBox
{
   UVector3 hlen; // half length of the box
   UVector3 pos; // position of the box
};

class UVoxelFinder
{
	friend class UVoxelCandidatesIterator;

public:

/*
// THIS METHOD IS NOT NECCESSARY. destructors are called automatically. it is enough to call destructor of
// object which contains the voxel finder
//
	inline void DeleteObjects()
	{
		for (int k = 0; k < 3; k++)
		{
			std::vector<double> &boundary = boundaries[k];
			boundary.resize(0);
			UBits &bitmask = bitmasks[k];
			bitmask.Clear();
			bitmask.SetBitNumber(1, false); // it is here so we can set the maximum 
		}
	}
*/

   void Voxelize(std::vector<VUSolid *> &solids, std::vector<UTransform3D *> &transforms);

   void Voxelize(std::vector<UFacet *> &facets);

   void DisplayVoxelLimits();
   void DisplayBoundaries();
   void DisplayListNodes();
   
   UVoxelFinder();
   ~UVoxelFinder();

   // Method displaying the nodes located in a voxel characterized by its three indexes:
   void               GetCandidatesVoxel(std::vector<int> &voxels);
   // Method returning in a vector container the nodes located in a voxel characterized by its three indexes:
   int GetCandidatesVoxelArray(const UVector3 &point, std::vector<int> &list, UBits *crossed=NULL) const;

   int GetCandidatesVoxelArray(const std::vector<int> &voxels, std::vector<int> &list, UBits *crossed=NULL) const;

   // Method returning the pointer to the array containing the characteristics of each box:
   inline const std::vector<UVoxelBox> &GetBoxes() const
   {
		return boxes;
   }
   inline const std::vector<double> &GetBoundary(int index) const
   {
	   return boundaries[index];
   }

   bool UpdateCurrentVoxel(const UVector3 &point, const UVector3 &direction, std::vector<int> &curVoxel) const;

   inline void GetVoxel(std::vector<int> &curVoxel, const UVector3 &point) const
   {
	   for (int i = 0; i <= 2; ++i) curVoxel[i] = UUtils::BinarySearch(GetBoundary(i), point[i]);
   }
  
   inline int GetBitsPerSlice () const { return nPerSlice*8*sizeof(unsigned int); }

   bool Contains(const UVector3 &point) const;
   
   double             DistanceToNext(const UVector3 &point, const UVector3 &direction, const std::vector<int> &curVoxel) const;

   double             DistanceToFirst(const UVector3 &point, const UVector3 &direction) const;

   double             SafetyFromOutside(const UVector3 &point) const;

   inline int GetVoxelsIndex(int x, int y, int z) const
   {
		int maxX = boundaries[0].size();
		int maxY = boundaries[1].size();
		int index = x + y*maxX + z*maxX*maxY;
		return index;
   }

   inline int GetVoxelsIndex(const std::vector<int> &voxels) const
   {
		return GetVoxelsIndex(voxels[0], voxels[1], voxels[2]);
   }

   inline int GetPointIndex(const UVector3 &p) const
   {
	    int maxX = boundaries[0].size();
		int maxY = boundaries[1].size();
		int x = UUtils::BinarySearch(boundaries[0], p[0]);
		int y = UUtils::BinarySearch(boundaries[1], p[1]);
		int z = UUtils::BinarySearch(boundaries[2], p[2]);
		int index = x + y*maxX + z*maxX*maxY;
		return index;
   }

   UBits empty;

   inline void BuildEmpty ()
   {
	   std::vector<int> xyz(3), max(3), candidates;

	   for (int i = 0; i <= 2; i++) max[i] = boundaries[i].size();
		unsigned int size = max[0] * max[1] * max[2];

		empty.Clear();
		empty.ResetBitNumber(size-1);

   		for (xyz[2] = 0; xyz[2] < max[2]; ++xyz[2])
		{
			for (xyz[1] = 0; xyz[1] < max[1]; ++xyz[1])
			{
				for (xyz[0] = 0; xyz[0] < max[0]; ++xyz[0])
				{
					int index = GetVoxelsIndex(xyz);
					int count = GetCandidatesVoxelArray(xyz, candidates);
					empty.SetBitNumber(index, count == 0);
				}
			}
		}
   }

   void SetMaxVoxels(int max);

   void SetMaxVoxels(UVector3 &reductionRatio);

private:
   std::string GetCandidatesAsString(const UBits &bits);

   void CreateSortedBoundary(std::vector<double> &boundaryRaw, int axis);
   
   void BuildBoundaries();

   void BuildReduceVoxels();

   void BuildVoxelLimits(std::vector<VUSolid *> &solids, std::vector<UTransform3D *> &transforms);

   void BuildVoxelLimits(std::vector<UFacet *> &facets);

   void DisplayBoundaries(std::vector<double> &boundaries);

   void BuildListNodes(bool countsOnly=false);

   void BuildBoundingBox();
   
private:

   int nPerSlice;

   std::vector<UVoxelBox> boxes;                // Array of box limits on the 3 cartesian axis

   std::vector<double> boundaries[3]; // Sorted and if need skimmed boundaries along X,Y,Z axis

   std::vector<int> candidatesCounts[3]; 

   UBits bitmasks[3];

   UVector3 boundingBoxCenter;

   UBox boundingBox;

   UVector3 reductionRatio;

   int maxVoxels;
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

