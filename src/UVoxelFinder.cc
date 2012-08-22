
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

#include <set>

#include "VUSolid.hh" 
#include "UVoxelFinder.hh"
#include "UUtils.hh"
#include "UOrb.hh"

using namespace std;

//______________________________________________________________________________   
UVoxelFinder::UVoxelFinder()
{
	SetMaxVoxels(0);
}

//______________________________________________________________________________   
UVoxelFinder::~UVoxelFinder()
{
}

//______________________________________________________________________________   
void UVoxelFinder::BuildVoxelLimits(vector<VUSolid *> &solids, vector<UTransform3D *> &transforms)
{
	// "BuildVoxelLimits"'s aim is to store the coordinates of the origin as well as
	// the half lengths related to the bounding box of each node.
	// These quantities are stored in the array "boxes" (6 different values per
	// node.
	if (int numNodes = solids.size()) // Number of nodes in "multiUnion"
	{
		boxes.resize(numNodes); // Array which will store the half lengths
		nPerSlice = 1+(boxes.size()-1)/(8*sizeof(unsigned int));

		// related to a particular node, but also
		// the coordinates of its origin

		UVector3 toleranceVector;
		toleranceVector.Set(VUSolid::Tolerance());

		for (int i = 0; i < numNodes; ++i)
		{
			VUSolid &solid = *solids[i];
			UTransform3D &transform = *transforms[i];
			UVector3 min, max;
			solid.Extent(min, max);
			if (solid.GetEntityType() == "Orb")
			{
				UOrb &orb = *(UOrb *) &solid;
				UVector3 orbToleranceVector;
				orbToleranceVector.Set(orb.GetRadialTolerance()/2.0);
				min -= orbToleranceVector; max += orbToleranceVector;
			}
			else
			{
				min -= toleranceVector; max += toleranceVector;
			}
			UUtils::TransformLimits(min, max, transform);

			boxes[i].hlen = (max - min) / 2;
			boxes[i].pos = transform.fTr;
		}
	}
}

void UVoxelFinder::BuildVoxelLimits(vector<UFacet *> &facets)
{
	// "BuildVoxelLimits"'s aim is to store the coordinates of the origin as well as
	// the half lengths related to the bounding box of each node.
	// These quantities are stored in the array "boxes" (6 different values per
	// node.
	if (int numNodes = facets.size()) // Number of nodes in "multiUnion"
	{
		boxes.resize(numNodes); // Array which will store the half lengths
		nPerSlice = 1+(boxes.size()-1)/(8*sizeof(unsigned int));

		UVector3 toleranceVector;
		toleranceVector.Set(100 * VUSolid::Tolerance());

		for (int i = 0; i < numNodes; ++i)
		{
			UFacet &facet = *facets[i];
			UVector3 min, max;
			UVector3 x(1,0,0), y(0,1,0), z(0,0,1);
			UVector3 extent;
			max.Set (facet.Extent(x), facet.Extent(y), facet.Extent(z));
			min.Set (-facet.Extent(-x), -facet.Extent(-y), -facet.Extent(-z));
			min -= toleranceVector; max += toleranceVector;
			UVector3 hlen = (max - min) / 2;
			boxes[i].hlen = hlen;
			boxes[i].pos = min + hlen;
		}
	}
}


//______________________________________________________________________________   
void UVoxelFinder::DisplayVoxelLimits()
{
	// "DisplayVoxelLimits" displays the dX, dY, dZ, pX, pY and pZ for each node
	int numNodes = boxes.size();
	for(int i = 0; i < numNodes; ++i)
	{
		cout << setw(10) << setiosflags(ios::fixed) << setprecision(16) <<
		"    -> Node " << i+1 <<  ":\n" << 
		"\t * [x,y,z] = " << boxes[i].hlen <<
		"\t * [x,y,z] = " << boxes[i].pos << "\n";
	}
}

//______________________________________________________________________________   
void UVoxelFinder::CreateSortedBoundary(vector<double> &boundary, int axis)
{
	// "CreateBoundaries"'s aim is to determine the slices induced by the bounding boxes,
	// along each axis. The created boundaries are stored in the array "boundariesRaw"
	int numNodes = boxes.size(); // Number of nodes in structure of "UMultiUnion" type
	// Determination of the boundries along x, y and z axis:
	for(int i = 0 ; i < numNodes; ++i)   
	{
		// For each node, the boundaries are created by using the array "boxes"
		// built in method "BuildVoxelLimits":
		double p = boxes[i].pos[axis], d = boxes[i].hlen[axis];
		// x boundaries
		boundary[2*i] = p - d;
		boundary[2*i+1] = p + d;
	}
	sort(boundary.begin(), boundary.end());
}

void UVoxelFinder::BuildBoundaries()
{
	// "SortBoundaries" orders the boundaries along each axis (increasing order)
	// and also does not take into account redundant boundaries, ie if two boundaries
	// are separated by a distance strictly inferior to "tolerance".
	// The sorted boundaries are respectively stored in:
	//              * boundaries[0..2]
	// In addition, the number of elements contained in the three latter arrays are
	// precised thanks to variables: boundariesCountX, boundariesCountY and boundariesCountZ.

	if (int numNodes = boxes.size())
	{
		const double tolerance = VUSolid::Tolerance() / 100.0; // Minimal distance to discrminate two boundaries.
		vector<double> sortedBoundary(2*numNodes);

		int added, considered;

		for (int j = 0; j <= 2; ++j)
		{
			CreateSortedBoundary(sortedBoundary, j);
			vector<double> &boundary = boundaries[j];
			boundary.clear();

			added = considered = 0;

			for(int i = 0 ; i < 2*numNodes; ++i)
			{
				double newBoundary = sortedBoundary[i];
				int size = boundary.size();
				if(!size || abs(boundary[size-1] - newBoundary) > tolerance)
				{
					considered++;
					{
						boundary.push_back(newBoundary);
						continue;
					}
				}
				// If two successive boundaries are too close from each other, only the first one is considered 	
				{
					// cout << "Skipping boundary [" << j << "] : " << i << endl;
				}
			}

			int n = boundary.size();
			int max = 100000;
			if (n > max/2)
			{
				int skip = n / (max /2); // n has to be 2x bigger then 50.000. therefore only from 100.000 re
				vector<double> reduced;
				for (int i = 0; i < n; i++)
				{
					// 50 ok for 2k, 1000, 2000
					if (i % skip == 0 || i == 0 || i == boundary.size() - 1) // this condition of merging boundaries was wrong, it did not count with right part, which can be completely ommited and not included in final consideration. Now should be OK
						reduced.push_back(boundary[i]);
				}
				boundary = reduced;
			}
		}
	}
}

void UVoxelFinder::DisplayBoundaries()
{
	char axis[3] = {'X', 'Y', 'Z'};
	for (int i = 0; i <= 2; ++i)
	{
		cout << " * " << axis[i] << " axis:" << endl << "    | ";
		DisplayBoundaries(boundaries[i]);
	}
}

//______________________________________________________________________________   
void UVoxelFinder::DisplayBoundaries(vector<double> &boundaries)
{
	// Prints the positions of the boundaries of the slices on the three axis:

	int count = boundaries.size();
	for(int i = 0; i < count; ++i)
	{
		cout << setw(10) << setiosflags(ios::fixed) << setprecision(16) << boundaries[i];
		//      printf("%19.15f ",boundaries[0][i]);      
		//      cout << boundaries[0][i] << " ";
		if(i != count-1) cout << "-> ";
	}
	cout << "|" << endl << "Number of boundaries: " << count << endl;
}

// TODO: needs to be rewrittten to make it much faster, using std::set

void UVoxelFinder::BuildListNodes(bool countsOnly)
{
	// "BuildListNodes" stores in the bitmasks solids present in each slice along an axis.
//	const double localTolerance = 0;
	int numNodes = boxes.size();
	int bitsPerSlice = GetBitsPerSlice();

	for (int k = 0; k < 3; k++)
	{
		int total = 0;
		vector<double> &boundary = boundaries[k];
		int voxelsCount = boundary.size() - 1;
		UBits &bitmask = bitmasks[k];

		if (!countsOnly)
		{
			bitmask.Clear();
			bitmask.SetBitNumber(voxelsCount*bitsPerSlice-1, false); // it is here so we can set the maximum number of bits. this line will rellocate the memory and set all to zero
		}
		vector<int> &candidatesCount = candidatesCounts[k];
		candidatesCount.resize(voxelsCount);

		for(int i = 0 ; i < voxelsCount; ++i) 
			candidatesCount[i] = 0;
			
		// Loop on the nodes, number of slices per axis
		for(int j = 0 ; j < numNodes; j++)
		{
			// Determination of the minimum and maximum position along x
			// of the bounding boxe of each node:
			double p = boxes[j].pos[k], d = boxes[j].hlen[k];

			double min = p - d; // - localTolerance;
			double max = p + d; // + localTolerance;

			int i = UUtils::BinarySearch(boundary, min);

			if (i < 0 || i >= voxelsCount)
				i = i;

			do
			{
				if (!countsOnly) 
					bitmask.SetBitNumber(i*bitsPerSlice+j);

				candidatesCount[i]++;
				total++;
				i++;
			}
			while (max > boundary[i] && i < voxelsCount);
		}

		/*
		if (voxelsCount < 20000)
		{
			int total2 = 0;
			for(int i = 0 ; i < voxelsCount; ++i)
			{
				candidatesCount[i] = 0;
				// Loop on the nodes, number of slices per axis
				for(int j = 0 ; j < numNodes; j++)
				{
					// Determination of the minimum and maximum position along x
					// of the bounding boxe of each node:
					double p = boxes[j].pos[k], d = boxes[j].hlen[k];
					double leftBoundary = boundary[i];
					double rightBoundary = boundary[i+1];
					double min = p - d; // - localTolerance;
					double max = p + d; // + localTolerance;
					if (min < rightBoundary && max > leftBoundary)
					{
						// Storage of the number of the node in the array:
						bitmask.SetBitNumber(i*bitsPerSlice+j);
						candidatesCount[i]++;
						total2++;
					}
				}
			}
			if (total2 != total)
				total2 = total;
		}
		*/
	}
	cout << "Build list nodes completed" << endl;
}

//______________________________________________________________________________   
string UVoxelFinder::GetCandidatesAsString(const UBits &bits)
{
	// Decodes the candidates in mask as string.
	stringstream ss;
	int numNodes = boxes.size();

	for(int i=0; i<numNodes; ++i)
		if (bits.TestBitNumber(i)) ss << i+1 << " ";
	string result = ss.str();
	return result;
}


//______________________________________________________________________________   
void UVoxelFinder::DisplayListNodes()
{
	char axis[3] = {'X', 'Y', 'Z'};
	// Prints which solids are present in the slices previously elaborated.
	int numNodes = boxes.size();
	int size=8*sizeof(int)*nPerSlice;
	UBits bits(size);

	for (int j = 0; j <= 2; ++j)
	{
		cout << " * " << axis[j] << " axis:" << endl;
		int count = boundaries[j].size();
		for(int i=0; i < count-1; ++i)
		{
			cout << "    Slice #" << i+1 << ": [" << boundaries[j][i] << " ; " << boundaries[j][i+1] << "] -> ";
			bits.Set(size,(const char *)bitmasks[j].allBits+i*nPerSlice*sizeof(int));
			string result = GetCandidatesAsString(bits);
			cout << "[ " << result.c_str() << "]  " << endl;
		}
	}
}

void UVoxelFinder::BuildBoundingBox()
{
	UVector3 sizes, toleranceVector;
	toleranceVector.Set(VUSolid::Tolerance()/100);
	for (int i = 0; i <= 2; ++i)
	{
		double min = boundaries[i].front();
		double max = boundaries[i].back();
		sizes[i] = (max-min)/2;
		boundingBoxCenter[i] = min + sizes[i];
	}
//	sizes -= toleranceVector;
	boundingBox.Set(sizes.x, sizes.y, sizes.z);
}

// algorithm - 

// in order to get balanced voxels, merge should always unite those regions, where the number of voxels is least
// the number

// we will keep sorted list (std::set) with all voxels. there will be comparator function between two voxels,
// which will tell if voxel is less by looking at his right neighbor.
// first, we will add all the voxels into the tree.
// we will be pick the first item in the tree, merging it, adding the right merged voxel into the a list for
// future reduction (bitmasks will be rebuilded later, therefore they need not to be updated).
// the merged voxel need to be added to the tree again, so it's position would be updated

struct VoxelInfo
{
	int count;
	int previous;
	int next;
};

class VoxelComparator
{
public:

	vector<VoxelInfo> &voxels;

	VoxelComparator(vector<VoxelInfo> &_voxels) : voxels(_voxels)
	{

	}

    bool operator()(int l, int r)
	{
		VoxelInfo &lv = voxels[l], &rv = voxels[r];
		int left = lv.count +  voxels[lv.next].count;
		int right = rv.count + voxels[rv.next].count;;
		return (left == right) ? l < r : left < right;
    }
};

void UVoxelFinder::BuildReduceVoxels()
{
	double maxTotal = (double) candidatesCounts[0].size() * candidatesCounts[1].size() * candidatesCounts[2].size();

	if (maxVoxels > 0 && maxVoxels < maxTotal)
	{
		double ratio = (double) maxVoxels / maxTotal;
		ratio = std::pow(ratio, 1./3.);
		if (ratio > 1) ratio = 1;
		reductionRatio.Set(ratio);
	}

	for (int k = 0; k <= 2; ++k)
	{
		vector<int> &candidatesCount = candidatesCounts[k];
		int max = candidatesCount.size();
		vector<VoxelInfo> voxels(max);
		VoxelComparator comp(voxels);
		set<int, VoxelComparator> voxelSet(comp);
//		set<int, VoxelComparator>::iterator it;
		vector<int> mergings;

		for (int j = 0; j < max; ++j)
		{
			VoxelInfo &voxel = voxels[j];
			voxel.count = candidatesCount[j];
			voxel.previous = j - 1;
			voxel.next = j + 1;
			voxels[j] = voxel;
		}
//		voxels[max - 1].count = 99999999999;

		for (int j = 0; j < max - 1; ++j) voxelSet.insert(j); // we go to size-1 to make sure we will not merge the last element

		bool goodEnough = false;
		int count = 0;
		while (true)
		{
			double reduction = reductionRatio[k];
			if (reduction == 0)
				break;
			double currentRatio = 1 - (double) count / max;
			if (currentRatio <= reduction)
				break;
			const int pos = *voxelSet.begin();
			mergings.push_back(pos);

			VoxelInfo &voxel = voxels[pos];
			VoxelInfo &nextVoxel = voxels[voxel.next];

			if (voxelSet.erase(pos) != 1)
				k = k;

			if (voxel.next != max - 1)
				if (voxelSet.erase(voxel.next) != 1)
					k = k;

			if (voxel.previous != -1)
				if (voxelSet.erase(voxel.previous) != 1)
					k = k;

			nextVoxel.count += voxel.count;
			voxel.count = 0;
			nextVoxel.previous = voxel.previous;

			if (voxel.next != max - 1)
				voxelSet.insert(voxel.next);

			if (voxel.previous != -1)
			{
				voxels[voxel.previous].next = voxel.next;
				voxelSet.insert(voxel.previous);
			}
			count++;
		}

//		for (int i = 0; i < max; i++) cout << voxels[i].count << ", ";

		if (mergings.size())
		{
			sort(mergings.begin(), mergings.end());

			vector<double> &boundary = boundaries[k];
			vector<double> reducedBoundary(boundary.size() - mergings.size());
			int skip = mergings[0] + 1, cur = 0, i = 0;
			max = boundary.size();
			for (int j = 0; j < max; ++j)
			{
				if (j != skip)
					reducedBoundary[cur++] = boundary[j];
				else 
					skip = mergings[++i] + 1;
			}
	//		boundaries[k].resize(reducedBoundary.size());
			boundaries[k] = reducedBoundary;
		}
	}
	int total = boundaries[0].size() * boundaries[1].size() * boundaries[2].size();
	cout << "Total number of voxels: " << total << endl;
}

//______________________________________________________________________________   
void UVoxelFinder::Voxelize(vector<VUSolid *> &solids, vector<UTransform3D *> &transforms)
{
	BuildVoxelLimits(solids, transforms);

	BuildBoundaries();

	BuildListNodes();

	BuildBoundingBox();

//	BuildEmpty(); // these does not work well for multi-union, actually only makes performance slower
}

void UVoxelFinder::Voxelize(std::vector<UFacet *> &facets)
{
	BuildVoxelLimits(facets);

	BuildBoundaries();

	BuildListNodes(true);

	BuildReduceVoxels();

	BuildListNodes();

	BuildBoundingBox();

	BuildEmpty();
}


//______________________________________________________________________________       
// "GetCandidates" should compute which solids are possibly contained in
// the voxel defined by the three slices characterized by the passed indexes.
void UVoxelFinder::GetCandidatesVoxel(vector<int> &voxels)
{
	cout << "   Candidates in voxel [" << voxels[0] << " ; " << voxels[1] << " ; " << voxels[2] << "]: ";
	vector<int> candidates;
	int count = GetCandidatesVoxelArray(voxels, candidates);
	cout << "[ ";
	for (int i = 0; i < count; ++i) cout << candidates[i];
	cout << "]  " << endl;
}

inline void findComponents2(unsigned int mask, vector<int> &list, int i)
{
    for (int bit = 0; bit < (int) (8*sizeof(unsigned int)); bit++)
    {
        if (mask & 1)
            list.push_back(8*sizeof(unsigned int)*i+bit);
        if (!(mask >>= 1)) break; // new
    }
}

inline void findComponents3(unsigned int mask, vector<int> &list, int i)
{
    // nejrychlejsi asi bude: ve for cyclu traversovat pres vsechny byty:
    // 1. voendovanou hodnotu pres 0xFF, aby zustal jen posledni byte
    // 2. pokud 0, continue
    // 3. najit prvni nastaveny bit pres lookup table
    // 4. pricist 8*j k nalezenemu bitu, ulozit do seznamu
    // 5. odecist 1 << bit, dokud neni *-*nula pokracovat na 3
    // 6. posunout hodnotu do prava >> 8, pokracovat na 1

    for (int byte = 0; byte < (int) (sizeof(unsigned int)); byte++)
    {
        if (int maskByte = mask & 0xFF)
        {
            do
            {
                static const int firstBits[256] = {
                            8,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            7,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
                            4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0};

                int bit = firstBits[maskByte];
                list.push_back(8*(sizeof(unsigned int)*i+ byte) + bit);
                maskByte -= 1 << bit;
            }
            while (maskByte);
        }
        mask >>= 8;
    }
}

inline void findComponentsFastest(unsigned int mask, vector<int> &list, int i)
{
	for (int byte = 0; byte < (int) (sizeof(unsigned int)); byte++)
	{
		if (int maskByte = mask & 0xFF)
		{
			for (int bit = 0; bit < 8; bit++)
			{
				if (maskByte & 1) 
					list.push_back(8*(sizeof(unsigned int)*i+ byte) + bit);
				if (!(maskByte >>= 1)) break;
			}
		}
		mask >>= 8;
	}
}

//______________________________________________________________________________       
// Method returning the candidates corresponding to the passed point
int UVoxelFinder::GetCandidatesVoxelArray(const UVector3 &point, vector<int> &list, UBits *crossed) const
{
	list.clear();

	for (int i = 0; i <= 2; ++i)
		if(point[i] < boundaries[i].front() || point[i] >= boundaries[i].back()) 
			return 0;

	if (boxes.size() == 1)
	{
		list.push_back(0);
		return 1;
	}
	else
	{
		if (nPerSlice == 1)
		{
			unsigned int mask;
			int slice = UUtils::BinarySearch(boundaries[0], point.x); 
			if (!(mask = ((unsigned int *) bitmasks[0].allBits)[slice]
)) return 0;
			slice = UUtils::BinarySearch(boundaries[1], point.y);
			if (!(mask &= ((unsigned int *) bitmasks[1].allBits)[slice]
)) return 0;
			slice = UUtils::BinarySearch(boundaries[2], point.z);
			if (!(mask &= ((unsigned int *) bitmasks[2].allBits)[slice]
)) return 0;
			if (crossed && (!(mask &= ~((unsigned int *)crossed->allBits)[0]))) return 0;

			findComponentsFastest(mask, list, 0);
		}
		else
		{
			unsigned int *masks[3], mask; // masks for X,Y,Z axis
			for (int i = 0; i <= 2; ++i)
			{
				int slice = UUtils::BinarySearch(boundaries[i], point[i]); 
	//			if (slice < 0 || slice == boundaries[i].size()-1) return 0; // not neccesary anymore
				masks[i] = ((unsigned int *) bitmasks[i].allBits) + slice*nPerSlice;
			}
			unsigned int *maskCrossed = crossed ? (unsigned int *)crossed->allBits : NULL;

			for (int i = 0 ; i < nPerSlice; ++i)
			{
				// Logic "and" of the masks along the 3 axes x, y, z:
				// removing "if (!" and ") continue" => slightly slower
				if (!(mask = masks[0][i])) continue;
				if (!(mask &= masks[1][i])) continue;
				if (!(mask &= masks[2][i])) continue;
				if (maskCrossed && !(mask &= ~maskCrossed[i])) continue;

				findComponentsFastest(mask, list, i);
			}
		}
	}
	return list.size();
}









//______________________________________________________________________________       
// Method returning the candidates corresponding to the passed point
int UVoxelFinder::GetCandidatesVoxelArray(const vector<int> &voxels, vector<int> &list, UBits *crossed) const
{
	list.clear();

	if (boxes.size() == 1)
	{
		list.push_back(0);
		return 1;
	}
	else
	{
		if (nPerSlice == 1)
		{
			unsigned int mask;
			if (!(mask = ((unsigned int *) bitmasks[0].allBits)[voxels[0]]
)) return 0;
			if (!(mask &= ((unsigned int *) bitmasks[1].allBits)[voxels[1]]
)) return 0;
			if (!(mask &= ((unsigned int *) bitmasks[2].allBits)[voxels[2]]
)) return 0;
			if (crossed && (!(mask &= ~((unsigned int *)crossed->allBits)[0]))) return 0;

			findComponentsFastest(mask, list, 0);
		}
		else
		{
			unsigned int *masks[3], mask; // masks for X,Y,Z axis
			for (int i = 0; i <= 2; ++i)
				masks[i] = ((unsigned int *) bitmasks[i].allBits) + voxels[i]*nPerSlice;

			unsigned int *maskCrossed = crossed ? (unsigned int *)crossed->allBits : NULL;

			for (int i = 0 ; i < nPerSlice; ++i)
			{
				// Logic "and" of the masks along the 3 axes x, y, z:
				// removing "if (!" and ") continue" => slightly slower
				if (!(mask = masks[0][i])) continue;
				if (!(mask &= masks[1][i])) continue;
				if (!(mask &= masks[2][i])) continue;
				if (maskCrossed && !(mask &= ~maskCrossed[i])) continue;

				findComponentsFastest(mask, list, i);
			}
		}
	}
	return list.size();
}



bool UVoxelFinder::Contains(const UVector3 &point) const
{
	for (int i = 0; i < 3; ++i)
		if (point[i] < boundaries[i].front() || point[i] > boundaries[i].back())
			return false;
	return true;
}



double UVoxelFinder::DistanceToFirst(const UVector3 &point, const UVector3 &direction) const
{
    UVector3 pointShifted = point - boundingBoxCenter;
    double shift = boundingBox.DistanceToIn(pointShifted, direction);
    return shift;
}

double UVoxelFinder::SafetyFromOutside(const UVector3 &point) const
{
    UVector3 pointShifted = point - boundingBoxCenter;
	double shift = boundingBox.SafetyFromOutside(pointShifted);
    return shift;
}

double UVoxelFinder::DistanceToNext(const UVector3 &point, const UVector3 &direction, const vector<int> &curVoxel) const
{
	double shift = UUtils::kInfinity;
	
	for (int i = 0; i <= 2; ++i)
	{
		// Looking for the next voxels on the considered direction X,Y,Z axis
		const vector<double> &boundary = boundaries[i];
		int cur = curVoxel[i];
		if(direction[i] >= 1e-10)
		{
			// if (point[i] != boundaries[i][binarySearch]) 
			cur++;
			if (cur >= (int) boundary.size())
				continue;
		}
		else 
		{
			if(direction[i] <= 1e-10) 
			{
				if (point[i] == boundary[cur]) 
					if (cur > 0)
						cur--;
					else
						continue;
			}
			else continue;
		}

		double distance = (boundary[cur] - point[i])/direction[i];

		if (shift > distance) 
			shift = distance;
	}

	/*
	if (shift)
	{
		double bonus = VUSolid::Tolerance()/10;
		shift += bonus;

		if (index == 0)
		{
			point.x = boundary;
			point.y += shift * direction.y;
			point.z += shift * direction.z;
		}
		if (index == 1)
		{
			point.y = boundary;
			point.x += shift * direction.x;
			point.z += shift * direction.z;
		}
		if (index == 2)
		{
			point.z = boundary;
			point.x += shift * direction.x;
			point.y += shift * direction.y;
		}
	}
	*/

	return shift;
}

bool UVoxelFinder::UpdateCurrentVoxel(const UVector3 &point, const UVector3 &direction, vector<int> &curVoxel) const
{
	for (int i = 0; i <= 2; ++i)
	{
		int index = curVoxel[i];
		const vector<double> &boundary = boundaries[i];

		if (direction[i] > 0) 
		{
			if (point[i] >= boundary[++index]) 
				if (++curVoxel[i] > (int) boundary.size())
					return false;
		}
		else
		{
			if (point[i] < boundary[index]) 
				if (--curVoxel[i] < 0) 
					return false;
		}
#ifdef DEBUG
		int indexOK = UUtils::BinarySearch(boundary, point[i]);
		if (curVoxel[i] != indexOK)
			curVoxel[i] = indexOK; // put breakpoint here
#endif
	}
	return true;
}

UVoxelCandidatesIterator::UVoxelCandidatesIterator(const UVoxelFinder &f, const UVector3 &point) : nextAvailable(true), curInt(-1), curBit((int) (8*sizeof(unsigned int)))
{
	carNodes = f.boxes.size();
	if (carNodes > 1)
	{
		n = 1+(carNodes-1)/(8*sizeof(unsigned int));
		int sliceX = UUtils::BinarySearch(f.boundaries[0], point.x); 
		int sliceY = UUtils::BinarySearch(f.boundaries[1], point.y);
		int sliceZ = UUtils::BinarySearch(f.boundaries[2], point.z);

		unsigned int *maskX = ((unsigned int *) f.bitmasks[0].allBits) + sliceX*n;
		unsigned int *maskXLeft = (sliceX && point.x == f.boundaries[0][sliceX]) ? maskX - n : NULL;

		unsigned int *maskY = ((unsigned int *) f.bitmasks[1].allBits) + sliceY*n;
		unsigned int *maskYLeft = (sliceY && point.y == f.boundaries[1][sliceY]) ? maskY - n : NULL;
		unsigned int *maskZ = ((unsigned int *) f.bitmasks[2].allBits) + sliceZ*n;
		unsigned int *maskZLeft = (sliceZ && point.z == f.boundaries[2][sliceZ]) ? maskZ - n  : NULL;

		if (sliceX == -1 || sliceX == f.boundaries[0].back()) nextAvailable = false;
		if (sliceY == -1 || sliceY == f.boundaries[1].back()) nextAvailable = false;
		if (sliceZ == -1 || sliceZ == f.boundaries[2].back()) nextAvailable = false;
	}
	else
	{
		if (point.x < f.boundaries[0].front() || point.x > f.boundaries[0].back()) nextAvailable = false;

		if (point.y < f.boundaries[1].front() || point.y > f.boundaries[1].back()) nextAvailable = false;

		if (point.z < f.boundaries[2].front() || point.z > f.boundaries[2].back()) nextAvailable = false;
	}
}

int UVoxelCandidatesIterator::Next()
{
	if (!nextAvailable) return -1;

	if (carNodes == 1)
	{
		nextAvailable = false;
		return 0;
	}
	else
	{
		do
		{
			while (curBit == (int) (8*sizeof(unsigned int)))
			{
				if (curInt++ >= n)
				{
					nextAvailable = false;
					return -1;
				}
				// Logic "and" of the masks along the 3 axes x, y, z:
				// removing "if (!" and ") continue" => slightly slower
				if (!(mask = maskXLeft ? maskX[curInt] | maskXLeft[curInt] : maskX[curInt]) ||
					!(mask &= maskYLeft ? maskY[curInt] | maskYLeft[curInt] : maskY[curInt]) ||
					!(mask &= maskZLeft ? maskZ[curInt] | maskZLeft[curInt] : maskZ[curInt]))
				{
					continue;
				}
				curBit = 0;
			}
			int shifted = 1 << curBit; // new
			if (mask & shifted)
			{
				int res = 8*sizeof(unsigned int)*curInt+curBit;
				if (!(mask -= shifted))
					curBit = (int) (8*sizeof(unsigned int));
					
				return res;
			}
			curBit++;
		}
		while (nextAvailable);
		return -1;
	}
}

void UVoxelFinder::SetMaxVoxels(int max)
{
	maxVoxels = max;
	reductionRatio.Set(0);
}

void UVoxelFinder::SetMaxVoxels(UVector3 &ratioOfReduction)
{
	maxVoxels = -1;
	reductionRatio = ratioOfReduction;
}
