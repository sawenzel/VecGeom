
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

#include "VUSolid.hh" 
#include "UVoxelFinder.hh"
#include "UUtils.hh"
#include "UOrb.hh"

using namespace std;

//______________________________________________________________________________   
UVoxelFinder::UVoxelFinder()
{
	boundingBox = NULL;
}

//______________________________________________________________________________   
UVoxelFinder::~UVoxelFinder()
{
	if (boundingBox) delete boundingBox;
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

		for (int j = 0; j <= 2; ++j)
		{
			CreateSortedBoundary(sortedBoundary, j);
			vector<double> &boundary = boundaries[j];
			boundary.clear();

			for(int i = 0 ; i < 2*numNodes; ++i)
			{
				double newBoundary = sortedBoundary[i];
				int size = boundary.size();
				if(!size || abs(boundary[size-1] - newBoundary) > tolerance)
					boundary.push_back(newBoundary);
				else // If two successive boundaries are too close from each other, only the first one is considered 	
					cout << "Skipping boundary [" << j << "] : " << i << endl;
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

void UVoxelFinder::BuildListNodes()
{
	// "BuildListNodes" stores in the bitmasks solids present in each slice along an axis.
//	const double localTolerance = 0;
	int numNodes = boxes.size();
	int bitsPerSlice = GetBitsPerSlice();

	for (int k = 0; k < 3; k++)
	{
		int boundariesCount = boundaries[k].size();
		UBits &bitmask = bitmasks[k];
		bitmask.Clear();
		bitmask.SetBitNumber((boundariesCount-1)*bitsPerSlice-1, false); // it is here so we can set the maximum number of bits. this line will rellocate the memory and set all to zero

		for(int i = 0 ; i < boundariesCount-1; ++i)
		{
			// Loop on the nodes, number of slices per axis
			for(int j = 0 ; j < numNodes; j++)
			{
				// Determination of the minimum and maximum position along x
				// of the bounding boxe of each node:
				double p = boxes[j].pos[k], d = boxes[j].hlen[k];
				double leftBoundary = boundaries[k][i];
				double rightBoundary = boundaries[k][i+1];
				double min = p - d; // - localTolerance;
				double max = p + d; // + localTolerance;
				if (min < rightBoundary && max > leftBoundary)
					// Storage of the number of the node in the array:
					bitmask.SetBitNumber(i*bitsPerSlice+j);
			}
		}
	}
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
	if (boundingBox) delete boundingBox;

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
	boundingBox = new UBox("VoxelBoundingBox", sizes.x, sizes.y, sizes.z);
}

//______________________________________________________________________________   
void UVoxelFinder::Voxelize(vector<VUSolid *> &solids, vector<UTransform3D *> &transforms)
{
	BuildVoxelLimits(solids, transforms);
	BuildBoundaries();
	BuildListNodes();
	BuildBoundingBox();
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
    double shift = boundingBox->DistanceToIn(pointShifted, direction);
    return shift;
}

double UVoxelFinder::DistanceToNext(const UVector3 &point, const UVector3 &direction, const vector<int> &curVoxel) const
{
	double shift = UUtils::kInfinity;
	
	for (int i = 0; i <= 2; ++i)
	{
		// Looking for the next voxels on the considered direction X,Y,Z axis
		const vector<double> &boundary = boundaries[i];
		int binarySearch = curVoxel[i];
		if(direction[i] >= 1e-10)
		{
			// if (point[i] != boundaries[i][binarySearch]) 
			binarySearch++;
			if (binarySearch >= (int) boundary.size())
				continue;
		}
		else 
		{
			if(direction[i] <= 1e-10) 
			{
				if (point[i] == boundary[binarySearch]) 
					if (binarySearch > 0)
						binarySearch--;
					else
						continue;
			}
			else continue;
		}

		double distance = (boundary[binarySearch] - point[i])/direction[i];

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
