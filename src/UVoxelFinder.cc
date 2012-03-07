
#include "UVoxelFinder.hh"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include "UUtils.hh"
#include <sstream>
#include <algorithm>

#include "VUSolid.hh" 

using namespace std;

#define boundariesCountX boundariesCounts[0]
#define boundariesCountY boundariesCounts[1]
#define boundariesCountZ boundariesCounts[2]

#define bitmaskX bitmasks[0]
#define bitmaskY bitmasks[1]
#define bitmaskZ bitmasks[2]

#define boundariesX boundaries[0]
#define boundariesY boundaries[1]
#define boundariesZ boundaries[2]

//______________________________________________________________________________   
UVoxelFinder::UVoxelFinder()
{
	boundariesCountX = boundariesCountY = boundariesCountZ = 0;
	boundingBox = NULL;
}

//______________________________________________________________________________   
UVoxelFinder::~UVoxelFinder()
{
	if (boundingBox)
	{
		delete boundingBox;
		boundingBox = NULL;
	}
}

//______________________________________________________________________________   
void UVoxelFinder::BuildVoxelLimits(std::vector<VUSolid *> &solids, std::vector<UTransform3D *> &transforms)
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

		for (int i = 0; i < numNodes; i++)   
		{
			VUSolid &solid = *solids[i];
			UTransform3D &transform = *transforms[i];
			UVector3 min, max;
			solid.Extent(min, max);
			min -= toleranceVector; max += toleranceVector;
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
	for(int i = 0; i < numNodes; i++)
	{
		UVector3 &hlen = boxes[i].hlen;
		UVector3 &pos = boxes[i].pos;
		cout << setw(10) << setiosflags(ios::fixed) << setprecision(16) <<
		"    -> Node " << i+1 <<  ":\n" << 
		"\t * dX = " << hlen.x << 
		" ; * dY = " << hlen.y << 
		" ; * dZ = " << hlen.z << "\n" << 
		"\t * pX = " << pos.x << 
		" ; * pY = " << pos.y << 
		" ; * pZ = " << pos.z << "\n";
	}
}

//______________________________________________________________________________   
void UVoxelFinder::CreateSortedBoundary(vector<double> &boundary, int axis)
{
	// "CreateBoundaries"'s aim is to determine the slices induced by the bounding boxes,
	// along each axis. The created boundaries are stored in the array "boundariesRaw"
	int numNodes = boxes.size(); // Number of nodes in structure of "UMultiUnion" type
	// Determination of the boundries along x, y and z axis:
	for(int i = 0 ; i < numNodes; i++)   
	{
		// For each node, the boundaries are created by using the array "boxes"
		// built in method "BuildVoxelLimits":
		double p = boxes[i].pos[axis], d = boxes[i].hlen[axis];
		// x boundaries
		boundary[2*i] = p - d;
		boundary[2*i+1] = p + d;
	}
	std::sort(boundary.begin(), boundary.end());
}

void UVoxelFinder::BuildOptimizedBoundaries()
{
	// "SortBoundaries" orders the boundaries along each axis (increasing order)
	// and also does not take into account redundant boundaries, ie if two boundaries
	// are separated by a distance strictly inferior to "tolerance".
	// The sorted boundaries are respectively stored in:
	//              * boundariesX
	//              * boundariesY
	//              * boundariesZ
	// In addition, the number of elements contained in the three latter arrays are
	// precised thanks to variables: boundariesCountX, boundariesCountY and boundariesCountZ.

	if (int numNodes = boxes.size())
	{
		const double tolerance = VUSolid::Tolerance() / 100.0; // Minimal distance to discrminate two boundaries.
		vector<double> sortedBoundary(2*numNodes);

		for (int j = 0; j < 3; j++) 
		{
			CreateSortedBoundary(sortedBoundary, j);
			vector<double> &boundary = boundaries[j];
			boundary.clear();

			for(int i = 0 ; i < 2*numNodes; i++)
			{
				double newBoundary = sortedBoundary[i];
				int size = boundary.size();
				if(!size || std::abs(boundary[size-1] - newBoundary) > tolerance)
					boundary.push_back(newBoundary);
				else // If two successive boundaries are too close from each other, only the first one is considered 	
					cout << "Skipping boundary [" << j << "] : " << i << endl;
			}
			boundariesCounts[j] = boundary.size();
		}
	}
}

void UVoxelFinder::DisplayBoundaries()
{
	char axis[3] = {'X', 'Y', 'Z'};
	for (int i = 0; i < 3; i++)
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
	for(int i = 0; i < count; i++)
	{
		cout << setw(10) << setiosflags(ios::fixed) << setprecision(16) << boundaries[i];
		//      printf("%19.15f ",boundariesX[i]);      
		//      cout << boundariesX[i] << " ";
		if(i != count-1) cout << "-> ";
	}
	cout << "|" << endl << "Number of boundaries: " << count << endl;
}

/*
inline bool between(double what, double min, double max)
{
	return what >= min && what <= max;
}
*/

void UVoxelFinder::BuildListNodes()
{
	// "BuildListNodes" stores in the bitmasks solids present in each slice along an axis.
//	const double localTolerance = 0;
	int numNodes = boxes.size();
	int bitsPerSlice = GetBitsPerSlice();

	for (int k = 0; k < 3; k++)
	{
		int boundariesCount = boundariesCounts[k];
		UBits &bitmask = bitmasks[k];
		bitmask.Clear();
		bitmask.SetBitNumber((boundariesCount-1)*bitsPerSlice-1, false); // it is here so we can set the maximum number of bits. this line will rellocate the memory and set all to zero

		for(int i = 0 ; i < boundariesCount-1; i++)
		{
			// Loop on the nodes, number of slices per axis
			for(int j = 0 ; j < numNodes; j++)
			{
				// Determination of the minimum and maximum position along x
				// of the bounding boxe of each node:
				double p = boxes[j].pos[k], d = boxes[j].hlen[k];
				double leftBoundary = boundaries[k][i];
				double rightBoundary = boundaries[k][i+1];
//				double rightBoundary = (i < boundariesCount) - 1 ? boundaries[k][i+1] : leftBoundary;

				double min = p - d; // - localTolerance;
				double max = p + d; // + localTolerance;

				// Is the considered node inside slice?
//				if (min > rightBoundary + localTolerance) ||
//				   (max < leftBoundary - localTolerance) continue;

				if (min < rightBoundary && max > leftBoundary)
					// Storage of the number of the node in the array:
					bitmask.SetBitNumber(i*bitsPerSlice+j);
			}
		}
	}
}

//______________________________________________________________________________   
void UVoxelFinder::GetCandidatesAsString(const UBits &bits, string &result)
{
	// Decodes the candidates in mask as string.
	stringstream ss;
	int numNodes = boxes.size();

	for(int i=0; i<numNodes; i++)
		if (bits.TestBitNumber(i)) ss << i+1 << " ";
	result = ss.str();
}

void UVoxelFinder::DisplayListNodes(vector<double> &boundaries, UBits &bitmask)
{
	// Prints which solids are present in the slices previously elaborated.
	int numNodes = boxes.size();
	string result = "";
	int size=8*sizeof(int)*nPerSlice;
	UBits bits(size);

	int count = boundaries.size();
	for(int i=0; i < count-1; i++)
	{
		cout << "    Slice #" << i+1 << ": [" << boundaries[i] << " ; " << boundaries[i+1] << "] -> ";
		bits.Set(size,(const char *)bitmask.allBits+i*nPerSlice*sizeof(int));
		GetCandidatesAsString(bits, result);
		cout << "[ " << result.c_str() << "]  " << endl;
	}
}


//______________________________________________________________________________   
void UVoxelFinder::DisplayListNodes()
{
	char axis[3] = {'X', 'Y', 'Z'};
	for (int i = 0; i < 3; i++)
	{
		cout << " * " << axis[i] << " axis:" << endl;
		DisplayListNodes(boundaries[i], bitmasks[i]);
	}
}

//______________________________________________________________________________   
void UVoxelFinder::Voxelize(std::vector<VUSolid *> &solids, std::vector<UTransform3D *> &transforms)
{
	BuildVoxelLimits(solids, transforms);

	BuildOptimizedBoundaries();

	BuildListNodes();

	if (boundingBox)
	{
		delete boundingBox;
		boundingBox = NULL;
	}

	UVector3 min, max, sizes;
	UVector3 toleranceVector;
	toleranceVector.Set(VUSolid::Tolerance()/100);
	for (int i = 0; i < 3; i++)
	{
		min[i] = boundaries[i][0];
		max[i] = boundaries[i][boundariesCounts[i]-1];
		sizes[i] = (max[i]-min[i])/2;
		boundingBoxCenter[i] = min[i] + sizes[i];
	}
//	sizes -= toleranceVector;
	boundingBox = new UBox("VoxelBoundingBox", sizes.x, sizes.y, sizes.z);
}

//______________________________________________________________________________       
void UVoxelFinder::GetCandidatesVoxel(int indexX, int indexY, int indexZ)
{
	// "GetCandidates" should compute which solids are possibly contained in
	// the voxel defined by the three slices characterized by the passed indexes.

	string result = "";  
	int numNodes = boxes.size();

	// Voxelized structure:      
	UBits bits;
	for(int i = 0 ; i < nPerSlice ; i++)
		bits = bitmaskX[(indexX-1)+i] & bitmaskY[nPerSlice*(indexY-1)+i] & bitmaskZ[(indexZ-1)+i];

	GetCandidatesAsString(bits,result);
	cout << "   Candidates in voxel [" << indexX << " ; " << indexY << " ; " << indexZ << "]: ";
	cout << "[ " << result.c_str() << "]  " << endl;
}


//TODO: delete this method, does not work anyway anymore, because we changed bits format
int UVoxelFinder::GetCandidatesVoxelBits(const UVector3 &point, UBits &bits) const
{
	vector<int> list;
	bits.ResetAllBits();
	if (boxes.size() == 1)
	{
		if(boundariesCountX && (point.x < boundariesX[0] || point.x > boundariesX[1])) return 0;

		if(boundariesCountY && (point.y < boundariesY[0] || point.y > boundariesY[1])) return 0;

		if(boundariesCountZ && (point.z < boundariesZ[0] || point.z > boundariesZ[1])) return 0;

		bits.SetBitNumber(0);
		return 1;
	}
	else
	{
		int sliceX = UUtils::BinarySearch(boundariesX, point.x); 
		if (sliceX == -1 || sliceX == boundariesCountX-1) return 0;

		int sliceY = UUtils::BinarySearch(boundariesY, point.y);
		if (sliceY == -1 || sliceY == boundariesCountY-1) return 0;

		int sliceZ = UUtils::BinarySearch(boundariesZ, point.z);     
		if (sliceZ == -1 || sliceZ == boundariesCountZ-1) return 0;

		bits = (sliceX && point.x == boundariesX[sliceX]) ? bitmaskX[sliceX] |  bitmaskX[sliceX-1] : bitmaskX[sliceX];

		bits &= (sliceY && point.y == boundariesY[sliceY]) ? bitmaskY[sliceY] |  bitmaskY[sliceY-1] : bitmaskY[sliceY];

		bits &= (sliceZ && point.z == boundariesZ[sliceZ]) ? bitmaskZ[sliceZ] |  bitmaskZ[sliceZ-1] : bitmaskZ[sliceZ];

	}
	return bits.CounUBits();
}





//______________________________________________________________________________       
// Method returning the candidates corresponding to the passed point
int UVoxelFinder::GetCandidatesVoxelArray(const UVector3 &point, vector<int> &list, UBits *crossed) const
{
	list.clear();
	if (boxes.size() == 1)
	{
		if(boundariesCountX && (point.x < boundariesX[0] || point.x > boundariesX[1])) return 0;

		if(boundariesCountY && (point.y < boundariesY[0] || point.y > boundariesY[1])) return 0;

		if(boundariesCountZ && (point.z < boundariesZ[0] || point.z > boundariesZ[1])) return 0;

		list.push_back(0);
		return 1;
	}
	else
	{
		int sliceX = UUtils::BinarySearch(boundariesX, point.x); 
		if (sliceX < 0 || sliceX == boundariesCountX-1) return 0;

		unsigned int *maskX = ((unsigned int *) bitmaskX.allBits) + sliceX*nPerSlice;
		if (nPerSlice == 1 && !maskX[0]) return 0;

		int sliceY = UUtils::BinarySearch(boundariesY, point.y);
		if (sliceY < 0 || sliceY == boundariesCountY-1) return 0;

		unsigned int *maskY = ((unsigned int *) bitmaskY.allBits) + sliceY*nPerSlice;
		if (nPerSlice == 1 && !maskY[0]) return 0;

		int sliceZ = UUtils::BinarySearch(boundariesZ, point.z);
		if (sliceZ < 0 || sliceZ == boundariesCountZ-1) return 0;

		unsigned int *maskZ = ((unsigned int *) bitmaskZ.allBits) + sliceZ*nPerSlice;

		unsigned int *maskCrossed = crossed ? (unsigned int *)crossed->allBits : NULL;

		for (int i = 0 ; i < nPerSlice; i++)
		{
			unsigned int mask;
			// Logic "and" of the masks along the 3 axes x, y, z:
			// removing "if (!" and ") continue" => slightly slower
			if (!(mask = maskZ[i])) continue;
			if (!(mask &= maskY[i])) continue;
			if (!(mask &= maskX[i])) continue;
			if (maskCrossed && !(mask &= ~maskCrossed[i])) continue;

			/*
            if (false)
            {
                for (int bit = 0; bit < (int) (8*sizeof(unsigned int)); bit++)
                {
                    if (mask & 1)
                    {
                        list.push_back(8*sizeof(unsigned int)*i+bit);
                    }
                    if (!(mask >>= 1)) break; // new
                }
            }
            else
			*/
            {
                // nejrychlejsi asi bude: ve for cyclu traversovat pres vsechny byty:
                // 1. voendovanou hodnotu pres 0xFF, aby zustal jen posledni byte
                // 2. pokud 0, continue
                // 3. najit prvni nastaveny bit pres lookup table
                // 4. pricist 8*j k nalezenemu bitu, ulozit do seznamu
                // 5. odecist 1 << bit, dokud neni *-*nula pokracovat na 3
                // 6. posunout hodnotu do prava >> 8, pokracovat na 1

				/*
                if (false)
                {
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
                else
				*/

                {
                    for (int byte = 0; byte < (int) (sizeof(unsigned int)); byte++)
                    {
                        if (int maskByte = mask & 0xFF)
                        {
                            for (int bit = 0; bit < 8; bit++)
                            {
                                if (maskByte & 1) list.push_back(8*(sizeof(unsigned int)*i+ byte) + bit);
                                if (!(maskByte >>= 1)) break;
                            }
                        }
                        mask >>= 8;
                    }
                }
            }
        }
	}
	return list.size();
}


/*
//______________________________________________________________________________       
// Method returning the candidates corresponding to the passed point
int UVoxelFinder::GetCandidatesVoxelArray(const UVector3 &point, vector<int> &list, UBits *crossed) const
{
	list.clear();
	if (boxes.size() == 1)
	{
		if(boundariesCountX && (point.x < boundariesX[0] || point.x > boundariesX[1])) return 0;

		if(boundariesCountY && (point.y < boundariesY[0] || point.y > boundariesY[1])) return 0;

		if(boundariesCountZ && (point.z < boundariesZ[0] || point.z > boundariesZ[1])) return 0;

		list.push_back(0);
		return 1;
	}
	else
	{
		int n = GetNPerSlice();
		double tolerance = 1.000000*VUSolid::Tolerance();

		int sliceX = UUtils::BinarySearch(boundariesCountX, boundariesX, point.x); 
		if (sliceX == -1 || sliceX == boundariesCountX-1) return 0;

		int sliceY = UUtils::BinarySearch(boundariesCountY, boundariesY, point.y);
		if (sliceY == -1 || sliceY == boundariesCountY-1) return 0;

		int sliceZ = UUtils::BinarySearch(boundariesCountZ, boundariesZ, point.z);
		if (sliceZ == -1 || sliceZ == boundariesCountZ-1) return 0;

		unsigned int *maskX, *maskY, *maskZ, *maskXLeft, *maskYLeft, *maskZLeft; 
		maskX = ((unsigned int *) bitmaskX.allBits) + sliceX*n;
		maskXLeft = (sliceX && std::abs(point.x - boundariesX[sliceX]) < tolerance) ? maskX - n : NULL;
		if (n == 1 && !maskX[0] && !maskXLeft) return 0;

		maskY = ((unsigned int *) bitmaskY.allBits) + sliceY*n;
		maskYLeft = (sliceY && std::abs(point.y - boundariesY[sliceY]) < tolerance) ? maskY - n : NULL;
		if (n == 1 && !maskY[0] && !maskYLeft) return 0;

		maskZ = ((unsigned int *) bitmaskZ.allBits) + sliceZ*n;
		maskZLeft = (sliceZ && std::abs(point.z - boundariesZ[sliceZ]) < tolerance) ? maskZ - n  : NULL;
		if (n == 1 && !maskZ[0] && !maskZLeft) return 0;

		maskXLeft = maskYLeft = maskZLeft = 0;

		unsigned int *maskCrossed = crossed ? (unsigned int *)crossed->allBits : NULL;

		for (int i = 0 ; i < n; i++)
		{
			unsigned int mask;
			// Logic "and" of the masks along the 3 axes x, y, z:
			// removing "if (!" and ") continue" => slightly slower
			if (!(mask = maskXLeft ? maskX[i] | maskXLeft[i] : maskX[i])) continue;
			if (!(mask &= maskYLeft ? maskY[i] | maskYLeft[i] : maskY[i])) continue;
			if (!(mask &= maskZLeft ? maskZ[i] | maskZLeft[i] : maskZ[i])) continue;
			if (maskCrossed && !(mask &= maskCrossed[i])) continue;

			if (false)
			{
				for (int bit = 0; bit < (int) (8*sizeof(unsigned int)); bit++)
				{
					if (mask & 1)
					{
						list.push_back(8*sizeof(unsigned int)*i+bit);
						if (!(mask >>= 1)) break; // new
					}
					else mask >>= 1;
				}
			}
			else
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
						if (false)
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

								for (int bit = 0; bit < 8; bit++)
								{
									if (maskByte & 1)
									{
										list.push_back(8*sizeof(unsigned int)*i+bit);
										if (!(mask >>= 1)) break; // new
									}
									else mask >>= 1;
								}
							}
							while (maskByte);
						}
						else
						{
							for (int bit = 0; bit < 8; bit++)
							{
								if (maskByte & 1)
								{
									list.push_back(8*(sizeof(unsigned int)*i+ byte) + bit);
									if (!(maskByte >>= 1)) break; // new
								}
								else maskByte >>= 1;
							}
						}

					}
					mask >>= 8;
				}
			}
		}
	}
	return list.size();
}
*/

//______________________________________________________________________________       
const vector<UVoxelBox> &UVoxelFinder::GetBoxes() const
{
	return boxes;
}

//______________________________________________________________________________     



bool UVoxelFinder::Contains(const UVector3 &point) const
{
	if (point.x < boundariesX[0]) 
		return false;
	if (point.y < boundariesY[0]) 
		return false;
	if (point.z < boundariesZ[0]) 
		return false;

	if (point.x > boundariesX[boundariesCountX - 1])
		return false;
	if (point.y > boundariesY[boundariesCountY - 1])
		return false;
	if (point.z > boundariesZ[boundariesCountZ - 1])
		return false;

	return true;
}



double UVoxelFinder::DistanceToFirst(UVector3 &point, const UVector3 &direction) const
{
    UVector3 pointShifted = point - boundingBoxCenter;
    double shift = boundingBox->DistanceToIn(pointShifted, direction);

    /*
    for (int i = 0; i < 3; i++)
    {
        double last = boundaries[i][boundariesCounts[i]-1];
        double rounding = last - point[i];
        if (rounding > 0 && rounding < VUSolid::Tolerance()/100)
            point[i] = last;
    }
    */

    return shift;

    /*
    // check weather point is outside the voxelized area
    if (!Contains(point))
    {
        int incDir[3];

        // X,Y,Z axis
        for (int i = 0; i < 3; i++)
        {
            if(std::abs(direction[i]) >= 1e-10) incDir[i] = (direction[i] > 0) ? 1 : -1;
            else incDir[i] = 0;

            if( (point[i] < boundaries[i][0] && incDir[i] < 0) || point[i] > boundaries[i][boundariesCounts[i]-1] && incDir[i] > 0)
                // we have found that with given direction will never reach voxel area
                return UUtils::kInfinity;
        }

        double distance, shift = UUtils::kInfinity;
        for (int i = 0; i < 3; i++)
        {
            // Looking for the first voxel on the considered direction
            if (point[i] < boundaries[i][0] && incDir[i] > 0)
            {
                distance = (boundaries[i][0] - point[i])/direction[i];
            }
            else if (point[i] > boundaries[i][boundariesCounts[i] - 1] && incDir[i] < 0)
            {
                distance = (boundaries[i][boundariesCounts[i] - 1] - point[i])/direction[i];
            }
//			else distance = UUtils::kInfinity;

            if (shift > distance) shift = distance;
        }
        return shift + VUSolid::Tolerance();
    }
    else return 0;
    */
}

double UVoxelFinder::DistanceToNext(UVector3 &point, const UVector3 &direction) const
{
	double distance, shift = UUtils::kInfinity;
	
	// Looking for the next voxels on the considered direction
	// X,Y,Z axis
	distance = UUtils::kInfinity;
//	int index = -1;
//	double boundary;
	for (int i = 0; i < 3; i++)
	{
		int incDir;
		if(std::abs(direction[i]) >= 1e-10) incDir = (direction[i] > 0) ? 1 : -1;
		else incDir = 0;
		int binarySearch = UUtils::BinarySearch(boundaries[i], point[i]);
		if (point[i] > boundaries[i][boundariesCounts[i] - 1] || point[i] < boundaries[i][0])
			return UUtils::kInfinity;

		if(incDir > 0)
		{
			// if (point[i] != boundaries[i][binarySearch]) 
			binarySearch++;
		}
		else if(incDir < 0 && point[i] == boundaries[i][binarySearch]) binarySearch--;

		if (incDir != 0) distance = (boundaries[i][binarySearch] - point[i])/direction[i];

		if (shift > distance) 
		{
			shift = distance;
//			boundary = boundaries[i][binarySearch];
//			index = i;
		}
	}

	if (shift)
	{
		double bonus = VUSolid::Tolerance()/10;
		shift += bonus;

		/*
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
		*/
	}

	return shift;
}

UVoxelCandidatesIterator::UVoxelCandidatesIterator(const UVoxelFinder &f, const UVector3 &point) : nextAvailable(true), curInt(-1), curBit((int) (8*sizeof(unsigned int)))
{
	carNodes = f.boxes.size();
	if (carNodes > 1)
	{
		n = 1+(carNodes-1)/(8*sizeof(unsigned int));
		int sliceX = UUtils::BinarySearch(f.boundariesX, point.x); 
		int sliceY = UUtils::BinarySearch(f.boundariesY, point.y);
		int sliceZ = UUtils::BinarySearch(f.boundariesZ, point.z);

		unsigned int *maskX = ((unsigned int *) f.bitmaskX.allBits) + sliceX*n;
		unsigned int *maskXLeft = (sliceX && point.x == f.boundariesX[sliceX]) ? maskX - n : NULL;

		unsigned int *maskY = ((unsigned int *) f.bitmaskY.allBits) + sliceY*n;
		unsigned int *maskYLeft = (sliceY && point.y == f.boundariesY[sliceY]) ? maskY - n : NULL;
		unsigned int *maskZ = ((unsigned int *) f.bitmaskZ.allBits) + sliceZ*n;
		unsigned int *maskZLeft = (sliceZ && point.z == f.boundariesZ[sliceZ]) ? maskZ - n  : NULL;

		if (sliceX == -1 || sliceX == f.boundariesCounts[0]-1) nextAvailable = false;
		if (sliceY == -1 || sliceY == f.boundariesCounts[1]-1) nextAvailable = false;
		if (sliceZ == -1 || sliceZ == f.boundariesCounts[2]-1) nextAvailable = false;
	}
	else
	{
		if(f.boundariesCountX && (point.x < f.boundariesX[0] || point.x > f.boundariesX[1])) nextAvailable = false;

		if(f.boundariesCountY && (point.y < f.boundariesY[0] || point.y > f.boundariesY[1])) nextAvailable = false;

		if(f.boundariesCountZ && (point.z < f.boundariesZ[0] || point.z > f.boundariesZ[1])) nextAvailable = false;
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
