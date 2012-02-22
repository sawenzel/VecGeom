#include "UVoxelFinder.hh"
#include "UMultiUnion.hh"

#include <iostream>
#include "UUtils.hh"
#include <sstream>

using namespace std;

//______________________________________________________________________________       
UMultiUnion::UMultiUnion(const char *name)
{
	SetName(name);
	solids.clear();
	transforms.clear();
}

//______________________________________________________________________________       
UMultiUnion::~UMultiUnion()
{
}
 
//______________________________________________________________________________       
void UMultiUnion::AddNode(VUSolid &solid, UTransform3D &trans)
{
	solids.push_back(&solid);
	transforms.push_back(&trans);
}

//______________________________________________________________________________       
double UMultiUnion::Capacity()
{
	// Capacity computes the cubic volume of the "UMultiUnion" structure using random points

	// Random initialization:
//	srand((unsigned int)time(NULL));

	double extentMin[3], extentMax[3];
	UVector3 point;
	double dX, dY, dZ, pX, pY, pZ;
	int generated = 0, inside = 0;

	Extent(extentMin,extentMax);

	dX = (extentMax[0] - extentMin[0])/2;
	dY = (extentMax[1] - extentMin[1])/2;
	dZ = (extentMax[2] - extentMin[2])/2;

	pX = (extentMax[0] + extentMin[0])/2;
	pY = (extentMax[1] + extentMin[1])/2;
	pZ = (extentMax[2] + extentMin[2])/2;     

	double vbox = (2*dX)*(2*dY)*(2*dZ);

	while (inside < 100000)
	{
		point.Set(pX - dX + 2*dX*(rand()/(double)RAND_MAX),
			pY - dY + 2*dY*(rand()/(double)RAND_MAX),
			pZ - dZ + 2*dZ*(rand()/(double)RAND_MAX));
		generated++;
		if(Inside(point) != eOutside) inside++;
	}
	double capacity = inside*vbox/generated;
	return capacity;      
}

//______________________________________________________________________________
void UMultiUnion::ComputeBBox (UBBox * /*aBox*/, bool /*aStore*/)
{
	// Computes bounding box.
	cout << "ComputeBBox - Not implemented" << endl;
}   

//______________________________________________________________________________
double UMultiUnion::DistanceToInDummy(const UVector3 &aPoint, const UVector3 &aDirection, // UVector3 &aNormal, 
	double aPstep) const
{
	UVector3 direction = aDirection.Unit();   
	UVector3 localPoint, localDirection;
	double minDistance = UUtils::kInfinity;

	int numNodes = solids.size();
	for (int i = 0 ; i < numNodes ; i++)
	{
		VUSolid &solid = *solids[i];
		UTransform3D &transform = *transforms[i];

		localPoint = transform.LocalPoint(aPoint);
		localDirection = transform.LocalVector(direction);                

		double distance = solid.DistanceToIn(localPoint, localDirection, aPstep);
		if (minDistance > distance) minDistance = distance;
	}
	return minDistance;
}



double UMultiUnion::DistanceToInCandidates(const UVector3 &aPoint, const UVector3 &direction, double aPstep, std::vector<int> &candidates, UBits &bits) const
{
	int candidatesCount = candidates.size();
	UVector3 localPoint, localDirection;

	double minDistance = UUtils::kInfinity;   
	for (int i = 0 ; i < candidatesCount; i++)
	{
		int candidate = candidates[i];
		VUSolid &solid = *solids[candidate];
		UTransform3D &transform = *transforms[candidate];

		localPoint = transform.LocalPoint(aPoint);
		localDirection = transform.LocalVector(direction);                
		double distance = solid.DistanceToIn(localPoint, localDirection, aPstep);
		if (minDistance > distance) minDistance = distance;
		bits.ResetBitNumber(candidate);
	}
	return minDistance;
}


// we have to look also for all other objects in next voxels, if the distance is not shorter ... we have to do it because,
// for example for objects which starts in first voxel in which they
// do not collide with direction line, but in second it collides...
// The idea of crossing voxels would be still applicable,
// because this way we could exclude from the testing such solids,
// which were found that obviously are not good candidates, because
// they would return infinity
// But if distance is smaller than the shift to next voxel, we can return it immediately

double UMultiUnion::DistanceToIn(const UVector3 &aPoint, 
	const UVector3 &aDirection, /*UVector3 &aNormal,*/ double aPstep) const
{
//	return DistanceToInDummy(aPoint, aDirection, aPstep);

	UVector3 direction = aDirection.Unit();
	double shift;
	vector<int> candidates;

#ifdef DEBUG
	double distanceToInDummy = DistanceToInDummy(aPoint, aDirection, aPstep);
#endif

	double minDistance = UUtils::kInfinity;

	UVector3 currentPoint = aPoint;
	shift = voxels.DistanceToFirst(currentPoint, direction);
	double totalShift = shift;
	UBits exclusion(voxels.GetBitsPerSlice());
	exclusion.ResetAllBits(true);

	while (shift < UUtils::kInfinity)
	{
		if (shift)
		{
			currentPoint += direction * shift;
#ifdef DEBUG
			if (!voxels.Contains(currentPoint)) 
				shift = shift; // put a breakpoint here
#endif
		}

//		cout << "New point: [" << currentPoint.x << " , " << currentPoint.y << " , " << currentPoint.z << "]" << endl; 

		shift = voxels.DistanceToNext(currentPoint, direction);
		totalShift += shift;
		if (!shift) 
			break;

		// we try to find a non-empty voxel
		if (voxels.GetCandidatesVoxelArray(currentPoint, candidates, &exclusion))
		{
			double distance = DistanceToInCandidates(aPoint, direction, aPstep, candidates, exclusion); 
			if (minDistance > distance) 
			{
				minDistance = distance;
				if (distance < totalShift) 
					break;
			}
		}
	}
#ifdef DEBUG
	if (fabs(minDistance - distanceToInDummy) > VUSolid::Tolerance())
		minDistance = distanceToInDummy; // you can place a breakpoint here
#endif

	return minDistance;
}


double UMultiUnion::DistanceToOutDummy(const UVector3 &aPoint, const UVector3 &aDirection,
	UVector3 &aNormal,
	bool     &convex,
	double   aPstep) const
{
	// Computes distance from a point presumably outside the solid to the solid 
	// surface. Ignores first surface if the point is actually inside. Early return
	// infinity in case the safety to any surface is found greater than the proposed
	// step aPstep.
	// The normal vector to the crossed surface is filled only in case the box is 
	// crossed, otherwise aNormal.IsNull() is true.

	// algorithm:
	UVector3 direction = aDirection.Unit();   
	UVector3 localPoint, localDirection;
	int ignoredSolid = -1;
	double resultDistToOut = UUtils::kInfinity;
	UVector3 currentPoint = aPoint;

	int numNodes = solids.size();
	for(int i = 0; i < numNodes; i++)
	{
		if (i != ignoredSolid)
		{
			VUSolid &solid = *solids[i];
			UTransform3D &transform = *transforms[i];
			localPoint = transform.LocalPoint(currentPoint);
			localDirection = transform.LocalVector(direction);
			VUSolid::EnumInside location = solid.Inside(localPoint);
			if (location != eOutside)
			{
				double distance = solid.DistanceToOut(localPoint, localDirection, aNormal, convex);
				if (distance < UUtils::kInfinity)
				{
					if (resultDistToOut == UUtils::kInfinity) resultDistToOut = 0;
					if (distance > 0)
					{
						currentPoint = transform.GlobalPoint(localPoint+distance*localDirection);
						resultDistToOut += distance;
						ignoredSolid = i; // skip the solid which we have just left
						i = -1; // force the loop to continue from 0
					}
				}
			}
		}
	}
	return resultDistToOut;
}



double UMultiUnion::DistanceToOut(const UVector3 &aPoint, const UVector3 &aDirection,
	UVector3 &aNormal,
	bool     &convex,
	double   aPstep) const
{
//	return DistanceToOutDummy(aPoint, aDirection, aNormal, convex, aPstep);

	double distanceToOutDummy = DistanceToOutDummy(aPoint, aDirection, aNormal, convex, aPstep);

	double distanceToOutVoxels = DistanceToOutVoxels(aPoint, aDirection, aNormal, convex, aPstep);

#ifdef DEBUG
	if (std::abs(distanceToOutVoxels - distanceToOutDummy) > VUSolid::Tolerance())
	{
		// distanceToOutVoxels = distanceToOutVoxels;
		Inside(aPoint);
	}
//		return distanceToOutDummy;
#endif
	return distanceToOutVoxels;
}

double UMultiUnion::DistanceToOutVoxelsCoreNew(const UVector3 &point, const UVector3 &direction, UVector3 &normal, bool &convex, vector<int> &candidates) const
{
	double distance = -1;
	UVector3 localPoint, localDirection, localNormal;
	UVector3 currentPoint = point;
	UBits exclusion(voxels.GetBitsPerSlice());
	exclusion.ResetAllBits(true);
	bool notOutside;
	UVector3 maxNormal;

	do
	{
		notOutside = false;

		double maxDistance = -UUtils::kInfinity;
		int maxCandidate;
		UVector3 maxLocalPoint;

		int limit = candidates.size();
		for(int i = 0 ; i < limit ; i++)
		{
			int candidate = candidates[i];
			// ignore the current component (that you just got out of) since numerically the propagated point will be on its surface

			VUSolid &solid = *solids[candidate];
			UTransform3D &transform = *transforms[candidate];

			// The coordinates of the point are modified so as to fit the intrinsic solid local frame:
			localPoint = transform.LocalPoint(currentPoint);

			// identify the current component via a version of UMultiUnion::Inside that gives it back (discussed with John and Jean-Marie)

			if(solid.Inside(localPoint) != eOutside)
			{
				notOutside = true;

				localDirection = transform.LocalVector(direction);
				// propagate with solid.DistanceToOut
				bool convex;
				double shift = solid.DistanceToOut(localPoint, localDirection, localNormal, convex);
				if (maxDistance < shift) 
				{
					maxDistance = shift;
					maxCandidate = candidate;
					maxNormal = localNormal;
				}
			}
		}

		if (notOutside)
		{
			UTransform3D &transform = *transforms[maxCandidate];
			localPoint = transform.LocalPoint(currentPoint);

			if (distance < 0) distance = 0;

			distance += maxDistance;
			currentPoint = transform.GlobalPoint(localPoint+maxDistance*localDirection);

			// convert from local normal
			normal = transform.GlobalVector(maxNormal);

			exclusion.ResetBitNumber(maxCandidate);
			VUSolid::EnumInside location = InsideWithExclusion(currentPoint, &exclusion);
			exclusion.SetBitNumber(maxCandidate);

			// perform a Inside 
			// it should be excluded current solid from checking
			// we have to collect the maximum distance from all given candidates. such "maximum" candidate should be then used for finding next candidates
			if(location != eInside)
			{
				// else return cumulated distances to outside of the traversed components
				return distance;               
			}
			// if inside another component, redo 1 to 3 but add the next DistanceToOut on top of the previous.

			// and fill the candidates for the corresponding voxel (just exiting current component along direction)
			candidates.clear();
			// the current component will be ignored
			exclusion.ResetBitNumber(maxCandidate);
			voxels.GetCandidatesVoxelArray(currentPoint, candidates, &exclusion);
			exclusion.SetBitNumber(maxCandidate);
		}
	}
	while (notOutside);

	return distance;
}

double UMultiUnion::DistanceToOutVoxelsCore(const UVector3 &point, const UVector3 &direction, UVector3 &normal, bool &convex, vector<int> &candidates) const
{
	bool notOutside = false;
	double distance = 0;
	UVector3 localPoint, localDirection, localNormal;
	UVector3 currentPoint = point;
	UBits exclusion(voxels.GetBitsPerSlice());
	exclusion.ResetAllBits(true);

	do
	{
		int limit = candidates.size();
		for(int i = 0 ; i < limit ; i++)
		{
			int candidate = candidates[i];
			// ignore the current component (that you just got out of) since numerically the propagated point will be on its surface

			VUSolid &solid = *solids[candidate];
			UTransform3D &transform = *transforms[candidate];

			// The coordinates of the point are modified so as to fit the intrinsic solid local frame:
			localPoint = transform.LocalPoint(currentPoint);

			// identify the current component via a version of UMultiUnion::Inside that gives it back (discussed with John and Jean-Marie)

			if(solid.Inside(localPoint) != eOutside)
			{
				notOutside = true;

				localDirection = transform.LocalVector(direction);
				// propagate with solid.DistanceToOut
				bool convex;
				double shift = solid.DistanceToOut(localPoint, localDirection, localNormal, convex);

				distance += shift;
				currentPoint = transform.GlobalPoint(localPoint+shift*localDirection);

				// convert from local normal
				normal = transform.GlobalVector(localNormal);

				exclusion.ResetBitNumber(candidate);
				VUSolid::EnumInside location = InsideWithExclusion(currentPoint, &exclusion);
				exclusion.SetBitNumber(candidate);

				// perform a Inside 
				// it should be excluded current solid from checking
				// we have to collect the maximum distance from all given candidates. such "maximum" candidate should be then used for finding next candidates
				if(location != eInside)
				{
					// else return cumulated distances to outside of the traversed components
					return distance;               
				}
				// if inside another component, redo 1 to 3 but add the next DistanceToOut on top of the previous.

				// and fill the candidates for the corresponding voxel (just exiting current component along direction)
				candidates.clear();
				// the current component will be ignored
				exclusion.ResetBitNumber(candidate);
				voxels.GetCandidatesVoxelArray(currentPoint, candidates, &exclusion);            
				exclusion.SetBitNumber(candidate);
				break;
			}
		}
	}
	while (notOutside);

	if (notOutside)
		return distance;

	return -UUtils::kInfinity;
}

//______________________________________________________________________________
double UMultiUnion::DistanceToOutVoxels(const UVector3 &aPoint, const UVector3 &aDirection,
	UVector3 &aNormal,
	bool     &convex,
	double   aPstep) const
{
	// Computes distance from a point presumably inside the solid to the solid 
	// surface. Ignores first surface along each axis systematically (for points
	// inside or outside. Early returns zero in case the second surface is behind
	// the starting point.
	// o The proposed step is ignored.
	// o The normal vector to the crossed surface is always filled.

	// In the case the considered point is located inside the UMultiUnion structure,
	// the treatments are as follows:
	//      - investigation of the candidates for the passed point
	//      - progressive moving of the point towards the surface, along the passed direction
	//      - processing of the normal

	UVector3 direction = aDirection.Unit();
	vector<int> candidates;

	if(voxels.GetCandidatesVoxelArray(aPoint, candidates))
	{
		// For normal case for which we presume the point is inside
		double distance = DistanceToOutVoxelsCoreNew(aPoint, direction, aNormal, convex, candidates);
		if (distance != -UUtils::kInfinity)
			return distance;
	}

	/*
	UVector3 currentPoint;
	const double localTolerance = 1E-5;

	// we will try to include in our estimations also neighbouring voxels within tolerance
	for(int i = -1 ; i <= 1 ; i +=2)
	{
		for(int j = -1 ; j <= 1 ; j +=2)
		{
			for(int k = -1 ; k <= 1 ; k +=2)
			{
				currentPoint.Set(aPoint.x+i*localTolerance, aPoint.y+j*localTolerance, aPoint.z+k*localTolerance);

				if(voxels.GetCandidatesVoxelArray(currentPoint, &candidates))
				{
					double distance = DistanceToOutVoxelsCoreNew(aPoint, direction, aNormal, convex, candidates);
					if (distance != -UUtils::kInfinity)
						return distance;

				}      
			}
		}
	}
	*/

	return 0;
}    


//TODO: delete this method, does not work anyway anymore, because we changes bits format
//______________________________________________________________________________
VUSolid::EnumInside UMultiUnion::InsideBits(const UVector3 &aPoint) const
{
#ifdef DEBUG
	VUSolid::EnumInside insideDummy = InsideDummy(aPoint);
#endif

	UVector3 localPoint;
	VUSolid::EnumInside location = eOutside;
	bool boolSurface = false;

	UBits bits;
	voxels.GetCandidatesVoxelBits(aPoint, bits);

	int limit = bits.CounUBits();
	for(int i = 0 ; i < limit ; i++)
	{
		int candidate = bits.FirstSetBit();
		bits.SetBitNumber(candidate, false);

		VUSolid &solid = *solids[candidate];
		UTransform3D &transform = *transforms[candidate];  

		// The coordinates of the point are modified so as to fit the intrinsic solid local frame:
		localPoint = transform.LocalPoint(aPoint);
		location = solid.Inside(localPoint);
		if(location == eSurface) boolSurface = true; 

		if(location == eInside) 
		{
#ifdef DEBUG 
			if (location != insideDummy)
				location = insideDummy; // you can place a breakpoint here
#endif

			return eInside;      
		}
	}          
	///////////////////////////////////////////////////////////////////////////
	// Important comment: When two solids touch each other along a flat
	// surface, the surface points will be considered as eSurface, while points 
	// located around will correspond to eInside (cf. G4UnionSolid in GEANT4)
	location = boolSurface ? eSurface : eOutside;

#ifdef DEBUG
	if (location != insideDummy)
		location = insideDummy; // you can place a breakpoint here
#endif
	return location;
}




//______________________________________________________________________________
VUSolid::EnumInside UMultiUnion::InsideWithExclusion(const UVector3 &aPoint, UBits *exclusion) const
{
	// Classify point location with respect to solid:
	//  o eInside       - inside the solid
	//  o eSurface      - close to surface within tolerance
	//  o eOutside      - outside the solid

	// Hitherto, it is considered that:
	//        - only parallelepipedic nodes can be added to the container

	// Implementation using voxelisation techniques:
	// ---------------------------------------------

	UVector3 localPoint;
	VUSolid::EnumInside location = eOutside;
	bool surface = false;

	vector<int> candidates;
	int limit = voxels.GetCandidatesVoxelArray(aPoint, candidates, exclusion);
	for(int i = 0 ; i < limit ; i++)
	{
		int candidate = candidates[i];
		VUSolid &solid = *solids[candidate];
		UTransform3D &transform = *transforms[candidate];  

		// The coordinates of the point are modified so as to fit the intrinsic solid local frame:
		localPoint = transform.LocalPoint(aPoint);
		location = solid.Inside(localPoint);
		if(location == eSurface) surface = true; 

		if(location == eInside) return eInside;      
	}          
	///////////////////////////////////////////////////////////////////////////
	// Important comment: When two solids touch each other along a flat
	// surface, the surface points will be considered as eSurface, while points 
	// located around will correspond to eInside (cf. G4UnionSolid in GEANT4)
	location = surface ? eSurface : eOutside;

	return location;
}


//______________________________________________________________________________
VUSolid::EnumInside UMultiUnion::Inside(const UVector3 &aPoint) const
{
	// Classify point location with respect to solid:
	//  o eInside       - inside the solid
	//  o eSurface      - close to surface within tolerance
	//  o eOutside      - outside the solid

	// Hitherto, it is considered that:
	//        - only parallelepipedic nodes can be added to the container

	// Implementation using voxelisation techniques:
	// ---------------------------------------------

//	return InsideIterator(aPoint);

#ifdef DEBUG
	VUSolid::EnumInside insideDummy = InsideDummy(aPoint);
#endif

	VUSolid::EnumInside location = InsideWithExclusion(aPoint);

#ifdef DEBUG
	if (location != insideDummy)
		location = insideDummy; // you can place a breakpoint here
#endif
	return location;
}

//______________________________________________________________________________
VUSolid::EnumInside UMultiUnion::InsideIterator(const UVector3 &aPoint) const
{
	// Classify point location with respect to solid:
	//  o eInside       - inside the solid
	//  o eSurface      - close to surface within tolerance
	//  o eOutside      - outside the solid

	// Hitherto, it is considered that:
	//        - only parallelepipedic nodes can be added to the container

	// Implementation using voxelisation techniques:
	// ---------------------------------------------

//	return InsideBits(aPoint);

#ifdef DEBUG
	VUSolid::EnumInside insideDummy = InsideDummy(aPoint);
#endif

	UVector3 localPoint;
	VUSolid::EnumInside location = eOutside;
	bool boolSurface = false;

	vector<int> candidates;
	//   candidates = voxels.GetCandidatesVoxelArrayOld(aPoint); 
	voxels.GetCandidatesVoxelArray(aPoint, candidates);

	UVoxelCandidatesIterator iterator(voxels, aPoint);
	int candidate;
	while ((candidate = iterator.Next()) >= 0)
	{
		VUSolid &solid = *solids[candidate];
		UTransform3D &transform = *transforms[candidate];  

		// The coordinates of the point are modified so as to fit the intrinsic solid local frame:
		localPoint = transform.LocalPoint(aPoint);
		location = solid.Inside(localPoint);
		if(location == eSurface) boolSurface = true; 

		if(location == eInside) 
		{
#ifdef DEBUG 
			if (location != insideDummy)
				location = insideDummy; // you can place a breakpoint here
#endif

			return eInside;      
		}
	}          
	///////////////////////////////////////////////////////////////////////////
	// Important comment: When two solids touch each other along a flat
	// surface, the surface points will be considered as eSurface, while points 
	// located around will correspond to eInside (cf. G4UnionSolid in GEANT4)
	location = boolSurface ? eSurface : eOutside;

#ifdef DEBUG
	if (location != insideDummy)
		location = insideDummy; // you can place a breakpoint here
#endif
	return location;
}

//______________________________________________________________________________
VUSolid::EnumInside UMultiUnion::InsideDummy(const UVector3 &aPoint) const
{
	UVector3 localPoint;
	VUSolid::EnumInside location = eOutside;
	int countSurface = 0;

	int numNodes = solids.size();
	for(int i = 0 ; i < numNodes ; i++)
	{
		VUSolid &solid = *solids[i];
		UTransform3D &transform = *transforms[i];  

		// The coordinates of the point are modified so as to fit the intrinsic solid local frame:
		localPoint = transform.LocalPoint(aPoint);

		location = solid.Inside(localPoint);        

		if(location == eSurface) 
			countSurface++; 

		if(location == eInside) return eInside;      
	}       
	if(countSurface != 0) return eSurface;
	return eOutside;
}

//______________________________________________________________________________ 
void UMultiUnion::Extent(EAxisType aAxis,double &aMin,double &aMax) const
{
	// Determines the bounding box for the considered instance of "UMultipleUnion"
	double mini, maxi;
	double arrMin[3], arrMax[3];
	UVector3 min, max;

	int numNodes = solids.size();
	for(int i = 0 ; i < numNodes ; i++)
	{
		VUSolid &solid = *solids[i];
		UTransform3D &transform = *transforms[i];

		solid.Extent(arrMin, arrMax);
		min.Set(arrMin[0],arrMin[1],arrMin[2]);      
		max.Set(arrMax[0],arrMax[1],arrMax[2]);           
		UUtils::TransformLimits(min, max, transform);

		if (i == 0)
		{
			switch(aAxis)
			{
				case eXaxis:
					mini = min.x;
					maxi = max.x;            
					break;
				case eYaxis:
					mini = min.y;
					maxi = max.y;            
					break;
				case eZaxis:
					mini = min.z;
					maxi = max.z;            
					break;
			}
		}
		else
		{
			// Deternine the min/max on the considered axis:
			switch(aAxis)
			{
				case eXaxis:
					if(min.x < mini)
						mini = min.x;
					if(max.x > maxi)
						maxi = max.x;
					break;
				case eYaxis:
					if(min.y < mini)
						mini = min.y;
					if(max.y > maxi)
						maxi = max.y;
					break;
				case eZaxis:
					if(min.z < mini)
						mini = min.z;
					if(max.z > maxi)
						maxi = max.z;
					break;
			}                 
		}
	}
	aMin = mini;
	aMax = maxi;
}

//______________________________________________________________________________ 
void UMultiUnion::Extent(double aMin[3],double aMax[3]) const
{
	double min = 0,max = 0;
	Extent(eXaxis,min,max);
	aMin[0] = min; aMax[0] = max;
	Extent(eYaxis,min,max);
	aMin[1] = min; aMax[1] = max;
	Extent(eZaxis,min,max);
	aMin[2] = min; aMax[2] = max;      
}

//______________________________________________________________________________
// This function is not in TGeoShapeAssembly
bool UMultiUnion::Normal(const UVector3& aPoint, UVector3 &aNormal) const
{
	// Computes the localNormal on a surface and returns it as a unit vector
	//   In case a point is further than toleranceNormal from a surface, set validNormal=false
	//   Must return a valid vector. (even if the point is not on the surface.)
	//
	//   On an edge or corner, provide an average localNormal of all facets within tolerance
	// NOTE: the tolerance value used in here is not yet the global surface
	//     tolerance - we will have to revise this value - TODO
	vector<int> candidates;   
	UVector3 localPoint, normal, localNormal;
	double safety = UUtils::kInfinity;
	int node = -1;
	double normalTolerance = 1E-5;

	///////////////////////////////////////////////////////////////////////////
	// Important comment: Cases for which the point is located on an edge or
	// on a vertice remain to be treated  

	// determine weather we are in voxel area
	if(voxels.GetCandidatesVoxelArray(aPoint, candidates))
	{
		int limit = candidates.size();
		for(int i = 0 ; i < limit ; i++)
		{
			int candidate = candidates[i];
			UTransform3D &transform = *transforms[candidate];
			// The coordinates of the point are modified so as to fit the intrinsic solid local frame:
			localPoint = transform.LocalPoint(aPoint);
			VUSolid &solid = *solids[candidate];
			VUSolid::EnumInside location = solid.Inside(localPoint);

			if(location == eSurface)
			{
				// normal case when point is on surface, we pick first solid
				solid.Normal(localPoint,localNormal);
				normal = transform.GlobalVector(localNormal);
				aNormal = normal.Unit();           
				return true;  
			}
			else
			{
				// collect the smallest safety and remember solid node
				if(location == eInside)
				{
					if(solid.SafetyFromInside(localPoint) < safety)
					{
						safety = solid.SafetyFromInside(localPoint);
						node = candidate;
					}
				}
				else // case eOutside
				{
					if(solid.SafetyFromOutside(localPoint) < safety)
					{
						safety = solid.SafetyFromOutside(localPoint);
						node = candidate;
					}    
				}
			}      
		}
		// on none of the solids, the point was not on the surface
		VUSolid &solid = *solids[node];
		UTransform3D &transform = *transforms[node];            
		localPoint = transform.LocalPoint(aPoint);

		solid.Normal(localPoint,localNormal);
		normal = transform.GlobalVector(localNormal);
		aNormal = normal.Unit();
		if(safety > normalTolerance) return false;
		return true;
	}
	else
	{
		// for the case when point is certainly outside:

		// find a solid in union with the smallest safety
		int node = SafetyFromOutsideNumberNode(aPoint,true,safety);
		VUSolid &solid = *solids[node];
		
		UTransform3D &transform = *transforms[node];            
		localPoint = transform.LocalPoint(aPoint);

		// evaluate normal for point at this found solid
		solid.Normal(localPoint,localNormal);

		// transform multi-union coordinates
		normal = transform.GlobalVector(localNormal);

		aNormal = normal.Unit();
		if (safety > normalTolerance) return false;
		return true;     
	}
}

//______________________________________________________________________________ 
// see also TGeoShapeAssembly::Safety
double UMultiUnion::SafetyFromInside(const UVector3 &point, bool aAccurate) const
{
	// Estimates isotropic distance to the surface of the solid. This must
	// be either accurate or an underestimate. 
	//  Two modes: - default/fast mode, sacrificing accuracy for speed
	//             - "precise" mode,  requests accurate value if available.   

	vector<int> candidates;
	UVector3 localPoint;
	double safetyMax = 0.;

	// In general, the value return by SafetyFromInside will not be the exact
	// but only an undervalue (cf. overlaps)    
	voxels.GetCandidatesVoxelArray(point, candidates);

	int limit = candidates.size();
	for(int i = 0; i < limit; i++)
	{
		int candidate = candidates[i];
		// The coordinates of the point are modified so as to fit the intrinsic solid local frame:      
		UTransform3D &transform = *transforms[candidate];
		localPoint = transform.LocalPoint(point);
		VUSolid &solid = *solids[candidate];
		if(solid.Inside(localPoint) == eInside)
		{
			double safety = solid.SafetyFromInside(localPoint, aAccurate);
			if (safetyMax < safety) safetyMax = safety;
		}   
	}
	return safetyMax;   
}

//______________________________________________________________________________
// see also TGeoShapeAssembly::Safety
double UMultiUnion::SafetyFromOutside(const UVector3 &point, bool aAccurate) const
{
	// Estimates the isotropic safety from a point outside the current solid to any 
	// of its surfaces. The algorithm may be accurate or should provide a fast 
	// underestimate.

	const std::vector<UVoxelBox> &boxes = voxels.GetBoxes();
	double safetyMin = UUtils::kInfinity;   
	UVector3 localPoint;

	int numNodes = solids.size();
	for(int i = 0; i < numNodes; i++)
	{
		if (i > 0)
		{
			// quick checks which help to speed up the things a bit
			double dxyz0 = std::abs(point.x-boxes[i].p.x)-boxes[i].d.x;
			if (dxyz0 > safetyMin) continue;
			double dxyz1 = std::abs(point.y-boxes[i].p.y)-boxes[i].d.y;
			if (dxyz1 > safetyMin) continue;
			double dxyz2 = std::abs(point.z-boxes[i].p.z)-boxes[i].d.z;      
			if (dxyz2 > safetyMin) continue;

			double d2xyz = 0.;
			if (dxyz0 > 0) d2xyz += dxyz0*dxyz0;
			if (dxyz1 > 0) d2xyz += dxyz1*dxyz1;
			if (dxyz2 > 0) d2xyz += dxyz2*dxyz2;
			if (d2xyz >= safetyMin*safetyMin) continue;
		}
		UTransform3D &transform = *transforms[i];
		localPoint = transform.LocalPoint(point); // NOTE: ROOT does not make this transformation, although it does it at SafetyFromInside
		VUSolid &solid = *solids[i];
		double safety = solid.SafetyFromOutside(localPoint, aAccurate);
		if (safety <= 0) return safety; // it was detected, that the point is not located outside
		if (safetyMin > safety) safetyMin = safety;
	}
	return safetyMin;
}

//______________________________________________________________________________       
double UMultiUnion::SurfaceArea()
{
	// Computes analytically the surface area.
	cout << "SurfaceArea - Not implemented" << endl;
	return 0.;
}     

 

//______________________________________________________________________________       

//______________________________________________________________________________       
void UMultiUnion::Voxelize()
{
	((UVoxelFinder &)voxels).Voxelize(solids, transforms);
}

//______________________________________________________________________________
int UMultiUnion::SafetyFromOutsideNumberNode(const UVector3 &aPoint, bool aAccurate, double &safetyMin) const
{
	// Method returning the closest node from a point located outside a UMultiUnion.
	// This is used to compute the normal in the case no candidate has been found.

	const std::vector<UVoxelBox> &boxes = voxels.GetBoxes();
	safetyMin = UUtils::kInfinity;
	int safetyNode = -1;
	UVector3 localPoint;    

	int numNodes = solids.size();
	for(int i = 0; i < numNodes; i++)
	{  
		double d2xyz = 0.;
		double dxyz0 = std::abs(aPoint.x-boxes[i].p.x)-boxes[i].d.x;
		if (dxyz0 > safetyMin) continue;
		double dxyz1 = std::abs(aPoint.y-boxes[i].p.y)-boxes[i].d.y;
		if (dxyz1 > safetyMin) continue;
		double dxyz2 = std::abs(aPoint.z-boxes[i].p.z)-boxes[i].d.z;
		if (dxyz2 > safetyMin) continue;

		if(dxyz0 > 0) d2xyz += dxyz0*dxyz0;
		if(dxyz1 > 0) d2xyz += dxyz1*dxyz1;
		if(dxyz2 > 0) d2xyz += dxyz2*dxyz2;
		if(d2xyz >= safetyMin*safetyMin) continue;

		VUSolid &solid = *solids[i];
		UTransform3D &transform = *transforms[i];
		localPoint = transform.LocalPoint(aPoint);      
		double safety = solid.SafetyFromOutside(localPoint,true);
		if(safetyMin > safety)
		{
			safetyMin = safety;
			safetyNode = i;
		}
	}
	return safetyNode;
}
