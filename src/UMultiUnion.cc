
#include <iostream>
#include <sstream>

#include "UVoxelFinder.hh"
#include "UMultiUnion.hh"
#include "UUtils.hh"

#include "UBox.hh"

using namespace std;

// test with boolean union in geant4 in SBT
// measure performance of DistanceToIn vs. DistanceToOut
// in distancetoin/out benchmark, try what influence has adding Ubits

// DONE
// struct UVoxelBox
// {
//    UVector3 hlen; // half length of the box
//    UVector3 pos; // position of the box
// 

// if (maskByte & 1)
// {
//   list.push_back(8*(sizeof(unsigned int)*i+ byte) + bit);
//   if (!(maskByte >>= 1)) break; // new
// }
// else maskByte >>= 1;
// invert bitmasks

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

	UVector3 extentMin, extentMax, d, p, point;
	int inside = 0, generated;
	Extent(extentMin,extentMax);
	d = (extentMax - extentMin) / 2.;
	p = (extentMax + extentMin) / 2.;
	UVector3 left = p - d;
	UVector3 length = d * 2;
	for (generated = 0; generated < 10000; generated++)
	{
		UVector3 random(rand(), rand(), rand());
		point = left + length.MultiplyByComponents(random / RAND_MAX);
		if (Inside(point) != eOutside) inside++;
	}
	double vbox = (2*d.x)*(2*d.y)*(2*d.z);
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
	for (int i = 0 ; i < numNodes ; ++i)
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
	for (int i = 0 ; i < candidatesCount; ++i)
	{
		int candidate = candidates[i];
		VUSolid &solid = *solids[candidate];
		UTransform3D &transform = *transforms[candidate];

		localPoint = transform.LocalPoint(aPoint);
		localDirection = transform.LocalVector(direction);                
		double distance = solid.DistanceToIn(localPoint, localDirection, aPstep);
		if (minDistance > distance) minDistance = distance;
		bits.SetBitNumber(candidate);
		if (minDistance == 0) break;
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
	const UVector3 &aDirection, double aPstep) const
{
//	return DistanceToInDummy(aPoint, aDirection, aPstep);

#ifdef DEBUG
	double distanceToInDummy = DistanceToInDummy(aPoint, aDirection, aPstep);
#endif

	double minDistance = UUtils::kInfinity;
	UVector3 currentPoint = aPoint;
	UVector3 direction = aDirection.Unit();
	double shift = voxels.DistanceToFirst(currentPoint, direction);
	if (shift == UUtils::kInfinity) return shift;
	double shiftBonus = VUSolid::Tolerance()/10;
	if (shift) 
		currentPoint += direction * (shift + shiftBonus);
//		if (!voxels.Contains(currentPoint)) 
//			return minDistance;
	double totalShift = shift;

	UBits exclusion(voxels.GetBitsPerSlice());
	vector<int> candidates, curVoxel(3);
	for (int i = 0; i <= 2; ++i) curVoxel[i] = UUtils::BinarySearch(voxels.GetBoundary(i), currentPoint[i]);

	do
	{
		if (!voxels.empty.GetNbits() || !voxels.empty[voxels.GetVoxelsIndex(curVoxel)])
		{
			if (voxels.GetCandidatesVoxelArray(curVoxel, candidates, &exclusion))
			{
				double distance = DistanceToInCandidates(aPoint, direction, aPstep, candidates, exclusion); 
				if (minDistance > distance) 
					if (distance < totalShift) break; else minDistance = distance;
			}
		}
		shift = voxels.DistanceToNext(currentPoint, direction, curVoxel);
		if (shift == UUtils::kInfinity /*|| shift == 0*/) break;

		totalShift += shift;
		if (minDistance < totalShift) break;

		currentPoint += direction * (shift + shiftBonus);
	}
	while (voxels.UpdateCurrentVoxel(currentPoint, direction, curVoxel));

#ifdef DEBUG
	if (fabs(minDistance - distanceToInDummy) > VUSolid::Tolerance())
	{
		VUSolid::EnumInside location = Inside(aPoint);
		minDistance = distanceToInDummy; // you can place a breakpoint here
	}
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
	double resultDistToOut = 0; // UUtils::kInfinity;
	UVector3 currentPoint = aPoint;

	int numNodes = solids.size();
	for(int i = 0; i < numNodes; ++i)
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

#ifdef DEBUG
	double distanceToOutDummy = DistanceToOutDummy(aPoint, aDirection, aNormal, convex, aPstep);
#endif

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
		double distance = -1;
		UVector3 localPoint, localDirection, localNormal;
		UVector3 currentPoint = aPoint;
		UBits exclusion(voxels.GetBitsPerSlice());
		bool notOutside;
		UVector3 maxNormal;

		do
		{
			notOutside = false;

			double maxDistance = -UUtils::kInfinity;
			int maxCandidate;
			UVector3 maxLocalPoint;

			int limit = candidates.size();
			for(int i = 0 ; i < limit ; ++i)
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
				aNormal = transform.GlobalVector(maxNormal);

				exclusion.SetBitNumber(maxCandidate);
				VUSolid::EnumInside location = InsideWithExclusion(currentPoint, &exclusion);
				exclusion.ResetBitNumber(maxCandidate);

				// perform a Inside 
				// it should be excluded current solid from checking
				// we have to collect the maximum distance from all given candidates. such "maximum" candidate should be then used for finding next candidates
				if(location != eInside)
				{
					// else return cumulated distances to outside of the traversed components
					break;               
				}
				// if inside another component, redo 1 to 3 but add the next DistanceToOut on top of the previous.

				// and fill the candidates for the corresponding voxel (just exiting current component along direction)
				candidates.clear();
				// the current component will be ignored
				exclusion.SetBitNumber(maxCandidate);
				voxels.GetCandidatesVoxelArray(currentPoint, candidates, &exclusion);
				exclusion.ResetBitNumber(maxCandidate);
			}
		}
		while (notOutside);

		if (distance != -UUtils::kInfinity)
			return distance;
	}

	return 0;
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

	// TODO: test if it works well and if so measure performance
//	if (!voxels.empty.GetNbits() || !voxels.empty[voxels.GetPointIndex(aPoint)])
	{
		int limit = voxels.GetCandidatesVoxelArray(aPoint, candidates, exclusion);
		for(int i = 0 ; i < limit ; ++i)
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
	for(int i = 0 ; i < numNodes ; ++i)
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
	UVector3 min, max;

	int numNodes = solids.size();
	for(int i = 0 ; i < numNodes ; ++i)
	{
		VUSolid &solid = *solids[i];
		UTransform3D &transform = *transforms[i];
		solid.Extent(min, max);
		UUtils::TransformLimits(min, max, transform);

		if (i == 0)
		{
			switch(aAxis)
			{
				case eXaxis:
					aMin = min.x;
					aMax = max.x;            
					break;
				case eYaxis:
					aMin = min.y;
					aMax = max.y;            
					break;
				case eZaxis:
					aMin = min.z;
					aMax = max.z;            
					break;
			}
		}
		else
		{
			// Deternine the min/max on the considered axis:
			switch(aAxis)
			{
				case eXaxis:
					if(min.x < aMin)
						aMin = min.x;
					if(max.x > aMax)
						aMax = max.x;
					break;
				case eYaxis:
					if(min.y < aMin)
						aMin = min.y;
					if(max.y > aMax)
						aMax = max.y;
					break;
				case eZaxis:
					if(min.z < aMin)
						aMin = min.z;
					if(max.z > aMax)
						aMax = max.z;
					break;
			}                 
		}
	}
}

//______________________________________________________________________________ 
void UMultiUnion::Extent (UVector3 &aMin, UVector3 &aMax) const
{
	for (int i = 0; i <= 2; ++i) Extent(eXaxis,aMin[i],aMax[i]);
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
		for(int i = 0 ; i < limit ; ++i)
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
	for(int i = 0; i < limit; ++i)
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
	for(int j = 0; j < numNodes; j++)
	{
		UVector3 dxyz;
		if (j > 0)
		{
			UVector3 pos = boxes[j].pos;
			UVector3 hlen = boxes[j].hlen;
			for (int i = 0; i <= 2; ++i)
				// distance to middle point - hlength => distance from point to border of x,y,z
				if ((dxyz[i] = std::abs(point[i]-pos[i])-hlen[i]) > safetyMin) 
					continue;
			
			double d2xyz = 0.;
			for (int i = 0; i <= 2; ++i)
				if (dxyz[i] > 0) d2xyz += dxyz[i]*dxyz[i];
			
			// minimal distance is at least this, but could be even higher. therefore, we can stop if previous was already lower
			if (d2xyz >= safetyMin*safetyMin) 
			{
#ifdef DEBUG
				UTransform3D &transform = *transforms[j];
				localPoint = transform.LocalPoint(point); // NOTE: ROOT does not make this transformation, although it does it at SafetyFromInside
				VUSolid &solid = *solids[j];
				double safety = solid.SafetyFromOutside(localPoint, true);
				if (safetyMin > safety)
					safety = safety;
#endif		
				continue;
			}
		}
		UTransform3D &transform = *transforms[j];
		localPoint = transform.LocalPoint(point); // NOTE: ROOT does not make this transformation, although it does it at SafetyFromInside
		VUSolid &solid = *solids[j];

		double safety = solid.SafetyFromOutside(localPoint, aAccurate); // careful, with aAcurate it can return underestimate, than the condition d2xyz >= safetyMin*safetyMin does not return same result as Geant4 or Root boolean union, it actually return better values, more close to surface
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
void UMultiUnion::Voxelize()
{
	voxels.Voxelize(solids, transforms);
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
	for(int i = 0; i < numNodes; ++i)
	{  
		double d2xyz = 0.;
		double dxyz0 = std::abs(aPoint.x-boxes[i].pos.x)-boxes[i].hlen.x;
		if (dxyz0 > safetyMin) continue;
		double dxyz1 = std::abs(aPoint.y-boxes[i].pos.y)-boxes[i].hlen.y;
		if (dxyz1 > safetyMin) continue;
		double dxyz2 = std::abs(aPoint.z-boxes[i].pos.z)-boxes[i].hlen.z;
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

const double unionMaxX = 1.; // putting it larger can cause particles to escape
const double unionMaxY = 1.;
const double unionMaxZ = 1.;

const double extentBorder = 1.1;

const int carBoxesX = 20;
const int carBoxesY = 20;
const int carBoxesZ = 20;


UMultiUnion *UMultiUnion::CreateTestMultiUnion(int numNodes) // Number of nodes to implement
{
   // Instance:
      // Creation of several nodes:

   double extentVolume = extentBorder * 2 * unionMaxX * extentBorder * 2 * unionMaxY * extentBorder * 2 * unionMaxZ;
   double ratio = 1.0/3.0; // ratio of inside points vs (inside + outside points)
   double length = 40;
   
   if (true) length = pow (ratio * extentVolume / numNodes, 1./3.) / 2;
	
   UBox *box = new UBox("UBox", length, length, length);

//   UOrb *box = new UOrb("UOrb", length);

   double capacity = box->Capacity();

   UTransform3D* arrayTransformations = new UTransform3D[numNodes];
  
     // Constructor:
   UMultiUnion *multiUnion = new UMultiUnion("multiUnion");

   if (true)
   {
	   for(int i = 0; i < numNodes ; ++i)
	   {
		   double x = UUtils::RandomUniform(-unionMaxX + length, unionMaxX - length);
		   double y = UUtils::RandomUniform(-unionMaxY + length, unionMaxY - length);
		   double z = UUtils::RandomUniform(-unionMaxZ + length, unionMaxZ - length);

		   arrayTransformations[i] = UTransform3D(x,y,z,0,0,0);
		   multiUnion->AddNode(*box,arrayTransformations[i]);
	   }
   }
   else 
   {   
	   // Transformation:
	   for(int n = 0, o = 0, m = 0; m < numNodes ; m++)
	   {
		   if (m >= carBoxesX*carBoxesY*carBoxesZ) break;
		   double spacing = 50;
		   double x = -unionMaxX+spacing+2*spacing*(m%carBoxesX);
		   double y = -unionMaxY+spacing+2*spacing*n;
		   double z = -unionMaxZ+spacing+2*spacing*o;

		  arrayTransformations[m] = UTransform3D(x,y,z,0,0,0);
		  multiUnion->AddNode(*box,arrayTransformations[m]);
           
		  // Preparing "Draw":
		  if (m % carBoxesX == carBoxesX-1)
		  {
			  if (n % carBoxesY == carBoxesY-1)
			  {
				 n = 0;
				 o++;
			  }      
			  else n++;
		  }
	   }
   }

   multiUnion->Voxelize();
   multiUnion->Capacity();

   return multiUnion;
}
