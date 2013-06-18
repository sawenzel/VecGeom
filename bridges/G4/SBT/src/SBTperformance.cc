
//10:30 - 17:30
//12:30 - 14:30

//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
// SBTperformance
//
// Implementation of the batch solid performance test (Geant4, ROOT and USolidss)
//

//TODO: why are points near surface of polyhedra different in geant4 and root. which one is right?

#include <iomanip>
#include <sstream>
#include <ctime>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "SBTperformance.hh"

#include "G4VSolid.hh"

#include "SBTVisManager.hh"
#include "G4Polyline.hh"
#include "G4Circle.hh"
#include "G4Color.hh"
#include "G4VisAttributes.hh"
#include "G4GeometryTolerance.hh"
#include "G4SolidStore.hh"
#include "G4Timer.hh"

#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Cons.hh"
#include "G4Orb.hh"
#include "G4Sphere.hh"
#include "G4Polycone.hh"
#include "G4Polyhedra.hh"
#include "G4UnionSolid.hh"
#include "G4TessellatedSolid.hh"
#include "G4Trap.hh"
#include "G4GenericTrap.hh"
#include "G4VFacet.hh"
#include "G4ExtrudedSolid.hh"

#include "VUSolid.hh"
#include "UBox.hh"
#include "UOrb.hh"
#include "UTrd.hh"
#include "UMultiUnion.hh"
#include "UTriangularFacet.hh"
#include "UQuadrangularFacet.hh"
#include "UTessellatedSolid.hh"
#include "UTubs.hh"
#include "UCons.hh"
#include "USphere.hh"
#include "UTrap.hh"
#include "UPolycone.hh"
#include "UPolycone2.hh"
#include "UPolycone3.hh"

#include "TGeoArb8.h"
#include "TGeoTrd2.h" 
#include "TGeoBBox.h"
#include "TGeoShape.h"
#include "TGeoSphere.h"
#include "TGeoPcon.h"
#include "TGeoPgon.h"
#include "TGeoTube.h"
#include "TGeoCone.h"
#include "TGeoXtru.h"

#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include "G4AffineTransform.hh"
#include "G4VoxelLimits.hh"
#include "G4GeometryTolerance.hh"

#include "G4Box.hh"
#include "G4Orb.hh"
#include "G4Tubs.hh"
#include "G4Sphere.hh"
#include "G4Cons.hh"
#include "G4Hype.hh"
#include "G4Para.hh"
#include "G4Torus.hh"
#include "G4Trd.hh"
#include "G4Tools.hh"

#include "G4Polyhedron.hh"
#include "HepPolyhedron.h"

#include "TGeoCompositeShape.h"
#include "TGeoMatrix.h"
 
#include "SBTrun.hh"

using namespace std;

SBTperformance::SBTperformance()
{
	SetDefaults();
}

SBTperformance::~SBTperformance()
{

}

void SBTperformance::SetDefaults()
{
	numCheckPoints = 10;
	maxPoints = 10000;
	repeat = 1000;
	insidePercent = 100.0/3;
	outsidePercent = 100.0/3;
	outsideMaxRadiusMultiple = 10;
	outsideRandomDirectionPercent = 50;
	differenceTolerance = 0.01;

	method = "*";
	perftab = perflabels = NULL;

  volumeGeant4 = NULL;
  volumeUSolids = NULL;
  volumeROOT = NULL;
}

// DONE: better random direction distribution
UVector3 SBTperformance::GetRandomDirection() 
{
	double phi = 2.*UUtils::kPi*G4UniformRand();
	double theta = UUtils::ACos(1.-2.*G4UniformRand());
	double vx = std::sin(theta)*std::cos(phi);
	double vy = std::sin(theta)*std::sin(phi);
	double vz = std::cos(theta);
	UVector3 vec(vx,vy,vz);
	vec.Normalize();

//	G4ThreeVector vec(-1 + 2*G4UniformRand(), -1 + 2*G4UniformRand(), -1 + 2*G4UniformRand());
	return vec;
} 

// DONE: all set point methods are performance equivalent


void SBTperformance::TestInsideGeant4(int iteration)
{
	G4ThreeVector point;

//  points[0].Set(1000, 1000, 1000);

	for (int i = 0; i < maxPoints; i++)
	{

//    points[0] = UVector3(46403.38739730667, -18973.09192006442, -59999.9999999999);
//    points[0] = UVector3(-41374.68053161167,      16708.86204374595      ,59999.9999999999 );

//    points[0].Set(0);

    if (i == 0)
      i = i;

    GetVectorGeant4(point, points, i);
		EInside inside = volumeGeant4->Inside(point);
		if (!iteration) resultDoubleGeant4[i] = 2 - (double) inside;
	}
}

void SBTperformance::TestInsideUSolids(int iteration)
{
	UVector3 point;

//  points[0].Set(1000, 1000, 1000);

	for (int i = 0; i < maxPoints; i++)
	{
		GetVectorUSolids(point, points, i);
    if (i == 0)
      i = i;

		VUSolid::EnumInside inside = volumeUSolids->Inside(point);

		if (!iteration) resultDoubleUSolids[i] = (double) inside;
	}
}

void SBTperformance::TestInsideROOT(int iteration)
{ 
	double point[3];
   
	for (int i = 0; i < maxPoints; i++)
	{
//    points[0] = UVector3(46403.38739730667, -18973.09192006442, -59999.9999999999);
//    points[0] = UVector3(-41374.68053161167 ,     16708.86204374595      ,-59999.9999999999 );

    GetVectorRoot(point, points, i);

		bool contains = volumeROOT->Contains(point);

		if (!iteration) resultDoubleRoot[i] = contains ? 0 : 2;
	}
}


void SBTperformance::TestNormalGeant4(int iteration)
{
	G4ThreeVector point, normal;

	for (int i = 0; i < maxPoints; i++)
	{
		GetVectorGeant4(point, points, i);
		normal = volumeGeant4->SurfaceNormal(point);

		if (!iteration) 
    {
      SetVectorGeant4(normal, resultVectorGeant4, i);
    }
	}
}

void SBTperformance::TestNormalUSolids(int iteration)
{
	UVector3 point, normal;

	for (int i = 0; i < maxPoints; i++)
	{
		GetVectorUSolids(point, points, i);
		bool valid = volumeUSolids->Normal(point, normal);

		if (!iteration) 
    {
      resultBoolUSolids[i] = valid;
      SetVectorUSolids(normal, resultVectorUSolids, i);
    }
	}
}

// NEW2: red points if something is bad

// NEW3: linux version
// NEW3: performance meausurement in Linux

// TODO: graph with 3 columns with software / vs. concrete results

void SBTperformance::TestNormalROOT(int iteration)
{
	double point[3], direction[3]={0, 0, 0}, normal[3];
	UVector3 vect;

	for (int i = 0; i < maxPoints; i++)
	{
		GetVectorRoot(point, points, i);
		volumeROOT->ComputeNormal(point, direction, normal);

		if (!iteration)
		{
			// NEW: dot product of vector should be > zero
			vect.Set(normal[0], normal[1], normal[2]);
			SetVectorUSolids(vect, resultVectorRoot, i);
		}
	}
}


double SBTperformance::ConvertInfinities(double value)
{
	double rInfinity = TGeoShape::Big();
	double gInfinity = kInfinity;

	// NEW: possible bug with two infinities in USolidss
	double uInfinity = UUtils::Infinity();
	double uInfinity2 = UUtils::kInfinity;

	if (value >= gInfinity || value >= rInfinity || value >= uInfinity || value >= uInfinity2) value = gInfinity;
	return value;
}

void SBTperformance::TestSafetyFromOutsideGeant4(int iteration)
{
	G4ThreeVector point;

	for (int i = 0; i < maxPoints; i++)
	{
//    points[0] = UVector3(2.981820718641945, -1.608431647524929, -2.547022407577875);

		GetVectorGeant4(point, points, i);
		double res = volumeGeant4->DistanceToIn(point);

		if (!iteration)
		{
			resultDoubleGeant4[i] = ConvertInfinities (res);

//			CheckPointsOnSurfaceOfOrb(point, res, numCheckPoints, kInside);
		}
	}
}

void SBTperformance::TestSafetyFromOutsideUSolids(int iteration)
{
	UVector3 point;

	for (int i = 0; i < maxPoints; i++)
	{
//    points[0] = UVector3(2.981820718641945, -1.608431647524929, -2.547022407577875);

		GetVectorUSolids(point, points, i);
		// NEW: detected significant performance drop at USolidss DistanceToIn , DistanceToOut was fixed by rewriting of USolidss interface
		double res = volumeUSolids->SafetyFromOutside(point);

    if (i == 31173)
      i = i;

		if (!iteration)
		{
			resultDoubleUSolids[i] = ConvertInfinities (res);

			G4ThreeVector p (point.x, point.y, point.z);

//			CheckPointsOnSurfaceOfOrb(p, res, numCheckPoints, kInside);
		}
	}
}

void SBTperformance::TestSafetyFromOutsideROOT(int iteration)
{
	double point[3];

	for (int i = 0; i < maxPoints; i++)
	{
		GetVectorRoot(point, points, i);
		double res = volumeROOT->Safety(point, false);

		if (!iteration)
		{
			resultDoubleRoot[i] = ConvertInfinities (res);

			G4ThreeVector p (point[0], point[1], point[2]);

//			CheckPointsOnSurfaceOfOrb(p, res, numCheckPoints, kInside);
		}
	}
}

void SBTperformance::CheckPointsOnSurfaceOfOrb(const G4ThreeVector &point, double radius, int count, EInside location)
{
	if (radius < kInfinity && radius > 10*VUSolid::Tolerance())
	{
		G4Orb &orb = *new G4Orb("CheckPointsOnSurfaceOfOrb", radius);
		G4ThreeVector p;
		for (int i = 0; i < count; i++)
		{
			p = orb.GetPointOnSurface() + point;
			EInside e = volumeGeant4->Inside(p);
			if (e == location)
			{
				e = e;
//				cout << "A point on orb constructed with radius of computed safety was found inside/outside.\n";
//				exit(1);
			}

		}
	}
}

void SBTperformance::TestSafetyFromInsideGeant4(int iteration)
{
	G4ThreeVector point;

	for (int i = 0; i < maxPoints; i++)
	{
		GetVectorGeant4(point, points, i);
		double res = volumeGeant4->DistanceToOut(point);

		if (!iteration)
		{
			resultDoubleGeant4[i] = ConvertInfinities (res);

//			CheckPointsOnSurfaceOfOrb(point, res, numCheckPoints, kOutside);
		}
	}
}

void SBTperformance::TestSafetyFromInsideUSolids(int iteration)
{
	UVector3 point;

	for (int i = 0; i < maxPoints; i++)
	{
		GetVectorUSolids(point, points, i);
		double res = volumeUSolids->SafetyFromInside(point);

		if (!iteration)
		{
			resultDoubleUSolids[i] = ConvertInfinities (res);

			G4ThreeVector p (point.x, point.y, point.z);

//			CheckPointsOnSurfaceOfOrb(p, res, numCheckPoints, kOutside);
		}
	}
}

void SBTperformance::TestSafetyFromInsideROOT(int iteration)
{
	double point[3];

	for (int i = 0; i < maxPoints; i++)
	{
		GetVectorRoot(point, points, i);
		double res = volumeROOT->Safety(point, true);

		if (!iteration) 
		{
			resultDoubleRoot[i] = ConvertInfinities (res);
		
			G4ThreeVector p (point[0], point[1], point[2]);

//			CheckPointsOnSurfaceOfOrb(p, res, numCheckPoints, kOutside);
		}
	}
}


// start, point, get distance, propagate point to the surface to the surface (i.e. get point near the surface), at that point computete normal in C++ store it

// ask dot product with original direction in matlab

// problem only if the dot product is larger than 1e-8, only for "scratching" points

void SBTperformance::PropagatedNormal(const G4ThreeVector &point, const G4ThreeVector &direction, double distance, G4ThreeVector &normal)
{
	normal.set(0,0,0);
	if (distance < kInfinity)
	{
		G4ThreeVector shift = distance * direction;
		G4ThreeVector surfacePoint = point + shift;
		normal = volumeGeant4->SurfaceNormal(surfacePoint);
		EInside e = volumeGeant4->Inside(surfacePoint);
		if (e != kSurface)
			e = e;
	}
}

void SBTperformance::PropagatedNormalU(const UVector3 &point, const UVector3 &direction, double distance, UVector3 &normal)
{
  normal.Set(0);
  if (distance < kInfinity)
  {
    UVector3 shift = distance * direction;
    UVector3 surfacePoint = point + shift;
    volumeUSolids->Normal(surfacePoint, normal);
    VUSolid::EnumInside e = volumeUSolids->Inside(surfacePoint);
    if (e != VUSolid::eSurface)
      e = e;
  }
}

void SBTperformance::TestDistanceToInGeant4(int iteration)
{
	G4ThreeVector point, direction;

	for (int i = 0; i < maxPoints; i++)
	{
    /*
    points[0] = UVector3(-3980913.296893585, -21552269.41233081, -21552269.41233081);
    directions[0] = UVector3(0.1908906498134092, 0.7127580504382666, 0.7127580504382666);
    points[1] = UVector3(7341580.738867766, 20654134.81170427, 20654134.81170427) ;
    directions[1] = UVector3(-0.1759615598226392, -0.6548197707145482, -0.6548197707145482);
    */
        GetVectorGeant4(point, points, i);
		GetVectorGeant4(direction, directions, i);
    double res = volumeGeant4->DistanceToIn(point, direction);

    if (i == 157)
      i = i;

		if (!iteration) 
		{
			res = ConvertInfinities (res);
			resultDoubleGeant4[i] = res;

//			CheckPointsOnSurfaceOfOrb(p, res, numCheckPoints,, kInside);
			G4ThreeVector normal;
			PropagatedNormal(point, direction, res, normal);
			SetVectorGeant4(normal, resultVectorGeant4, i);
		}
	}
}

void SBTperformance::TestDistanceToInUSolids(int iteration)
{
	UVector3 point, direction;

	for (int i = 0; i < maxPoints; i++)
	{
//    points[0] = UVector3(-4.5, 0, 0);
//    directions[0] = UVector3(1, 0, 0);

		GetVectorUSolids(point, points, i);
		GetVectorUSolids(direction, directions, i);
		double res = volumeUSolids->DistanceToIn(point, direction);
    if (i == 18950) 
      i = i;

		if (!iteration) 
		{
			resultDoubleUSolids[i] = ConvertInfinities (res);

			G4ThreeVector p (point.x, point.y, point.z);
//			CheckPointsOnSurfaceOfOrb(p, res, numCheckPoints, kInside);	

      UVector3 normal;
      PropagatedNormalU(point, direction, res, normal);
      SetVectorUSolids(normal, resultVectorUSolids, i);
		}
	}
}

void SBTperformance::TestDistanceToInROOT(int iteration)
{
	double point[3], direction[3];

	for (int i = 0; i < maxPoints; i++)
	{
    /*
    points[0] = UVector3(-3980913.296893585, -21552269.41233081, -21552269.41233081);
    directions[0] = UVector3(0.1908906498134092, 0.7127580504382666, 0.7127580504382666);
    points[1] = UVector3(7341580.738867766, 20654134.81170427, 20654134.81170427) ;
    directions[1] = UVector3(-0.1759615598226392, -0.6548197707145482, -0.6548197707145482);
    */

		GetVectorRoot(point, points, i);
		GetVectorRoot(direction, directions, i);
		double res = volumeROOT->DistFromOutside(point, direction);

		if (!iteration) 
		{
			res = ConvertInfinities (res);
			resultDoubleRoot[i] = res;

//			CheckPointsOnSurfaceOfOrb(p, res, numCheckPoints, kInside);

			G4ThreeVector p(point[0], point[1], point[2]);
			G4ThreeVector d(direction[0], direction[1], direction[2]);
			G4ThreeVector normal;
			PropagatedNormal(p, d, res, normal);
			SetVectorGeant4(normal, resultVectorRoot, i);
		}
	}
}

void SBTperformance::TestDistanceToOutGeant4(int iteration)
{
	G4ThreeVector point, direction, normal;
	bool validNorm;

	for (int i = 0; i < maxPoints; i++)
	{
//    points[0] = UVector3(-5, 0, 0);
//    directions[0] = UVector3(1, 0, 0);

		GetVectorGeant4(point, points, i);
		GetVectorGeant4(direction, directions, i);
//    normal.set(0,0,0);
		double res = volumeGeant4->DistanceToOut(point, direction, true, &validNorm, &normal);

		if (!iteration)
		{
      if (!validNorm) normal.set(0, 0, 0);
//			EInside inside = volumeGeant4->Inside(point);
			resultDoubleGeant4[i] = ConvertInfinities (res);
      resultBoolGeant4[i] = validNorm;
      SetVectorGeant4(normal, resultVectorGeant4, i);

//			CheckPointsOnSurfaceOfOrb(point, res, numCheckPoints, kOutside);
		}
	} 
}

void SBTperformance::TestDistanceToOutROOT(int iteration)
{
	double point[3], direction[3];

	for (int i = 0; i < maxPoints; i++)
	{
		GetVectorRoot(point, points, i);
		GetVectorRoot(direction, directions, i);

    if (i == 3) 
      i = i;

    double res = volumeROOT->DistFromInside(point, direction);

		if (!iteration)
		{
			resultDoubleRoot[i] = ConvertInfinities (res);

			G4ThreeVector p(point[0], point[1], point[2]);
//			CheckPointsOnSurfaceOfOrb(p, res, numCheckPoints, kOutside);
		}
	}
}

void SBTperformance::TestDistanceToOutUSolids(int iteration)
{
	UVector3 point,normal,direction;
	bool convex;

	for (int i = 0; i < maxPoints; i++)
	{
//    points[0] = UVector3(-5, 0, 0);
//    directions[0] = UVector3(1, 0, 0);

    GetVectorUSolids(point, points, i);
		GetVectorUSolids(direction, directions, i);

    if (i == 3) 
      i = i;

//    normal.Set(0);
    double res = volumeUSolids->DistanceToOut(point, direction, normal, convex);

		if (!iteration)
		{
      if (convex == false) normal.Set(0);

			resultDoubleUSolids[i] = ConvertInfinities (res);
      resultBoolUSolids[i] = convex;
      SetVectorUSolids(normal, resultVectorUSolids, i);

			G4ThreeVector p (point.x, point.y, point.z);

//			CheckPointsOnSurfaceOfOrb(p, res, numCheckPoints, kOutside);
		}
	}
}

void SBTperformance::FlushSS(stringstream &ss)
{
	string s = ss.str();
	cout << s;
	*log << s;
	ss.str("");
}

void SBTperformance::Flush(const string &s)
{
	cout << s;
	*log << s;
}


// NEW: results written normalized to nano seconds per operation
double SBTperformance::NormalizeToNanoseconds(double time)
{
	double res = ((time * (double) 1e+9) / ((double)repeat * (double)maxPoints));
	return res;
}

double SBTperformance::MeasureTest (void (SBTperformance::*funcPtr)(int), const string &method)
{
	Flush("Measuring performance of method "+method+"\n");

	// NEW: storing data phase, timer is off
	// See: http://www.newty.de/fpt/fpt.html , The Function Pointer Tutorials
	(*this.*funcPtr)(0);

	G4Timer timer; // = new G4Timer();
	timer.Start();

	// performance phase, timer is on
	for (int i = 1; i <= repeat; i++)
	{
		(*this.*funcPtr)(i); 
	}

	timer.Stop();
	double realTime = timer.GetUserElapsed(); 	// NOTE: other methods are GetSystemElapsed, GetRealElapsed

	stringstream ss;
	// NEW: write time per operation / bunch of operations
	ss << "Time elapsed: " << realTime << "s\n";
	ss << "Time per one repeat: " << realTime / repeat << "s\n";
	ss << "Nanoseconds per one method call: " << NormalizeToNanoseconds(realTime) << "\n";
	FlushSS(ss);

	return realTime;
}


// Number of nodes to implement
void SBTperformance::ConvertMultiUnionFromGeant4(UMultiUnion &multiUnion, G4UnionSolid &solid, string &rootString)
{
	if (G4UnionSolid *leaf = (G4UnionSolid *) solid.GetConstituentSolid(0)) 
	{
		ConvertMultiUnionFromGeant4(multiUnion, *leaf, rootString);
		leaf = (G4UnionSolid *) solid.GetConstituentSolid(1);
		ConvertMultiUnionFromGeant4(multiUnion, *leaf, rootString);
	}
	else
	{
		// updating USolids multi-union
		G4DisplacedSolid &node = *(G4DisplacedSolid *) &solid;
		G4AffineTransform at = node.GetTransform();
		G4ThreeVector t = at.NetTranslation();
		G4RotationMatrix r = at.NetRotation();
		t *= -1;

		UTransform3D *transformation = new UTransform3D(t.x(), t.y(), t.z(), r.phi(), r.theta(), r.psi());
		SetupSolids(node.GetConstituentMovedSolid()); // fills volumeUSolids and volumeROOT
		multiUnion.AddNode(*volumeUSolids, *transformation);

		// updating ROOT composite
		std::stringstream count; count << compositeCounter;

		string solidName = string("CS")+count.str();
		string transName = string("t")+count.str();
		volumeROOT->SetName(solidName.c_str());

		TGeoRotation *rotationRoot = new TGeoRotation("", r.phi(), r.theta(), r.psi()); // transformations need names
		TGeoCombiTrans *transformationRoot = new TGeoCombiTrans(transName.c_str(), t.x(), t.y(), t.z(), rotationRoot);
		transformationRoot->RegisterYourself();
		if (rootString != "") rootString += "+";
		rootString += solidName+":"+transName; // "(A:t1+B:t2)"
		compositeCounter++;
	}
}


void SBTperformance::SetupSolids(G4VSolid *testVolume)
{
	string type(testVolume->GetEntityType());

	volumeUSolids = NULL;
	volumeROOT = NULL;
	stringstream ss;

	// DONE: box and other USolidss are not slow anymore (it linking issue at VS2010)
	if (type == "G4UnionSolid" || type == "G4DisplacedSolid")
	{
		G4UnionSolid &unionSolid = *(G4UnionSolid *) testVolume;

		UMultiUnion &multiUnion = *new UMultiUnion("multiUnion");
		
		string rootComposite = ""; // "(A:t1+B:t2)"
		compositeCounter = 1;
		ConvertMultiUnionFromGeant4(multiUnion, unionSolid, rootComposite);
		volumeUSolids = &multiUnion;
		rootComposite = "("+rootComposite+")";
		volumeROOT = new TGeoCompositeShape("CS", rootComposite.c_str());
//		volumeROOT = NULL;
    
		multiUnion.Voxelize();
		multiUnion.GetVoxels().DisplayListNodes();

		// double capacity = testVolume->GetCubicVolume();
		// double area = testVolume->GetSurfaceArea();

		multiUnion.Capacity();

		ss << "UMultiUnion()";
	}
    if (type == "G4TessellatedSolid")
    {
        UTessellatedSolid &utessel = *new UTessellatedSolid("ts");

        G4TessellatedSolid &tessel = *(G4TessellatedSolid *) testVolume;
        int n = tessel.GetNumberOfFacets();
        for (int i = 0; i < n; i++)
        {
            G4VFacet &facet = *tessel.GetFacet(i);
            int verticesCount = facet.GetNumberOfVertices();
            vector<UVector3> v(verticesCount);
            for (int j = 0; j < verticesCount; j++) 
            {
                G4ThreeVector vec = facet.GetVertex(j);
                v[j].Set(vec.x(), vec.y(), vec.z());
            }
            VUFacet *ufacet;
            switch (verticesCount)
            {
                case 3:
                    ufacet = new UTriangularFacet(v[0], v[1], v[2], UABSOLUTE);
                    break;
                case 4:
                    ufacet = new UQuadrangularFacet(v[0], v[1], v[2], v[3], UABSOLUTE);
                    break;
                default:
                    break;
            }
            utessel.AddFacet(ufacet);
        }
		    utessel.SetMaxVoxels(SBTrun::maxVoxels);
        utessel.SetMaxVoxels(1);
        utessel.SetSolidClosed(true);
        volumeUSolids = &utessel;
    }
    if (type == "G4Box")
	{
		G4Box &box = *(G4Box *) testVolume;
		double x = box.GetXHalfLength();
		double y = box.GetYHalfLength();
		double z = box.GetZHalfLength();
		volumeROOT = new TGeoBBox("UBox", x, y, z);
		volumeUSolids = new UBox("UBox", x, y, z);
		ss << "Ubox("<<x<<","<<y<<","<<z<<")";
	}
	if (type == "G4Orb")
	{
		G4Orb &orb = *(G4Orb *) testVolume;
		double radius = orb.GetRadius();
		volumeROOT = new TGeoSphere("UOrb", 0, radius);
		volumeUSolids = new UOrb("UOrb", radius);
		ss << "UOrb("<<radius<<")";
	}
	if (type == "G4Sphere")
	{
		G4Sphere &sphere = *(G4Sphere *) testVolume;
		double innerRadius = sphere.GetInnerRadius();
		double outerRadius = sphere.GetOuterRadius();
		double sTheta = sphere.GetSTheta();
		double dTheta = sphere.GetDTheta();
		double sPhi  = sphere.GetSPhi();
		double dPhi = sphere.GetDPhi();

		volumeUSolids = new USphere("USphere", innerRadius, outerRadius, sPhi, dPhi, sTheta, dTheta);

		sPhi = 180 * sPhi / UUtils::kPi;
		dPhi = 180 * dPhi / UUtils::kPi;
		dPhi += sPhi;
		if (dPhi > 360) dPhi -= 360; 

		sTheta = 180 * sTheta / UUtils::kPi;
		dTheta = 180 * dTheta / UUtils::kPi;
		dTheta += sTheta;
		if (dTheta > 360) dTheta -= 360;

		volumeROOT = new TGeoSphere("USphere", innerRadius, outerRadius, sTheta, dTheta, sPhi, dPhi);
		ss << "USphere("<<innerRadius<<")";
	}
	if (type == "G4Trd")
	{
		G4Trd &trd = *(G4Trd *) testVolume;
		double x1 = trd.GetXHalfLength1();
		double x2 = trd.GetXHalfLength2();
		double y1 = trd.GetYHalfLength1();
		double y2 = trd.GetYHalfLength2();
		double z = trd.GetZHalfLength(); 
		volumeUSolids = new UTrd("UTrd", x1, x2, y1, y2, z);
		volumeROOT = new TGeoTrd2(x1, x2, y1, y2, z);
		ss << "UTrd("<<x1<<","<<x2<<","<<y1<<","<<y2<<","<<z<<")";
	}
  if (type == "G4GenericTrap")
  {
    G4GenericTrap &trap = *(G4GenericTrap *) testVolume;
    double z = trap.GetZHalfLength();
    std::vector<G4TwoVector> vertices = trap.GetVertices();
    double tVertices[16];
    for (int i = 0; i < vertices.size(); i++)
    {
       G4TwoVector &vec = vertices[i];
       tVertices[2*i] = vec.x();
       tVertices[2*i+1] = vec.y();
    }
    volumeROOT = new TGeoArb8("Trap", z, tVertices);
  }

	if (type == "G4Trap")
	{
		G4Trap &trap = *(G4Trap *) testVolume;
		double x1 = trap.GetXHalfLength1();
		double x2 = trap.GetXHalfLength2();
		double x3 = trap.GetXHalfLength3();
		double x4 = trap.GetXHalfLength4();
		double y1 = trap.GetYHalfLength1();
		double y2 = trap.GetYHalfLength2();
		double z = trap.GetZHalfLength();
		double talpha1 = atan (trap.GetTanAlpha1());
		double talpha2 = atan (trap.GetTanAlpha2());
    double theta = 0; // geant4 does not have method GetTheta???
    double phi = 0; // geant4 does not have phi???
//		volumeUSolids = new UTrd("UTrd", x1, x2, y1, y2, z);

    ////////////////////////////////////////////////////////////////////////////
    //                                                                        //
    // TGeoTrap                                                               //
    //                                                                        //
    // Trap is a general trapezoid, i.e. one for which the faces perpendicular//
    // to z are trapezia and their centres are not the same x, y. It has 11   //
    // parameters: the half length in z, the polar angles from the centre of  //
    // the face at low z to that at high z, H1 the half length in y at low z, //
    // LB1 the half length in x at low z and y low edge, LB2 the half length  //
    // in x at low z and y high edge, TH1 the angle w.r.t. the y axis from the//
    // centre of low y edge to the centre of the high y edge, and H2, LB2,    //
    // LH2, TH2, the corresponding quantities at high z.                      //
    //                                                                        //
    ////////////////////////////////////////////////////////////////////////////

//    TGeoTrap("", dz=z, theta=theta, phi=phi, h1=y1, bl1=x1, tl1=x2, alpha1=talpha1, h2=y2, bl2=x3, tl2=x4, alpha2=talpha2);

    // fTheta; // theta angle
    // fPhi;   // phi angle
    // fH1;    // half length in y at low z
    // fBl1;   // half length in x at low z and y low edge
    // fTl1;   // half length in x at low z and y high edge
    // fAlpha1;// angle between centers of x edges an y axis at low z
    // fH2;    // half length in y at high z
    // fBl2;   // half length in x at high z and y low edge
    // fTl2;   // half length in x at high z and y high edge
    // fAlpha2;// angle between centers of x edges an y axis at low z

    if (z == 60000.0) theta = 20, phi = 5;

    volumeUSolids = new UTrap("Trap", z, UUtils::kPi * theta / 180, UUtils::kPi * phi / 180, y1, x1, x2, talpha1, y2, x3, x4, talpha2);

//    theta = 180 * theta / UUtils::kPi;
//    phi = 180 * phi / UUtils::kPi;
      talpha1 = 180 * talpha1 / UUtils::kPi;
      talpha2 = 180 * talpha2 / UUtils::kPi;
//		volumeROOT = new TGeoTrap("Trap", z, theta, phi, y1, x1, x2, talpha1, y2, x3, x4, ,       z = 60000.000000000000, theta = 20.000000000000000, phi = 5.0000000000000000,      y1 = 40000.000000000000, x1 = 30000.000000000000, x2 = 40000.000000000000,       talpha1 = 10.000000000000000, y2 = 16000.000000000000, x3 = 10000.000000000000, x4 = 14000.000000000000, talpha2 = 10.000000000000000

		volumeROOT = new TGeoTrap("Trap", z, theta, phi, y1, x1, x2, talpha1, y2, x3, x4, talpha2);
//		ss << "UTrd("<<x1<<","<<x2<<","<<y1<<","<<y2<<","<<z<<")";
	} 
	if (type == "G4Tubs")
	{
		G4Tubs &tubs = *(G4Tubs *) testVolume;
//		double deltaPhiAngle = tubs.GetDeltaPhiAngle();
		double rMin = tubs.GetRMin();
		double rMax = tubs.GetRMax();
		double dZ = tubs.GetDz();
		double sPhi = tubs.GetSPhi();
		double dPhi = tubs.GetDPhi();
		
		volumeUSolids = new UTubs("UTubs", rMin, rMax, dZ, sPhi, dPhi);
		
		sPhi = 180 * sPhi / UUtils::kPi;
		dPhi = 180 * dPhi / UUtils::kPi;
		dPhi += sPhi;
		if (dPhi > 360) dPhi -= 360;
		volumeROOT = (sPhi == 0 && dPhi ==  360) ? new TGeoTube(rMin, rMax, dZ) : new TGeoTubeSeg(rMin, rMax, dZ, sPhi, dPhi);
	}
	if (type == "G4Cons")
	{
		G4Cons &cons = *(G4Cons *) testVolume;
		double rMin1 = cons.GetRmin1();
		double rMax1 = cons.GetRmax1();
		double rMin2 = cons.GetRmin2();
		double rMax2 = cons.GetRmax2();
		double dz = cons.GetDz();
		double sPhi = cons.GetSPhi();
		double dPhi = cons.GetDPhi();

		volumeUSolids = new UCons("UCons", rMin1, rMax1, rMin2, rMax2, dz, sPhi, dPhi);

		sPhi = 180 * sPhi / UUtils::kPi;
		dPhi = 180 * dPhi / UUtils::kPi;

		dPhi += sPhi;
		if (dPhi > 360) dPhi -= 360;
		volumeROOT = (sPhi == 0 && dPhi == 360) ? new TGeoCone(dz, rMin1, rMax1, rMin2, rMax2) : new TGeoConeSeg(dz, rMin1, rMax1, rMin2, rMax2, sPhi, dPhi);
	}
        if (type == "G4ExtrudedSolid")
        {
		G4ExtrudedSolid &extru = *(G4ExtrudedSolid *) testVolume;
                int nV = extru.GetNofVertices();
                //G4TwoVector GetVertex(G4int index) const;
		std::vector<G4TwoVector> poly=extru.GetPolygon();
                int nZ=extru.GetNofZSections();
                //ZSection    GetZSection(G4int index) const;
		//vector<ZSection> GetZSections() const;
                double vx[100],vy[100];
                //extru.DumpInfo();
                

         for ( G4int i=0; i<nV; ++i )
         {
          vx[i]=poly[i].x();
          vy[i]=poly[i].y();
          }
	 //volumeRoot      
         TGeoXtru *xr =new TGeoXtru( nZ);
	 xr->DefinePolygon(nV,vx,vy);

         for ( int iz=0; iz<nZ; ++iz ) 
         {
         xr->DefineSection(iz,extru.GetZSection(iz).fZ,extru.GetZSection(iz).fOffset.x(),extru.GetZSection(iz).fOffset.y(),extru.GetZSection(iz).fScale);
         }     

         volumeROOT=xr;
  
        }

	if (type == "G4Polyhedra")
	{
		G4Polyhedra &polyhedra = *(G4Polyhedra *) testVolume;

		double startPhi = 180*polyhedra.GetStartPhi()/UUtils::kPi;
		double endPhi = 180*polyhedra.GetEndPhi()/UUtils::kPi;
		int numRZCorner = polyhedra.GetNumRZCorner();
		int numSide = polyhedra.GetNumSide();

		vector<double> rmin(numRZCorner);
		vector<double> rmax(numRZCorner);
		vector<double> z(numRZCorner);

		TGeoPgon *gonRoot = new TGeoPgon(startPhi, endPhi, numSide, numRZCorner);
		for (int i = 0; i < numRZCorner; i++)
		{
			G4PolyhedraSideRZ sideRZ = polyhedra.GetCorner(i);
			gonRoot->DefineSection(i, sideRZ.z, 0, sideRZ.r);
			cout << sideRZ.r << ":" << sideRZ.z << endl;
		}

		if (false)
		{
			// /solid/G4Polyhedra 0 360 8 4 (1.0,1.2,1.4, 1.2) (-1.0,0,1.0,2.0)
			double rootR[] = {0,1.2,1.4, 0};
			double rootZ[] = {-1.0,0,1.0, 2.0};
			int len = sizeof(rootZ)/sizeof(double);
			gonRoot = new TGeoPgon(startPhi, endPhi, numSide, len);
			for (int i = 0; i < len; i++) gonRoot->DefineSection(i, 1000*rootZ[i], 0, 1000*rootR[i]);
		}

		volumeROOT = gonRoot;
	}
	if (type == "G4Polycone")
	{
		G4Polycone &polycone = *(G4Polycone *) testVolume;
		double startPhi = 180*polycone.GetStartPhi()/UUtils::kPi;
		double endPhi = 180*polycone.GetEndPhi()/UUtils::kPi;

		G4PolyconeHistorical *parameter = polycone.GetOriginalParameters();
    int numPlanes = parameter->Num_z_planes;

    G4double *rmin = parameter->Rmin;
    G4double *rmax = parameter->Rmax;
    G4double *z = parameter->Z_values;

		TGeoPcon *conRoot;
    VUSolid *conUSolid = NULL;

		conRoot = new TGeoPcon(startPhi, endPhi, numPlanes);
    for (int i = 0; i < numPlanes; i++) 
      conRoot->DefineSection(i, z[i], rmin[i], rmax[i]);

    conUSolid = new UPolycone3("", polycone.GetStartPhi(), polycone.GetEndPhi(), numPlanes, &z[0], &rmin[0], &rmax[0]);

//    conUSolid = new UPolycone2("", polycone.GetStartPhi(), polycone.GetEndPhi(), numPlanes, &z[0], &rmin[0], &rmax[0]);

		volumeROOT = conRoot;
    volumeUSolids = conUSolid;
	}

	if (type == "G4PolyconeG") // THis code tests generic polycone!!!
	{
		G4Polycone &polycone = *(G4Polycone *) testVolume;
		double startPhi = 180*polycone.GetStartPhi()/UUtils::kPi;
		double endPhi = 180*polycone.GetEndPhi()/UUtils::kPi;
		int numRZCorner = polycone.GetNumRZCorner();

		vector<double> rmin(numRZCorner);
		vector<double> rmax(numRZCorner);
		vector<double> z(numRZCorner);

		G4PolyconeHistorical *parameter = polycone.GetOriginalParameters();

		int back = 0;

		for (int i = 0; i < numRZCorner; i++)
		{
			G4PolyconeSideRZ sideRZ = polycone.GetCorner(i);
			// we disabled putting back stream, because it actually never worked for general cases
			if (false && i == numRZCorner/2 && z[i-1] == sideRZ.z) back = i;
			if (back)
			{
				int j = back - 1 - (i - back);
				double r = rmax[j];	
				rmin[j] = std::min(r, sideRZ.r);
				rmax[j] = std::max(r, sideRZ.r);
			}
			else
			{
				rmin[i] = 0;
				rmax[i] = sideRZ.r;
				z[i] = sideRZ.z;
			}
		}

		TGeoPcon *conRoot;
    VUSolid *conUSolid = NULL;

		if (numRZCorner == 15)
		{
			double rootR[] = {1, 1, 2, 2, 3, 3, 1, 4, 1, 1.2};
			double rootZ[] = {1, 2, 2, 3, 3, 4, 10, 15, 15, 18};
			int len = sizeof(rootZ)/sizeof(double);
			conRoot = new TGeoPcon(startPhi, endPhi, len);
			for (int i = 0; i < len; i++) conRoot->DefineSection(i, 1000*rootZ[i], 0, 1000*rootR[i]);
		}
		else if (numRZCorner == 6)
		{
			// /solid/G4Polycone 0 360 8 (1, 1, 1, 0, 0, 1.5,1.5, 1.2) (1,2,3,4,4,3,2,1) 

			double rootRmax[] = {1.2, 1.5,1.5, 0};
			double rootRmin[] = {1, 1, 1, 0};
			double rootZ[] = {1,2,3,4};
			int len = sizeof(rootZ)/sizeof(double);
			conRoot = new TGeoPcon(startPhi, endPhi, len);
			for (int i = 0; i < len; i++) conRoot->DefineSection(i, 1000*rootZ[i], 1000*rootRmin[i], 1000*rootRmax[i]);
		}
		else
		{
			int len = numRZCorner - back;
			conRoot = new TGeoPcon(startPhi, endPhi, len);
      conUSolid = new UPolycone("", polycone.GetStartPhi(), polycone.GetEndPhi(), numRZCorner, &rmax[0], &z[0]);
//      conUSolid = new UPolycone3("", polycone.GetStartPhi(), polycone.GetEndPhi(), numRZCorner, &z[0], &rmin[0], &rmax[0]);

			for (int i = 0; i < len; i++) 
			{
				conRoot->DefineSection(i, z[i], rmin[i], rmax[i]);
			}
		}

		volumeROOT = conRoot;
    volumeUSolids = conUSolid;

/*
		for (int i = 0; i < parameter->Num_z_planes; i++) 
		{
			double rmin = std::min(parameter->Rmin[i], parameter->Rmax[i]);
			double rmax = std::max(parameter->Rmin[i], parameter->Rmax[i]);
			conRoot->DefineSection(i, parameter->Z_values[i], rmin, rmax);
		}
		*/

		double capacityRoot = conRoot->Capacity();
		double areaRoot = conRoot->GetFacetArea();
//		double difCapacity = (capacityRoot - capacity) / capacity;
//		double phiTotal = polycone->GetTolerance
	}

	volumeString = ss.str();
}


void SBTperformance::CreatePointsAndDirectionsSurface()
{
	UVector3 norm, point; 

	for (int i = 0; i < maxPointsSurface; i++)
	{
		double random = G4UniformRand();

    G4ThreeVector pointG4;
    do 
    {
      pointG4 = volumeGeant4->GetPointOnSurface();
    }
    while (volumeGeant4->Inside(pointG4) != kSurface);

		UVector3 vec = GetRandomDirection();
		directions[i] = vec;

		UVector3 point;
		G4ThreeVector pointTest;
		int attempt = 0;
    EInside position;

		do
		{
      point.Set(pointG4.getX(), pointG4.getY(), pointG4.getZ());
      if (0) 
      {
        position = volumeGeant4->Inside(pointG4);
        break;
      }

			attempt++;
			if (attempt > 2) 
				attempt = attempt;

			// NEW: points will not be exactly on surface, but near the surface +- (default) 10 tolerance, configurable for BOTH

//			point.x += RandomIncrease();
//			point.y += RandomIncrease();
//			point.z += RandomIncrease();

			pointTest.set(point.x, point.y, point.z);
      position = volumeGeant4->Inside(pointTest);
		}
		while (position != kSurface && attempt < 100);

    if (position != kSurface)
      cout << "Warning, point " << i << " is still on surface" << endl;

//		if (i == 0) point.Set(0, 0, -1000.001);
		points[i+offsetSurface] = point;
	}
}

void SBTperformance::CreatePointsAndDirectionsOutside()
{
	G4VisExtent extent = volumeGeant4->GetExtent();
	double rOut = extent.GetExtentRadius();
	G4Point3D center = extent.GetExtentCenter();
	G4ThreeVector centerG4(center.x(), center.y(), center.z());

	int maxOrbsOutside = 10; // do not make parametrization using macro file
	G4Orb &orbOut = *new G4Orb("OrbOut", 1);

	double rb = 1.5 * rOut;
	G4Box *box = new G4Box("BoxOut", rb, rb, rb);

	for (int i = 0; i < maxPointsOutside; i++)
	{
		double ratio = (double) i / maxPointsOutside;
		int stage = (int) (maxOrbsOutside * ratio);

		// 10 times radius outside sphere will be max
		double radiusRatio = 1 + outsideMaxRadiusMultiple * (pow(2.0, stage) - 1) / pow(2.0, maxOrbsOutside); // NEW: made  parametrization using macro file (default = 10)
		orbOut.SetRadius(rOut * radiusRatio);

		UVector3 vec, point;

		G4ThreeVector pointG4 = orbOut.GetPointOnSurface();
		pointG4 += centerG4;

		// NEW: outside points randomly generated inside 10 times bounding box, but not inside solids
		do
		{
			point.x = -1 + 2 * G4UniformRand();
			point.y = -1 + 2 * G4UniformRand(); 
			point.z = -1 + 2 * G4UniformRand();
			point *= rOut * outsideMaxRadiusMultiple;
			pointG4.set(point.x, point.y, point.z);
		}
		while (volumeGeant4->Inside(pointG4));

		/*
		// NEW: outside points randomly generated on the surface of box 1.5 times bounding box
		pointG4 = box->GetPointOnSurface();
		pointG4 += centerG4;

		point.Set(pointG4.getX(), pointG4.getY(), pointG4.getZ());
		if (volumeGeant4->Inside(pointG4))
		{
			cout << "Outside point is detected as inside!";
			exit(1);
		}
		*/

		double random = G4UniformRand();
		if (random <= outsideRandomDirectionPercent/100.) // NEW: made parametrization using macro file - 0 - 100%
		{
			vec = GetRandomDirection();
		}
		else
		{
			G4ThreeVector pointSurfaceG4 = volumeGeant4->GetPointOnSurface();
			UVector3 pointSurface(pointSurfaceG4.getX(), pointSurfaceG4.getY(), pointSurfaceG4.getZ());
			vec = pointSurface - point;
			vec.Normalize();
		}
		points[i+offsetOutside] = point;
		directions[i+offsetOutside] = vec;
	}
}

// DONE: inside points generation uses random points inside bounding box
void SBTperformance::CreatePointsAndDirectionsInside()
{
	G4VisExtent extent = volumeGeant4->GetExtent();

	int i = 0; 
	while (i < maxPointsInside)
	{
		double x = RandomRange(extent.GetXmin(), extent.GetXmax());
    double y = RandomRange(extent.GetYmin(), extent.GetYmax());
    if (extent.GetYmin() == extent.GetYmax())
      y = RandomRange(-1000, +1000);
		double z = RandomRange(extent.GetZmin(), extent.GetZmax());
		G4ThreeVector pointG4(x, y, z);
		if (volumeGeant4->Inside(pointG4))
		{
			UVector3 point(x, y, z);
			UVector3 vec = GetRandomDirection();
			points[i+offsetInside] = point;
			directions[i+offsetInside] = vec;
			i++;
		}
	}
}

void SBTperformance::CreatePointsAndDirections()
{ 
	maxPointsInside = (int) (maxPoints * (insidePercent/100));
	maxPointsOutside = (int) (maxPoints * (outsidePercent/100));
	maxPointsSurface = maxPoints - maxPointsInside - maxPointsOutside;

	offsetInside = 0;
	offsetSurface = maxPointsInside;
	offsetOutside = offsetSurface + maxPointsSurface;

	points.resize(maxPoints);
	directions.resize(maxPoints);

	resultDoubleDifference.resize(maxPoints);
	resultDoubleGeant4.resize(maxPoints);
	resultDoubleRoot.resize(maxPoints);
  resultBoolGeant4.resize(maxPoints);
  resultBoolUSolids.resize(maxPoints);
  resultDoubleUSolids.resize(maxPoints);

	resultVectorDifference.resize(maxPoints);
	resultVectorGeant4.resize(maxPoints);
	resultVectorRoot.resize(maxPoints);
	resultVectorUSolids.resize(maxPoints);

	CreatePointsAndDirectionsOutside();
	CreatePointsAndDirectionsInside();
	CreatePointsAndDirectionsSurface();
}


#include <sys/types.h>  // For stat().
#include <sys/stat.h>   // For stat().

#ifdef WIN32
		#include <io.h>   // For access().
		#include <direct.h>   // For access().
#endif

int directoryExists (string s)
{
	#ifdef WIN32
		if (_access(s.c_str(), 0) == 0)
	#endif
		{
			struct stat status;
			stat(s.c_str(), &status);

			return (status.st_mode & S_IFDIR);
		}
	return false;
}

void SBTperformance::CompareResults(double resG, double resR, double resU)
{
	string faster;
	stringstream ss;
	double ratio;

	if (volumeROOT)
	{
		if (resR <= resG)
		{
			faster = "ROOT";
			ratio = resG/resR;
		}
		else
		{
			faster = "Geant4";
			ratio = resR/resG;
		}
		if (ratio > 0.01)
		{
			ss << "\nFaster is " << faster << " (" << ratio << "x)\n";
		}
	} 
	ss << "====================================\n\n"; 
	FlushSS(ss);
}

int SBTperformance::SaveVectorToMatlabFile(const vector<double> &vector, const string &filename)
{
	Flush("Saving vector<double> to "+filename+"\n");

	int res = UUtils::SaveVectorToMatlabFile(vector, filename);

	if (res) 
		Flush ("Unable to create file"+filename+"\n");
	
	return res;
}


int SBTperformance::SaveVectorToMatlabFile(const vector<UVector3> &vector, const string &filename)
{
	Flush("Saving vector<UVector3> to "+filename+"\n");

	int res = UUtils::SaveVectorToMatlabFile(vector, filename);

	if (res) 
		Flush ("Unable to create file"+filename+"\n");

	return res;
}


int SBTperformance::SaveLegend(const string &filename)
{
	vector<double> offsets(3);
	offsets[0] = maxPointsInside, offsets[1] = maxPointsSurface, offsets[2] = maxPointsOutside;
	return SaveVectorToMatlabFile(offsets, filename);
}

// NEW: put results to a file which could be visualized
// NEW: matlab scripts created to visualize the results

int SBTperformance::SaveDoubleResults(const string &filename)
{
	int result = 0;
	
	result += SaveVectorToMatlabFile(resultDoubleGeant4, folder+filename+"Geant4.dat");
	result += SaveVectorToMatlabFile(resultDoubleRoot, folder+filename+"Root.dat");
	result += SaveVectorToMatlabFile(resultDoubleUSolids, folder+filename+"USolids.dat");
//	result += SaveVectorToMatlabFile(resultDoubleDifference, folder+filename+"Dif.dat");
	return result;
}

int SBTperformance::SaveVectorResults(const string &filename)
{
	int result = 0;
	
	result += SaveVectorToMatlabFile(resultVectorGeant4, folder+filename+"Geant4.dat");
	result += SaveVectorToMatlabFile(resultVectorRoot, folder+filename+"Root.dat");
	result += SaveVectorToMatlabFile(resultVectorUSolids, folder+filename+"USolids.dat");
//	result += SaveVectorToMatlabFile(resultVectorDifference, folder+filename+"Dif.dat");
	return result;
}


template <class T> void SBTperformance::VectorDifference(const vector<T> &first, const vector<T> &second, vector<T> &result)
{
	int size = result.size();
	for (int i = 0; i < size; i++) 
	{
		T value1 = first[i];
		T value2 = second[i];
		T difference = value2 - value1;
		result[i] = difference;
	}
}

/*
void VectorDifference(vector<double> &first, vector<double> &second, vector<double> &result)
{
	double size = result.size();
	for (int i = 0; i < size; i++) 
	{
		double value1 = first[i];
		double value2 = second[i];
		double difference = value2 - value1;
		result[i] = difference;
	}
}
*/

// NEW: outside tolarance counts + store whole difference to plot
// NEW: tolerance(s) which will be used for determining number of differences
// NEW: difference tolerance(s) was moved to macro

void SBTperformance::PrintCoordinates (stringstream &ss, const UVector3 &vec, const string &delimiter, int precision)
{ 
	ss.precision(precision);
	ss << vec.x << delimiter << vec.y << delimiter << vec.z;
}

string SBTperformance::PrintCoordinates (const UVector3 &vec, const string &delimiter, int precision)
{
	static stringstream ss;
	PrintCoordinates(ss, vec, delimiter, precision);
	string res(ss.str());
	ss.str("");
	return res;
}

string SBTperformance::PrintCoordinates (const UVector3 &vec, const char *delimiter, int precision)
{
	string d(delimiter);
	return PrintCoordinates(vec, d, precision);
}

void SBTperformance::PrintCoordinates (stringstream &ss, const UVector3 &vec, const char *delimiter, int precision)
{
	string d(delimiter);
	return PrintCoordinates(ss, vec, d, precision);
}


int SBTperformance::CountDoubleDifferences(const vector<double> &differences, const vector<double> &values1, const vector<double> &values2)	 
{
	int countOfDifferences = 0;
	stringstream ss;

	for (int i = 0; i < maxPoints; i++) 
	{
		double value1 = values1[i];
		double value2 = values2[i];
		double dif = differences[i];
		double difference = std::abs (dif);
		if (difference > std::abs (differenceTolerance*value1))
		{
			if (++countOfDifferences <= 10) ss << "Different point found: index " << i << 
				"; point coordinates:" << PrintCoordinates(points[i], ",") << 
				"; direction coordinates:" << PrintCoordinates(directions[i], ",") <<
				"; difference=" << difference << ")" << 
				"; value2 =" << value2 <<
				"; value1 = " << value1 << "\n";
		}
	}
	// NEW: if different point found, 1st hundred printed
	ss << "Number of differences is " << countOfDifferences << "\n";
	FlushSS(ss);
	return countOfDifferences;
}

int SBTperformance::CountDoubleDifferences(const vector<double> &differences)
{
	int countOfDifferences = 0;

	for (int i = 0; i < maxPoints; i++) 
	{
		double difference = std::abs (differences[i]);
		if (difference > differenceTolerance) countOfDifferences++;
	}
	stringstream ss;
	ss << "Number of differences is " << countOfDifferences << "\n";
	FlushSS(ss); 
	return countOfDifferences;
}

// NEW: output values precision setprecision (16)
// NEW: for each method, one file

// NEW: print also different point coordinates

void SBTperformance::VectorToDouble(const vector<UVector3> &vectorUVector, vector<double> &vectorDouble)
{
	UVector3 vec;

	int size = vectorUVector.size();
	for (int i = 0; i < size; i++)
	{
		vec = vectorUVector[i];
		double mag = vec.Mag();
		if (mag > 1.1) 
			mag = 1;
		vectorDouble[i] = mag;
	}
}

void SBTperformance::BoolToDouble(const std::vector<bool> &vectorBool, std::vector<double> &vectorDouble)
{
  int size = vectorBool.size();
  for (int i = 0; i < size; i++)
    vectorDouble[i] = (double) vectorBool[i];
}

void SBTperformance::SavePolyhedra(const string &method)
{
	vector<UVector3> vertices;
	vector<vector<int> > nodes;

	int res = 0;
	if (volumeUSolids)
	res = G4Tools::GetPolyhedra(*volumeUSolids, vertices, nodes);

	if (!res) res = G4Tools::GetPolyhedra(*volumeGeant4, vertices, nodes);

	if (res)
	{
		SaveVectorToMatlabFile(vertices, folder+method+"Vertices.dat");

		ofstream fileQuads((folder+method+"Quads.dat").c_str());
		ofstream fileTriangles((folder+method+"Triangles.dat").c_str());
	
		int size = nodes.size();
		for (int i = 0; i < size; i++)
		{
			int n = nodes[i].size();
			if (n == 3 || n == 4)
			{
				ofstream &file = (n == 3) ? fileTriangles : fileQuads;
				vector<int> node = nodes[i];
				for (int j = 0; j < n; j++) file << setprecision(16) << node[j] << "\t";
				file << endl;
			}
		}
	}
}

int SBTperformance::SaveResultsToFile(const string &method)
{
	string filename(folder+method+"All.dat");
	Flush("Saving all results to "+filename+"\n");
	ofstream file(filename.c_str());
	bool saveVectors = (method == "Normal");
	int prec = 16;
	if (file.is_open())
	{
		file.precision(prec);
		file << volumeString << "\n";
		string spacer("\t");
		for (int i = 0; i < maxPoints; i++)
		{
			// NEW: point coordinates, vector coordinates to separate file: e.g. trd + parameters \n x, y, z, xp, yp, zp, safetyfromoutside, all the values		
			file << PrintCoordinates(points[i], spacer, prec) << spacer << PrintCoordinates(directions[i], spacer, prec) << spacer; 
			if (saveVectors) file << PrintCoordinates(resultVectorGeant4[i], spacer, prec) << spacer << PrintCoordinates(resultVectorRoot[i], spacer, prec) << spacer << PrintCoordinates(resultVectorUSolids[i], spacer, prec) << "\n";
			else file << resultDoubleGeant4[i] << spacer << resultDoubleRoot[i] << spacer << resultDoubleUSolids[i] << "\n";
		}
		return 0;
	}
	Flush("Unable to create file"+filename+"\n");
	return 1;
}

void SBTperformance::CompareAndSaveResults(const string &method, double resG, double resR, double resU)
{
	CompareResults (resG, resR, resU);		

	if (method == "Normal")
	{
		VectorDifference(resultVectorGeant4, resultVectorRoot, resultVectorDifference);
		// convert vector x,y,z values to their magnitudes
		VectorToDouble(resultVectorGeant4, resultDoubleGeant4);
		VectorToDouble(resultVectorUSolids, resultDoubleUSolids);
		VectorToDouble(resultVectorRoot, resultDoubleRoot);
		VectorToDouble(resultVectorDifference, resultDoubleDifference);
		SaveVectorResults(method);

    BoolToDouble(resultBoolUSolids, resultDoubleUSolids);
    resultDoubleRoot.assign(resultDoubleRoot.size(), 0);
    resultDoubleGeant4.assign(resultDoubleGeant4.size(), 0);
    SaveDoubleResults(method+"Valid");
	}
	else
	{
    if (method == "DistanceToOut")
    {
      SaveDoubleResults(method);
      VectorDifference(resultDoubleGeant4, resultDoubleRoot, resultDoubleDifference);
      SaveVectorResults(method+"Normal");
      BoolToDouble(resultBoolGeant4, resultDoubleGeant4);
      BoolToDouble(resultBoolUSolids, resultDoubleUSolids);
      resultDoubleRoot.assign(resultDoubleRoot.size(), 0);
      SaveDoubleResults(method+"Convex");
    }
    else
    {
      VectorDifference(resultDoubleGeant4, resultDoubleRoot, resultDoubleDifference);
      SaveDoubleResults(method);
    }
  }
	if (method == "DistanceToIn")
	{
		SaveVectorResults("DistanceToInSurfaceNormal");
	}
	SaveLegend(folder+method+"Legend.dat");
//	SaveResultsToFile(method);

	string name = volumeGeant4->GetName();
	// if (name != "MultiUnion") 

#ifdef DEBUG
	Flush("Saving polyhedra for visualization\n");
	SavePolyhedra(method);
#endif
	SaveVectorToMatlabFile(points, folder+method+"Points.dat");
	SaveVectorToMatlabFile(directions, folder+method+"Directions.dat");

	if (volumeROOT)
	{
		Flush("Differences betweeen Geant4 and ROOT\n");
		CountDoubleDifferences(resultDoubleDifference, resultDoubleGeant4, resultDoubleRoot);
	}

	*perflabels << method << "\n";
	perflabels->flush();
	*perftab << NormalizeToNanoseconds(resG) << "\t" << NormalizeToNanoseconds(resR) << "\t" << NormalizeToNanoseconds(resU) << "\n";
	perftab->flush();
}

void SBTperformance::CompareInside()
{
	double resG=0, resU=0, resR=0;

	resG = MeasureTest (&SBTperformance::TestInsideGeant4, "G4VSolid::Inside()");
	if (volumeROOT) resR = MeasureTest (&SBTperformance::TestInsideROOT, "TGeoShape::Contains()");
	if (volumeUSolids) resU = MeasureTest (&SBTperformance::TestInsideUSolids, "VUSolid::Inside()");

	CompareAndSaveResults ("Inside", resG, resR, resU);
}

// NEW: fixed 0.1184...
// NEW: vec a - vec b, than magnitude
// NEW3: fix compare normal method, vectors difference should be 0, not between 0 and 1; works in orb and box; not in trd

void SBTperformance::CompareNormal()
{
	double resG=0, resU=0, resR=0;

	resG = MeasureTest (&SBTperformance::TestNormalGeant4, "G4VSolid::SurfaceNormal()");
	if (volumeROOT) resR = MeasureTest (&SBTperformance::TestNormalROOT, "TGeoShape::ComputeNormal()");
	if (volumeUSolids) resU = MeasureTest (&SBTperformance::TestNormalUSolids, "VUSolid::Normal()");

	for (int i = 0; i < maxPoints; i++)
	{
		//NEW: normal fix
		UVector3 &vect = resultVectorRoot[i];
		if (vect.Dot(resultVectorGeant4[i]) <= 0) vect = -vect;
	}

	CompareAndSaveResults ("Normal", resG, resR, resU);
}

void SBTperformance::CompareSafetyFromInside()
{
	double resG=0, resU=0, resR=0;

	resG = MeasureTest (&SBTperformance::TestSafetyFromInsideGeant4, "G4VSolid::DistanceToOut()");
	if (volumeROOT) resR = MeasureTest (&SBTperformance::TestSafetyFromInsideROOT, "TGeoShape::Safety()");
	if (volumeUSolids) resU = MeasureTest (&SBTperformance::TestSafetyFromInsideUSolids, "VUSolid::SafetyFromInside()");

	CompareAndSaveResults ("SafetyFromInside", resG, resR, resU);
}

void SBTperformance::CompareSafetyFromOutside()
{
	double resG=0, resU=0, resR=0;

	resG = MeasureTest (&SBTperformance::TestSafetyFromOutsideGeant4, "G4VSolid::DistanceToIn()");
	if (volumeROOT) resR = MeasureTest (&SBTperformance::TestSafetyFromOutsideROOT, "TGeoShape::Safety()");
	if (volumeUSolids) resU = MeasureTest (&SBTperformance::TestSafetyFromOutsideUSolids, "VUSolid::SafetyFromOutside ()");

	CompareAndSaveResults ("SafetyFromOutside", resG, resR, resU);
}

void SBTperformance::CompareDistanceToIn()
{
	double resG=0, resU=0, resR=0;

	resG = MeasureTest (&SBTperformance::TestDistanceToInGeant4, "G4VSolid::DistanceToIn()");
	if (volumeROOT) resR = MeasureTest (&SBTperformance::TestDistanceToInROOT, "TGeoShape::DistFromOutside()");
	if (volumeUSolids) resU = MeasureTest (&SBTperformance::TestDistanceToInUSolids, "VUSolid::DistanceToIn()");

	CompareAndSaveResults ("DistanceToIn", resG, resR, resU);
}

void SBTperformance::CompareDistanceToOut()
{
	double resG=0, resU=0, resR=0;

	resG = MeasureTest (&SBTperformance::TestDistanceToOutGeant4, "G4VSolid::DistanceToOut()");
	if (volumeROOT) resR = MeasureTest (&SBTperformance::TestDistanceToOutROOT, "TGeoShape::DistFromInside()");
	if (volumeUSolids) resU = MeasureTest (&SBTperformance::TestDistanceToOutUSolids, "VUSolid::DistanceToOut()");

	CompareAndSaveResults ("DistanceToOut", resG, resR, resU);
}


void SBTperformance::TestMethod(void (SBTperformance::*funcPtr)())
{
	stringstream ss;

	time_t now = time(0);
	time(&now);
	G4String dateTime(ctime(&now));

	ss << "% SBT logged output " << dateTime;
//	s += << "% " << CurrentSolid << G4endl;
	ss << "% Creating " <<  maxPoints << " points and directions\n\n";
	FlushSS(ss);
	CreatePointsAndDirections();

	(*this.*funcPtr)();

	ss << setprecision(20);
	setprecision(20);

//	ss << "" << "Time elapsed: " << dateTime2 << "\n";

	ss << "% Statistics: points=" << maxPoints << ", repeat=" << repeat << ",\n";

	ss << "%             ";
	ss << "surface=" << maxPointsSurface << ", inside=" << maxPointsInside << ", outside=" << maxPointsOutside << "\n";

	ss << "%             cpu time=" << clock()/CLOCKS_PER_SEC << G4endl;
	FlushSS(ss);
}

//will run all tests. in this case, one file stream will be used
void SBTperformance::TestMethodAll()
{
	TestMethod(&SBTperformance::CompareInside);
	TestMethod(&SBTperformance::CompareNormal);
	TestMethod(&SBTperformance::CompareSafetyFromInside);
	TestMethod(&SBTperformance::CompareSafetyFromOutside);
	TestMethod(&SBTperformance::CompareDistanceToIn);
	TestMethod(&SBTperformance::CompareDistanceToOut);
}

void SBTperformance::SetFolder(const string &newFolder)
{
	if (perftab)
	{
		perftab->flush();
		delete perftab;
		perftab = NULL;
	}
	if (perflabels)
	{
		perflabels->flush();
		delete perflabels;
		perflabels = NULL;
	}
	cout << "Checking for existance of " << newFolder << endl;
	if (!directoryExists(newFolder))
	{
		string command;
		#ifdef WIN32
			_mkdir(newFolder.c_str());
		#else
			mkdir(newFolder.c_str(), 0777);
		#endif
		if (!directoryExists(newFolder))
		{
			cout << "Directory "+newFolder+" does not exist, it must be created first\n";
			exit(1);
		}
	}
	folder = newFolder+"/";
}

// DONE: all methods now can be tested separately
// DONE: macros are used to specify and set the concerete test to perform
void SBTperformance::Run(G4VSolid *testVolume, ofstream &logger)
{
	stringstream ss;

	void (SBTperformance::*funcPtr)()=NULL;

	volumeGeant4 = testVolume;

//	SavePolyhedra(method);

	log = &logger;

	cout << "Converting solid" << "\n";
	FlushSS(ss);

	SetupSolids(testVolume);

	if (perftab == NULL) perftab = new ofstream((folder+"Performance.dat").c_str());
	if (perflabels == NULL) perflabels = new ofstream((folder+"PerformanceLabels.txt").c_str());

	if (method == "") method = "*";
	string name = testVolume->GetName();
	ss << "\n\n";
	ss << "===============================================================================\n";
	ss << "Invoking performance test for method " << method << " on " << name << " ..." << "\nFolder is " << folder << std::endl;
	ss << "===============================================================================\n";
	ss << "\n";
	FlushSS(ss);

	if (method == "Inside") funcPtr = &SBTperformance::CompareInside;
	if (method == "Normal") funcPtr = &SBTperformance::CompareNormal;
	if (method == "SafetyFromInside") funcPtr = &SBTperformance::CompareSafetyFromInside;
	if (method == "SafetyFromOutside") funcPtr = &SBTperformance::CompareSafetyFromOutside;
	if (method == "DistanceToIn") funcPtr = &SBTperformance::CompareDistanceToIn;
	if (method == "DistanceToOut") funcPtr = &SBTperformance::CompareDistanceToOut;

	if (method == "*") TestMethodAll();
	else if (funcPtr) TestMethod(funcPtr);
	else ss << "Method " << method << " is not supported" << std::endl;

	perftab->flush();
	perflabels->flush();

	FlushSS(ss);

	if (volumeUSolids) delete volumeUSolids;
	if (volumeROOT) delete volumeROOT;
}

// NEW: *elTubeArgs[6] was causing crash, changed to *elTubeArgs[3]

// safetyfrominside for outside points: 0
// safetyfrominside for outside points: 0

// o: make sphere of points based on distance to in, for safety's, all points on surface must be inside

// DONE: make sphere in matlab for safety's, if one point is specified

/*

Notes for the meeting of September 13 (John, Andrei, Gabiele, Tatiana, Marek)

    A goal: Make a standalone program for testing solids performance:

    Based on precomputed random points. Points can be located randomly on spheres with different radius, with base point 0,0,0 i.e. usually the "center" of the solid. after finishing the calculation of random points, these can be stored to file...

    Using outside/inside of the solid extent - 50/50 ratio; 

    SBT code can be used for inspiration, vectors 50/50 pointing to collision direction to solid surface, or pointing away

    Use timer to measure routines performance

    For points needed to be on surface, in Geant4 there is usefull method GetPointOnSurface()
*/

// NEW2: for each method, graph which shows times in all available solids, for 3 software
// NEW2: vectors of both software in one graph
// NEW3: configure folder for storing in .geant4, where log and results will be found

// DONE: size of the circles of bad points proportional to value of error - the error 
// TODO: length of vectors proportional to values
// NEW2: for different points, print also the values
