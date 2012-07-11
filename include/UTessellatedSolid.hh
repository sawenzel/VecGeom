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
// * technical work of the GEANT4 collaboration and of QinetiQ Ltd,   *
// * subject to DEFCON 705 IPR conditions.                            *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// $Id: UTessellatedSolid.hh,v 1.11 2010-10-20 08:54:18 gcosmo Exp $
// GEANT4 tag $Name: not supported by cvs2svn $
//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// MODULE:              UTessellatedSolid.hh
//
// Date:                15/06/2005
// Author:              P R Truscott
// Organisation:        QinetiQ Ltd, UK
// Customer:            UK Ministry of Defence : RAO CRP TD Electronic Systems
// Contract:            C/MAT/N03517
//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// CHANGE HISTORY
// --------------
// 22 November 2005, F Lei, 
//  - Added GetPolyhedron()
//
// 31 October 2004, P R Truscott, QinetiQ Ltd, UK
//  - Created.
//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// Class description:
//
//    UTessellatedSolid is a special Geant4 solid defined by a number of 
//    facets (UVFacet). It is important that the supplied facets shall form a
//    fully enclose space which is the solid. 
//    At the moment only two types of facet can be used for the construction of 
//    a UTessellatedSolid, i.e. the UTriangularFacet and UQuadrangularFacet.
//
//    How to contruct a UTessellatedSolid:
//  
//    First declare a tessellated solid:
//
//      UTessellatedSolid* solidTarget = new UTessellatedSolid("Solid_name");
//
//    Define the facets which form the solid
// 
//      double targetSiz = 10*cm ;
//      UTriangularFacet *facet1 = new
//      UTriangularFacet (UVector3(-targetSize,-targetSize,        0.0),
//                         UVector3(+targetSize,-targetSize,        0.0),
//                         UVector3(        0.0,        0.0,+targetSize),
//                         ABSOLUTE);
//      UTriangularFacet *facet2 = new
//      UTriangularFacet (UVector3(+targetSize,-targetSize,        0.0),
//                         UVector3(+targetSize,+targetSize,        0.0),
//                         UVector3(        0.0,        0.0,+targetSize),
//                         ABSOLUTE);
//      UTriangularFacet *facet3 = new
//      UTriangularFacet (UVector3(+targetSize,+targetSize,        0.0),
//                         UVector3(-targetSize,+targetSize,        0.0),
//                         UVector3(        0.0,        0.0,+targetSize),
//                         ABSOLUTE);
//      UTriangularFacet *facet4 = new
//      UTriangularFacet (UVector3(-targetSize,+targetSize,        0.0),
//                         UVector3(-targetSize,-targetSize,        0.0),
//                         UVector3(        0.0,        0.0,+targetSize),
//                         ABSOLUTE);
//      UQuadrangularFacet *facet5 = new
//      UQuadrangularFacet (UVector3(-targetSize,-targetSize,      0.0),
//                           UVector3(-targetSize,+targetSize,      0.0),
//                           UVector3(+targetSize,+targetSize,      0.0),
//                           UVector3(+targetSize,-targetSize,      0.0),
//                           ABSOLUTE);
//
//    Then add the facets to the solid:    
//
//      solidTarget->AddFacet((UVFacet*) facet1);
//      solidTarget->AddFacet((UVFacet*) facet2);
//      solidTarget->AddFacet((UVFacet*) facet3);
//      solidTarget->AddFacet((UVFacet*) facet4);
//      solidTarget->AddFacet((UVFacet*) facet5);
//
//    Finally declare the solid is complete:
//
//      solidTarget->SetSolidClosed(true);
//
///////////////////////////////////////////////////////////////////////////////
#ifndef UTessellatedSolid_hh
#define UTessellatedSolid_hh 1

#include "VUSolid.hh"
#include "UFacet.hh"

#include "UVoxelFinder.hh"

//#include "UVGraphicsScene.hh"
//#include "UVPVParameterisation.hh"
//#include "UVPhysicalVolume.hh"

//#include "UVoxelLimits.hh"
//#include "UAffineTransform.hh"
//#include "UVisExtent.hh"

#include <iostream>
#include <vector>
#include <set>
#include <map>


struct VertexInfo
{
	int id;
	double mag2;
};


class VertexComparator
{
public:
    bool operator()( const VertexInfo &l, const VertexInfo &r)
    {
		return l.mag2 == r.mag2 ? l.id < r.id : l.mag2 < r.mag2;
    }
};

class UTessellatedSolid : public VUSolid
{
  public:  // with description

    UTessellatedSolid ();
    UTessellatedSolid (const char *name);
    virtual ~UTessellatedSolid ();
    
    UTessellatedSolid (const UTessellatedSolid &s);
    const UTessellatedSolid &operator= (const UTessellatedSolid &s);
    const UTessellatedSolid &operator+= (const UTessellatedSolid &right);
   
    bool AddFacet (UFacet *aFacet);
    UFacet *GetFacet (int i) const;
    int GetNumberOfFacets () const;
    
//    virtual double GetCubicVolume ();
    virtual double GetSurfaceArea ();


//
//  virtual void ComputeDimensions (UVPVParameterisation* p, const int n,
//                                  const UVPhysicalVolume* pRep) const;
    
	virtual VUSolid::EnumInside Inside (const UVector3 &p) const;

	VUSolid::EnumInside InsideDummy (const UVector3 &p) const;

	VUSolid::EnumInside InsideWithExclusion(const UVector3 &aPoint, UBits *exclusion) const;

	void Voxelize();

	void PrecalculateInsides();

    void SetRandomVectorSet ();
    virtual bool Normal (const UVector3 &p, UVector3 &aNormal) const;

   double DistanceToInCandidates(const UVector3 &aPoint, const UVector3 &aDirection, double aPstep, std::vector<int> &candidates, UBits &bits) const;

    void DistanceToOutCandidates(const UVector3 &aPoint, const UVector3 &direction, double &minDist, UVector3 &minNormal, bool &aConvex, double aPstep, std::vector<int > &candidates, UBits &bits) const;

	virtual double DistanceToIn(const UVector3 &p, const UVector3 &v, double aPstep = UUtils::kInfinity) const;

	double DistanceToInDummy(const UVector3 &p, const UVector3 &v, double aPstep = UUtils::kInfinity) const;

    virtual double SafetyFromOutside(const UVector3 &p, bool aAccurate=false) const;
    virtual double DistanceToOut(const UVector3 &p,
                                   const UVector3 &v,
                                UVector3       &aNormalVector,
                                bool           &aConvex,
                                double aPstep = UUtils::kInfinity						   

						   ) const;


    double DistanceToOutDummy(const UVector3 &p,
                                   const UVector3 &v,
                                UVector3       &aNormalVector,
                                bool           &aConvex,
                                double aPstep = UUtils::kInfinity						   

						   ) const;

    virtual double SafetyFromInside (const UVector3 &p, bool aAccurate=false) const;
    virtual UGeometryType GetEntityType () const;
    
    void SetSolidClosed (const bool t);

	void CreateVertexList ();

    bool GetSolidClosed () const;

	virtual VUSolid* Clone() const { return 0; }

	int SetAllUsingStack(const std::vector<int> &voxel, const std::vector<int> &max, bool status, UBits &checked);
       
    virtual UVector3 GetPointOnSurface() const;

//    virtual bool CalculateExtent(const EAxis pAxis, const UVoxelLimits& pVoxelLimit, const UAffineTransform& pTransform, double& pMin, double& pMax) const;

    virtual std::ostream &StreamInfo(std::ostream &os) const;

	virtual double Capacity() {return 0;}

	void SetExtremeFacets();

    virtual double SurfaceArea() {return 0; }

   virtual void GetParametersList(int aNumber,double *aArray) const{} 

   virtual UPolyhedron* GetPolyhedron() const{return 0;}

   virtual void ComputeBBox(UBBox *aBox, bool aStore = false) {}

  // Functions for visualization
 
//    virtual void  DescribeYourselfTo (UVGraphicsScene& scene) const;
//    virtual UVisExtent   GetExtent () const;

   void                         Extent (EAxisType aAxis, double &aMin, double &aMax) const;
   void							Extent (UVector3 &aMin, UVector3 &aMax) const;

   /*
    double      GetMinXExtent () const;
    double      GetMaxXExtent () const;
    double      GetMinYExtent () const;
    double      GetMaxYExtent () const;
    double      GetMinZExtent () const;
    double      GetMaxZExtent () const;
	*/

	// when we would have visualization, these routines would be enabled
//    virtual UPolyhedron* CreatePolyhedron () const;
//    virtual UPolyhedron* GetPolyhedron    () const;
//    virtual UNURBS*      CreateNURBS      () const;
 
  protected:  // with description
 
    void DeleteObjects ();
    void CopyObjects (const UTessellatedSolid &s);

//     UVector3List* CreateRotatedVertices(const UAffineTransform& pTransform) const;
      // Create the List of transformed vertices in the format required
      // for VUSolid:: ClipCrossSection and ClipBetweenSections.

  private:

    mutable UPolyhedron* fpPolyhedron;

    std::vector<UFacet *>  facets;
	std::multiset<UFacet *>     extremeFacets; // Does all other facets lie
                                            // on or behind this surface?
    UGeometryType           geometryType;
    double                 cubicVolume;
    double                 surfaceArea;
    std::vector<UVector3>  vertexList;
	std::set<VertexInfo,VertexComparator> facetList;
	UVector3 minExtent, maxExtent;

    bool                   solidClosed;

    double          dirTolerance;
    std::vector<UVector3>     randir;

    int             maxTries;

	UVoxelFinder        voxels;  // Pointer to the vozelized solid

	UBits insides;

	std::vector<UVoxelBox> voxelBoxes;
	std::vector<std::vector<int>> voxelBoxesFaces;
};

#endif
