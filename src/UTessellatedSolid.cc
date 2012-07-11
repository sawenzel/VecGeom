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
// $Id: UTessellatedSolid.cc,v 1.27 2010-11-02 11:29:07 gcosmo Exp $
// GEANT4 tag $Name: not supported by cvs2svn $
//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// MODULE:              UTessellatedSolid.cc
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
// 22 August 2011,    I Hrivnacova, Orsay, fix in DistanceToOut(p) and
//                    DistanceToIn(p) to exactly compute distance from facet
//                    avoiding use of 'outgoing' flag shortcut variant.
//
// 04 August 2011,    T Nikitina, CERN, added SetReferences() to
//                    CreatePolyhedron() for Visualization of Boolean Operations  
//
// 12 April 2010,     P R Truscott, QinetiQ, bug fixes to treat optical
//                    photon transport, in particular internal reflection
//                    at surface.
//
// 14 November 2007,  P R Truscott, QinetiQ & Stan Seibert, U Texas
//                    Bug fixes to CalculateExtent
//
// 17 September 2007, P R Truscott, QinetiQ Ltd & Richard Holmberg
//                    Updated extensively prior to this date to deal with
//                    concaved tessellated surfaces, based on the algorithm
//                    of Richard Holmberg.  This had been slightly modified
//                    to determine with inside the geometry by projecting
//                    random rays from the point provided.  Now random rays
//                    are predefined rather than making use of random
//                    number generator at run-time.
//
// 22 November 2005, F Lei
//  - Changed ::DescribeYourselfTo(), line 464
//  - added GetPolyHedron()
// 
// 31 October 2004, P R Truscott, QinetiQ Ltd, UK
//  - Created.
//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include "UTessellatedSolid.hh"

// #include "UPolyhedronArbitrary.hh"

#include <iostream>
#include <stack>

#include <iostream>
#include <iomanip>
#include <fstream>

#include <algorithm>

#include <list>

using namespace std;

/// TODO: make a benchmark for automatic selection of number of voxels. random voxels will be selected,
/// than for them methods distancetoin/out and inside will be launched. 

///////////////////////////////////////////////////////////////////////////////
//
// Standard contructor has blank name and defines no facets.
//
UTessellatedSolid::UTessellatedSolid ()
  : VUSolid("dummy"), fpPolyhedron(0), cubicVolume(0.), surfaceArea(0.)
{
  dirTolerance = 1.0E-14;
  
  geometryType = "UTessellatedSolid";
  facets.clear();
  facetList.clear();
  voxelBoxes.clear();
  solidClosed  = false;

  minExtent.Set(UUtils::kInfinity);
  maxExtent.Set(-UUtils::kInfinity);

  SetRandomVectorSet();
}

///////////////////////////////////////////////////////////////////////////////
//
// Alternative constructor. Simple define name and geometry type - no facets
// to detine.
//
UTessellatedSolid::UTessellatedSolid (const char *name)
  : VUSolid(name), fpPolyhedron(0), cubicVolume(0.), surfaceArea(0.)
{
  dirTolerance = 1.0E-14;
  
  geometryType = "UTessellatedSolid";
  facets.clear();
  solidClosed  = false;

  minExtent.Set(UUtils::kInfinity);
  maxExtent.Set(-UUtils::kInfinity);

  SetRandomVectorSet();
}

/*
///////////////////////////////////////////////////////////////////////////////
//
// Fake default constructor - sets only member data and allocates memory
//                            for usage restricted to object persistency.
//
UTessellatedSolid::UTessellatedSolid( __void__& a )
  : UVSolid(a), fpPolyhedron(0), facets(0),
    geometryType("UTessellatedSolid"), cubicVolume(0.), surfaceArea(0.),
    vertexList(), minExtent.x(0.), maxExtent.x(0.),
    minExtent.y(0.), maxExtent.y(0.), minExtent.z(0.), maxExtent.z(0.),
    solidClosed(false), dirTolerance(1.0E-14)
{
  SetRandomVectorSet();
}
*/

///////////////////////////////////////////////////////////////////////////////
//
// Destructor.
//
UTessellatedSolid::~UTessellatedSolid ()
{
  DeleteObjects ();
}

///////////////////////////////////////////////////////////////////////////////
//
// Define copy constructor.
//
UTessellatedSolid::UTessellatedSolid (const UTessellatedSolid &s)
  : VUSolid(s), fpPolyhedron(0)
{
  dirTolerance = 1.0E-14;
  
  geometryType = "UTessellatedSolid";
  facets.clear();
  solidClosed  = false;

  cubicVolume = s.cubicVolume;  
  surfaceArea = s.surfaceArea;  

  minExtent.Set(UUtils::kInfinity);
  maxExtent.Set(-UUtils::kInfinity);

  SetRandomVectorSet();

  CopyObjects (s);
}

///////////////////////////////////////////////////////////////////////////////
//
// Define assignment operator.
//
const UTessellatedSolid &
UTessellatedSolid::operator= (const UTessellatedSolid &s)
{
  if (&s == this)
	  return *this;
  
  // Copy base class data
  //
  VUSolid::operator=(s);

  // Copy data
  //
  cubicVolume = s.cubicVolume;  
  surfaceArea = s.surfaceArea;
  fpPolyhedron = 0; 

  DeleteObjects ();
  CopyObjects (s);
  
  return *this;
}

///////////////////////////////////////////////////////////////////////////////
//
void UTessellatedSolid::DeleteObjects ()
{
  for (std::vector<UFacet *>::iterator f=facets.begin(); f!=facets.end(); ++f)
    delete *f;

  facets.clear();
  facetList.clear();
  vertexList.clear();
  voxelBoxes.clear();
}

///////////////////////////////////////////////////////////////////////////////
//
void UTessellatedSolid::CopyObjects (const UTessellatedSolid &s)
{
  int n = s.GetNumberOfFacets();
  for (int i = 0; i < n; ++i)
  {
    UFacet *facetClone = (s.GetFacet(i))->GetClone();
    AddFacet(facetClone);
  }
  
  if (s.GetSolidClosed()) SetSolidClosed(true);
}


///////////////////////////////////////////////////////////////////////////////
//
// Add a facet to the facet list.  Note that you can add, but you cannot
// delete.
//
bool UTessellatedSolid::AddFacet (UFacet *aFacet)
{
  // Add the facet to the vector.

  if (solidClosed)
  {
//    UException("UTessellatedSolid::AddFacet()", "GeomSolids1002", JustWarning, "Attempt to add facets when solid is closed.");
    return false;
  }
  else if (aFacet->IsDefined())
  {
    set<VertexInfo,VertexComparator>::iterator begin = facetList.begin(), end = facetList.end(), res, j;
    UVector3 p = aFacet->GetCircumcentre();
	VertexInfo value;
	value.id = facetList.size();
	value.mag2 = p.Mag2();

	res = facetList.lower_bound(value);
		 
	j = res;
	bool found = false;
	double pMag2 = p.Mag2();
	while (!found && j != end)
	{
		int id = (*j).id;
		UVector3 q = facets[id]->GetCircumcentre();
		if (found = (facets[id] == aFacet)) break;
		double dif = q.Mag2() - pMag2;
		if (dif > fgTolerance * fgTolerance / 4.0) break;
		j++;
	}

	if (facets.size() > 1)
	{
		j = res;
		double pMag2 = p.Mag2();
		while (!found && j != begin)
		{
			--j;
			int id = (*j).id;
			UVector3 q = facets[id]->GetCircumcentre();
			if (found = (facets[id] == aFacet)) break;
			double dif = pMag2 - q.Mag2();
			if (dif > fgTolerance * fgTolerance / 4.0) break;
		}
	}
   
    if (!found)
    {
        facets.push_back(aFacet);
		facetList.insert(value);
    }

	  /* CURRENTLY SKIPPING THIS CODE. IT WAS WRONG AND WAS THEREFORE NOT NECCESSARY
      UFacetI it    = facets.begin();
      do
      {
        found = (**it == *aFacet);
      } while (!found && ++it!=facets.end());
    
      if (found)
      {
        delete *it;
        facets.erase(it);
      }
      else
	  */    
    return true;
  }
  else
  {
    // UException("UTessellatedSolid::AddFacet()", "GeomSolids1002", JustWarning, "Attempt to add facet not properly defined.");    
    aFacet->StreamInfo(cout);
    return false;
  }
}


int UTessellatedSolid::SetAllUsingStack(const vector<int> &voxel, const vector<int> &max, bool status, UBits &checked)
{
	vector<int> xyz = voxel;

	stack<vector<int> > pos;
	pos.push(xyz);
	int filled = 0;
	int cc = 0, nz = 0;

	vector<int> candidates;

	while (!pos.empty())
	{
		xyz = pos.top();
		pos.pop();
		int index = voxels.GetVoxelsIndex(xyz);
		if (!checked[index])
		{
			checked.SetBitNumber(index, true);
			cc++;

			if (voxels.empty[index])
			{
				filled++;

				insides.SetBitNumber(index, status);

				for (int i = 0; i <= 2; ++i)
				{
					if (xyz[i] < max[i] - 1)
					{
						xyz[i]++;
						pos.push(xyz);
						xyz[i]--;
					}

					if (xyz[i] > 0)
					{
						xyz[i]--;
						pos.push(xyz);
						xyz[i]++;
					}
				}
			}
			else
			{
				nz++;
			}
		}
	}
	return filled;
}

void UTessellatedSolid::PrecalculateInsides()
{
	// DONE: this algorithm makes problems in Inside (for Inside points), see matlab picture left.
	// it is visible, when the number of voxels is reduced using % 100, for the case of orb+box+trd. dela to jen v release verzi
	vector<int> voxel(3), maxVoxels(3);
	for (int i = 0; i <= 2; ++i) maxVoxels[i] = voxels.GetBoundary(i).size();
	unsigned int size = maxVoxels[0] * maxVoxels[1] * maxVoxels[2];

	UBits checked(size-1);
	insides.Clear();
	insides.ResetBitNumber(size-1);

	UVector3 point;
	for (voxel[2] = 0; voxel[2] < maxVoxels[2]; ++voxel[2])
	{
		for (voxel[1] = 0; voxel[1] < maxVoxels[1]; ++voxel[1])
		{
			for (voxel[0] = 0; voxel[0] < maxVoxels[0]; ++voxel[0])
			{
				int index = voxels.GetVoxelsIndex(voxel);
				if (!checked[index] && voxels.empty[index])
				{
					for (int i = 0; i <= 2; ++i) point[i] = voxels.GetBoundary(i)[voxel[i]];
					bool inside = (InsideDummy(point) == eInside) ? true : false;
					int res = SetAllUsingStack(voxel, maxVoxels, inside, checked);
					cout << "SetAllUsingStack " << res << " items with status " << inside << "\n";
				}
				else checked.SetBitNumber(index);

				if (!voxels.empty[index])
				{
					// find a box for corresponding non-empty voxel
					UVoxelBox box;
					for (int i = 0; i <= 2; ++i)
					{
						int index = voxel[i];
						const vector<double> &boundary = voxels.GetBoundary(i);
						double hlen = 0.5 * (boundary[index+1] - boundary[index]);
						box.hlen[i] = hlen;
						box.pos[i] = boundary[index] + hlen;
					}
					voxelBoxes.push_back(box);

					  vector<int> candidates;
					  int limit = voxels.GetCandidatesVoxelArray(box.pos, candidates, NULL);
					  voxelBoxesFaces.push_back(candidates);
					  
				}
			}
		}
	}

	/*
	ofstream file("insides.txt"); insides.Output(file);

	ofstream file2("checked.txt"); checked.Output(file2);

	ofstream file3("empty.txt"); empty.Output(file3);
	*/
}

void UTessellatedSolid::Voxelize ()
{
	voxels.Voxelize(facets);

	if (voxels.empty.GetNbits())
	{
		cout << "Precalculating Insides...\n";
		PrecalculateInsides();
	}
}


void UTessellatedSolid::SetExtremeFacets()
{
//
//
// Compute extremeFacets, i.e. find those facets that have surface
// planes that bound the volume.
// Note that this is going to reject concaved surfaces as being extreme.  Also
// note that if the vertex is on the facet, displacement is zero, so IsInside
// returns true.  So will this work??  Need non-equality
// "bool inside = displacement < 0.0;"
// or
// "bool inside = displacement <= -0.5*fgTolerance" 
// (Notes from PT 13/08/2007).
//
	for (UFacetCI it=facets.begin(); it!=facets.end(); ++it)
	{
		bool isExtreme = true;

		for (int i=0; i< (int) vertexList.size(); i++)
		{
			UFacet &facet = *(*it);
			if (!facet.IsInside(vertexList[i]))
			{
				isExtreme = false;
				break;
			}
		}
		if (isExtreme) extremeFacets.insert(*it);
	}
}


void UTessellatedSolid::CreateVertexList()
{
	  // the algorithm will be:
	  // we will have additional vertexListSorted, where all the items will be sorted by magnitude of vertice vector
	  // new candidate for vertexList - we will determine the position fo first item which would be within it's magnitude - 0.5*fgTolerance. we will go trough until we will reach > +0.5 fgTolerance. comparison (q-p).Mag() < 0.5*fgTolerance will be made
	  // they can be just stored in std::vector, with custom insertion based on binary search

	set<VertexInfo,VertexComparator> vertexListSorted;
	vector<UVector3> vertexListNew;
    set<VertexInfo,VertexComparator>::iterator begin = vertexListSorted.begin(), end = vertexListSorted.end(), res, j;
    UVector3 p;
	VertexInfo value;

    vertexList.clear();
    for (UFacetCI it=facets.begin(); it!=facets.end(); it++)
    {
		UFacet &facet = *(*it);
	  int max = facet.GetNumberOfVertices();
      for (int i = 0; i < max; i++)
      {
        p = facet.GetVertex(i);
		value.id = vertexList.size();
		value.mag2 = p.Mag2();
		res = vertexListSorted.lower_bound(value);
		 
		j = res;
		bool found = false;
		double pMag2 = p.Mag2();
		while (!found && j != end)
		{
			int id = (*j).id;
			UVector3 q = vertexList[id];
			double dif = (q-p).Mag2();
			if (found = (dif < fgTolerance * fgTolerance / 4.0)) break;
			dif = q.Mag2() - pMag2;
			if (dif > fgTolerance * fgTolerance / 4.0) break;
			j++;
		}

		if (vertexList.size() > 1)
		{
			j = res;
			double pMag2 = p.Mag2();
			while (!found && j != begin)
			{
				--j;
				int id = (*j).id;
				UVector3 q = vertexList[id];
				double dif = (q-p).Mag2();
				if (found = (dif < fgTolerance * fgTolerance / 4.0)) break;
				dif = pMag2 - q.Mag2();
				if (dif > fgTolerance * fgTolerance / 4.0) break;
			}
		}

//		cout << "Total checked: " << checked << " from " << vertexList.size() << endl;
    
        if (!found)
        {
          vertexList.push_back(p);
		  vertexListSorted.insert(value);
		  begin = vertexListSorted.begin();
		  end = vertexListSorted.end();
		  facet.SetVertexIndex(i, value.id);

			//
			// Now update the maximum x, y and z limits of the volume.
			//
		    if (value.id == 0) minExtent = maxExtent = p; 
			else
			{
				if (p.x > maxExtent.x) maxExtent.x = p.x;
				else if (p.x < minExtent.x) minExtent.x = p.x;
				if (p.y > maxExtent.y) maxExtent.y = p.y;
				else if (p.y < minExtent.y) minExtent.y = p.y;
				if (p.z > maxExtent.z) maxExtent.z = p.z;
				else if (p.z < minExtent.z) minExtent.z = p.z;
			}
        }
        else
        {
			int index = (*j).id;
          facet.SetVertexIndex(i,index);
        }
      }
	}

#ifdef DEBUG
	double previousValue = 0;
	for (res=vertexListSorted.begin(); res!=vertexListSorted.end(); res++)
	{
		int id = (*res).id;
		UVector3 vec = vertexList[id];
		double value = abs(vec.Mag());
		if (previousValue > value) 
			cout << "Error!" << "\n";
		previousValue = value;
	}
#endif

	  vertexListNew = vertexList;

	  /*
	  // OLD CODE
    vertexList.clear();
    for (UFacetCI it=facets.begin(); it!=facets.end(); it++)
    {
		UFacet &facet = *(*it);
      int m = vertexList.size();
      UVector3 p;
      for (int i=0; i< facet.GetNumberOfVertices(); i++)
      {
        p = facet.GetVertex(i);

        bool found = false;
        int j = 0;
        while (j < m && !found)
        {
          UVector3 q = vertexList[j];
		  found = (q-p).Mag() < 0.5*fgTolerance;
          if (!found) j++;
        }
        if (!found)
        {
          vertexList.push_back(p);
          facet.SetVertexIndex(i,vertexList.size()-1);
        }
        else
        {
          facet.SetVertexIndex(i,j);
        }
      }
    }
	*/
}

///////////////////////////////////////////////////////////////////////////////
//
void UTessellatedSolid::SetSolidClosed (const bool t)
{
   // TODO: this is very slow, must be rewritten. maybe is not neccessary
  if (t)
  {
	  CreateVertexList();
	  
	  SetExtremeFacets();
	  
	  Voxelize();
  }  
  solidClosed = t;
}

///////////////////////////////////////////////////////////////////////////////
//
// GetSolidClosed
//
// Used to determine whether the solid is closed to adding further facets.
//
bool UTessellatedSolid::GetSolidClosed () const
{
	return solidClosed;
}

///////////////////////////////////////////////////////////////////////////////
//
// operator+=
//
// This operator allows the user to add two tessellated solids together, so
// that the solid on the left then includes all of the facets in the solid
// on the right.  Note that copies of the facets are generated, rather than
// using the original facet set of the solid on the right.
//
const UTessellatedSolid &UTessellatedSolid::operator+=
  (const UTessellatedSolid &right)
{
  for (int i=0; i<right.GetNumberOfFacets(); ++i)
    AddFacet(right.GetFacet(i)->GetClone());

  return *this;
}

///////////////////////////////////////////////////////////////////////////////
//
// GetFacet
//
// Access pointer to facet in solid, indexed by integer i.
//
UFacet *UTessellatedSolid::GetFacet (int i) const
{
  return facets[i];
}

///////////////////////////////////////////////////////////////////////////////
//
// GetNumberOfFacets
//
int UTessellatedSolid::GetNumberOfFacets () const
{
  return facets.size();
}

///////////////////////////////////////////////////////////////////////////////
//
// VUSolid::EnumInside UTessellatedSolid::Inside (const UVector3 &p) const
//
// This method must return:
//    * kOutside if the point at offset p is outside the shape
//      boundaries plus fgTolerance/2,
//    * kSurface if the point is <= fgTolerance/2 from a surface, or
//    * kInside otherwise.
//


//______________________________________________________________________________
VUSolid::EnumInside UTessellatedSolid::InsideWithExclusion(const UVector3 &p, UBits *exclusion) const
{
//
// First the simple test - check if we're outside of the X-Y-Z extremes
// of the tessellated solid.
//
  if ( p.x < minExtent.x - fgTolerance ||
       p.x > maxExtent.x + fgTolerance ||
       p.y < minExtent.y - fgTolerance ||
       p.y > maxExtent.y + fgTolerance ||
       p.z < minExtent.z - fgTolerance ||
       p.z > maxExtent.z + fgTolerance )
  {
    return eOutside;
  }  

    vector<int> candidates;
    int limit = voxels.GetCandidatesVoxelArray(p, candidates, NULL);
	if (limit == 0 && insides.GetNbits())
    {
		int index = voxels.GetPointIndex(p);
		EnumInside location = insides[index] ? eInside : eOutside;
        return location;
    }

    double minDist = UUtils::kInfinity;

    for(int i = 0 ; i < limit ; ++i)
    {
        int candidate = candidates[i];
        UFacet &facet = *facets[candidate];
        double dist = facet.Distance(p,minDist);
        if (dist < minDist) minDist = dist;
        if (dist <= 0.5*fgTolerance)
            return eSurface;
    }
//
//
// The following is something of an adaptation of the method implemented by
// Rickard Holmberg augmented with information from Schneider & Eberly,
// "Geometric Tools for Computer Graphics," pp700-701, 2003.  In essence, we're
// trying to determine whether we're inside the volume by projecting a few rays
// and determining if the first surface crossed is has a normal vector between
// 0 to pi/2 (out-going) or pi/2 to pi (in-going).  We should also avoid rays
// which are nearly within the plane of the tessellated surface, and therefore
// produce rays randomly.  For the moment, this is a bit over-engineered
// (belt-braces-and-ducttape).
//
#if USPECSDEBUG
  int nTry                = 7;
#else
  int nTry                = 3;
#endif
  double distOut          = UUtils::kInfinity;
  double distIn           = UUtils::kInfinity;
  double distO            = 0.0;
  double distI            = 0.0;
  double distFromSurfaceO = 0.0;
  double distFromSurfaceI = 0.0;
  UVector3 normalO, normalI;
  bool crossingO          = false;
  bool crossingI          = false;
  VUSolid::EnumInside location          = VUSolid::eOutside;
  VUSolid::EnumInside locationprime     = VUSolid::eOutside;
  int m                   = 0;

  int count = 0;

//  for (int i=0; i<nTry; i++)
//  {
    bool nearParallel = false;
    do
    {
//
//
// We loop until we find direction where the vector is not nearly parallel
// to the surface of any facet since this causes ambiguities.  The usual
// case is that the angles should be sufficiently different, but there are 20
// random directions to select from - hopefully sufficient.
//
      distOut          =  distIn           = UUtils::kInfinity;
      UVector3 v  = randir[m];
      m++;

	  // This code could be voxelized by same algorithm, which is used for DistanceToOut.
	  // we will traverse through voxels. we will call intersect only for those, which would be candidates
	  // and was not checked before.

//	double minDistance = UUtils::kInfinity;
	UVector3 currentPoint = p;
	UVector3 direction = v.Unit();
	UBits exclusion(voxels.GetBitsPerSlice());
	vector<int> candidates, curVoxel(3);
	voxels.GetVoxel(curVoxel, currentPoint);
	double shiftBonus = VUSolid::Tolerance()/10;

	UBits bits(voxels.GetBitsPerSlice());

	do
	{
		if (!voxels.empty.GetNbits() || !voxels.empty[voxels.GetVoxelsIndex(curVoxel)])
		{					
			if (voxels.GetCandidatesVoxelArray(curVoxel, candidates, &exclusion))
			{
				int candidatesCount = candidates.size();
				for (int i = 0 ; i < candidatesCount; ++i)
				{
					int candidate = candidates[i];
					bits.SetBitNumber(candidate);
					UFacet &facet = *facets[candidate];

					crossingO = facet.Intersect(p,v,true,distO,distFromSurfaceO,normalO);
					crossingI = facet.Intersect(p,v,false,distI,distFromSurfaceI,normalI);

					if (crossingO || crossingI)
					{
						double dot = std::fabs(normalO.Dot(v));
					  nearParallel = (crossingO && std::fabs(normalO.Dot(v))<dirTolerance) ||
									 (crossingI && std::fabs(normalI.Dot(v))<dirTolerance);
					  if (!nearParallel)
					  {
						if (crossingO && distO > 0.0 && distO < distOut) 
							distOut = distO;
						if (crossingI && distI > 0.0 && distI < distIn)  
							distIn  = distI;
					  }
					  else break;
					}
				}
				if (nearParallel) break;
			}
		}

		double shift = voxels.DistanceToNext(currentPoint, direction, curVoxel);
		if (shift == UUtils::kInfinity) break;

		currentPoint += direction * (shift + shiftBonus);
	}
	while (voxels.UpdateCurrentVoxel(currentPoint, direction, curVoxel));

	}
	while (nearParallel && m!=maxTries);

	//
//
// Here we loop through the facets to find out if there is an intersection
// between the ray and that facet.  The test if performed separately whether
// the ray is entering the facet or exiting.
//

#ifdef UVERBOSE
    if (m == maxTries)
    {
//
//
// We've run out of random vector directions.  If nTries is set sufficiently
// low (nTries <= 0.5*maxTries) then this would indicate that there is
// something wrong with geometry.
//
      std::ostringstream message;
      int oldprc = message.precision(16);
      message << "Cannot determine whether point is inside or outside volume!"
              << endl
              << "Solid name       = " << GetName()  << endl
              << "Geometry Type    = " << geometryType  << endl
              << "Number of facets = " << facets.size() << endl
              << "Position:"  << endl << endl
              << "p.x() = "   << p.x()/mm << " mm" << endl
              << "p.y() = "   << p.y()/mm << " mm" << endl
              << "p.z() = "   << p.z()/mm << " mm";
      message.precision(oldprc);
      UException("UTessellatedSolid::Inside()",
                  "GeomSolids1002", JustWarning, message);
    }
#endif
//
//
// In the next if-then-elseif string the logic is as follows:
// (1) You don't hit anything so cannot be inside volume, provided volume
//     constructed correctly!
// (2) Distance to inside (ie. nearest facet such that you enter facet) is
//     shorter than distance to outside (nearest facet such that you exit
//     facet) - on condition of safety distance - therefore we're outside.
// (3) Distance to outside is shorter than distance to inside therefore we're
//     inside.
//
    if (distIn == UUtils::kInfinity && distOut == UUtils::kInfinity)
      location = eOutside;
    else if (distIn <= distOut - fgTolerance*0.5)
      location = eOutside;
    else if (distOut <= distIn - fgTolerance*0.5)
      location = eInside;
 // }

  return location;
}

VUSolid::EnumInside UTessellatedSolid::Inside (const UVector3 &aPoint) const
{
#ifdef DEBUG
	VUSolid::EnumInside insideDummy = InsideDummy(aPoint);
#endif

	VUSolid::EnumInside location = InsideWithExclusion(aPoint, NULL);

#ifdef DEBUG
	if (location != insideDummy)
		location = insideDummy; // you can place a breakpoint here
#endif
	return location;
}

VUSolid::EnumInside UTessellatedSolid::InsideDummy (const UVector3 &p) const
{
//
// First the simple test - check if we're outside of the X-Y-Z extremes
// of the tessellated solid.
//
  if ( p.x < minExtent.x - fgTolerance ||
       p.x > maxExtent.x + fgTolerance ||
       p.y < minExtent.y - fgTolerance ||
       p.y > maxExtent.y + fgTolerance ||
       p.z < minExtent.z - fgTolerance ||
       p.z > maxExtent.z + fgTolerance )
  {
    return eOutside;
  }  

  double minDist = UUtils::kInfinity;
//
//
// Check if we are close to a surface
//
  for (UFacetCI f=facets.begin(); f!=facets.end(); f++)
  {
	UFacet &facet = *(*f);
    double dist = facet.Distance(p,minDist);
    if (dist < minDist) minDist = dist;
    if (dist <= 0.5*fgTolerance)
    {
      return eSurface;
    }
  }
//
//
// The following is something of an adaptation of the method implemented by
// Rickard Holmberg augmented with information from Schneider & Eberly,
// "Geometric Tools for Computer Graphics," pp700-701, 2003.  In essence, we're
// trying to determine whether we're inside the volume by projecting a few rays
// and determining if the first surface crossed is has a normal vector between
// 0 to pi/2 (out-going) or pi/2 to pi (in-going).  We should also avoid rays
// which are nearly within the plane of the tessellated surface, and therefore
// produce rays randomly.  For the moment, this is a bit over-engineered
// (belt-braces-and-ducttape).
//
#if USPECSDEBUG
  int nTry                = 7;
#else
  int nTry                = 3;
#endif
  double distOut          = UUtils::kInfinity;
  double distIn           = UUtils::kInfinity;
  double distO            = 0.0;
  double distI            = 0.0;
  double distFromSurfaceO = 0.0;
  double distFromSurfaceI = 0.0;
  UVector3 normalO(0.0,0.0,0.0);
  UVector3 normalI(0.0,0.0,0.0);
  bool crossingO          = false;
  bool crossingI          = false;
  VUSolid::EnumInside location          = VUSolid::eOutside;
  VUSolid::EnumInside locationprime     = VUSolid::eOutside;
  int m = 0;

  for (int i=0; i<nTry; i++)
  {
    bool nearParallel = false;
    do
    {
//
//
// We loop until we find direction where the vector is not nearly parallel
// to the surface of any facet since this causes ambiguities.  The usual
// case is that the angles should be sufficiently different, but there are 20
// random directions to select from - hopefully sufficient.
//
      distOut          =  distIn           = UUtils::kInfinity;
      UVector3 v  = randir[m];
      m++;
      UFacetCI f = facets.begin();

      do
      {
//
//
// Here we loop through the facets to find out if there is an intersection
// between the ray and that facet.  The test if performed separately whether
// the ray is entering the facet or exiting.
//
        crossingO =  ((*f)->Intersect(p,v,true,distO,distFromSurfaceO,normalO));
        crossingI =  ((*f)->Intersect(p,v,false,distI,distFromSurfaceI,normalI));
        if (crossingO || crossingI)
        {
          nearParallel = (crossingO && std::fabs(normalO.Dot(v))<dirTolerance) ||
                         (crossingI && std::fabs(normalI.Dot(v))<dirTolerance);
          if (!nearParallel)
          {
            if (crossingO && distO > 0.0 && distO < distOut) distOut = distO;
            if (crossingI && distI > 0.0 && distI < distIn)  distIn  = distI;
          }
        }
      } while (!nearParallel && ++f!=facets.end());
    } while (nearParallel && m!=maxTries);

#ifdef UVERBOSE
    if (m == maxTries)
    {
//
//
// We've run out of random vector directions.  If nTries is set sufficiently
// low (nTries <= 0.5*maxTries) then this would indicate that there is
// something wrong with geometry.
//
      std::ostringstream message;
      int oldprc = message.precision(16);
      message << "Cannot determine whether point is inside or outside volume!"
              << endl
              << "Solid name       = " << GetName()  << endl
              << "Geometry Type    = " << geometryType  << endl
              << "Number of facets = " << facets.size() << endl
              << "Position:"  << endl << endl
              << "p.x() = "   << p.x()/mm << " mm" << endl
              << "p.y() = "   << p.y()/mm << " mm" << endl
              << "p.z() = "   << p.z()/mm << " mm";
      message.precision(oldprc);
      UException("UTessellatedSolid::Inside()",
                  "GeomSolids1002", JustWarning, message);
    }
#endif
//
//
// In the next if-then-elseif string the logic is as follows:
// (1) You don't hit anything so cannot be inside volume, provided volume
//     constructed correctly!
// (2) Distance to inside (ie. nearest facet such that you enter facet) is
//     shorter than distance to outside (nearest facet such that you exit
//     facet) - on condition of safety distance - therefore we're outside.
// (3) Distance to outside is shorter than distance to inside therefore we're
//     inside.
//
    if (distIn == UUtils::kInfinity && distOut == UUtils::kInfinity)
      locationprime = eOutside;
    else if (distIn <= distOut - fgTolerance*0.5)
      locationprime = eOutside;
    else if (distOut <= distIn - fgTolerance*0.5)
      locationprime = eInside;

    if (i == 0)  { location = locationprime; }
  }

  return location;
}

///////////////////////////////////////////////////////////////////////////////
//
// UVector3 UTessellatedSolid::SurfaceNormal (const UVector3 &p) const
//
// Return the outwards pointing unit normal of the shape for the
// surface closest to the point at offset p.

bool UTessellatedSolid::Normal (const UVector3 &p,  UVector3 &aNormal) const
{
  UFacetCI minFacet;
  double minDist   = UUtils::kInfinity;
  double dist      = 0.0;
  UVector3 normal;
  
  for (UFacetCI f=facets.begin(); f!=facets.end(); ++f)
  {
	  UFacet &facet = *(*f);
    dist = facet.Distance(p,minDist);
    if (dist < minDist)
    {
      minDist  = dist;
      minFacet = f;
    }
  }
  
  if (minDist != UUtils::kInfinity)
  {
	  UFacet &mf = *(*minFacet);
     normal = mf.GetSurfaceNormal();
  }
  else
  {
#ifdef UVERBOSE
    std::ostringstream message;
    message << "Point p is not on surface !?" << endl
            << "          No facets found for point: " << p << " !" << endl
            << "          Returning approximated value for normal.";
    UException("UTessellatedSolid::SurfaceNormal(p)", "GeomSolids1002",
                JustWarning, message );
#endif
    normal = (p.z > 0 ? UVector3(0,0,1) : UVector3(0,0,-1));
  }

  aNormal = normal;
  return true;
}

///////////////////////////////////////////////////////////////////////////////
//
// double DistanceToIn(const UVector3& p, const UVector3& v)
//
// Return the distance along the normalised vector v to the shape,
// from the point at offset p. If there is no intersection, return
// UUtils::kInfinity. The first intersection resulting from ‘leaving’ a
// surface/volume is discarded. Hence, this is tolerant of points on
// surface of shape.

double UTessellatedSolid::DistanceToInDummy (const UVector3 &p,
  const UVector3 &v, double aPstep) const
{
  double minDist         = UUtils::kInfinity;
  double dist            = 0.0;
  double distFromSurface = 0.0;
  UVector3 normal;
  
#if USPECSDEBUG
  if ( Inside(p) == kInside )
  {
     std::ostringstream message;
     int oldprc = message.precision(16) ;
     message << "Point p is already inside!?" << endl
             << "Position:"  << endl << endl
             << "   p.x() = "   << p.x()/mm << " mm" << endl
             << "   p.y() = "   << p.y()/mm << " mm" << endl
             << "   p.z() = "   << p.z()/mm << " mm" << endl
             << "DistanceToOut(p) == " << DistanceToOut(p);
     message.precision(oldprc) ;
     UException("UTriangularFacet::DistanceToIn(p,v)", "GeomSolids1002",
                 JustWarning, message);
  }
#endif

  for (UFacetCI f=facets.begin(); f!=facets.end(); f++)
  {
    if ((*f)->Intersect(p,v,false,dist,distFromSurface,normal))
    {
//
//
// Set minDist to the new distance to current facet if distFromSurface is in
// positive direction and point is not at surface.  If the point is within
// 0.5*fgTolerance of the surface, then force distance to be zero and
// leave member function immediately (for efficiency), as proposed by & credit
// to Akira Okumura.
//
      if (distFromSurface > 0.5*fgTolerance && dist >= 0.0 && dist < minDist)
      {
        minDist  = dist;
      }
      else if (-0.5*fgTolerance <= dist && dist <= 0.5*fgTolerance)
      {
        return 0.0;
      }
    }
  }

  return minDist;
}


double UTessellatedSolid::DistanceToOutDummy (const UVector3 &p,
                    const UVector3 &v, 
					
                                UVector3       &aNormalVector,
                                bool           &aConvex,
                                double aPstep

//					const bool calcNorm, bool *validNorm, UVector3 *n
						  ) const
{
  double minDist         = UUtils::kInfinity;
  double dist            = 0.0;
  double distFromSurface = 0.0;
  UVector3 normal, minNormal;
  
#if USPECSDEBUG
  if ( Inside(p) == kOutside )
  {
     std::ostringstream message;
     int oldprc = message.precision(16) ;
     message << "Point p is already outside!?" << endl
             << "Position:"  << endl << endl
             << "   p.x() = "   << p.x()/mm << " mm" << endl
             << "   p.y() = "   << p.y()/mm << " mm" << endl
             << "   p.z() = "   << p.z()/mm << " mm" << endl
             << "DistanceToIn(p) == " << DistanceToIn(p);
     message.precision(oldprc) ;
     UException("UTriangularFacet::DistanceToOut(p)", "GeomSolids1002",
                 JustWarning, message);
  }
#endif

  bool isExtreme = false;
  for (UFacetCI f=facets.begin(); f!=facets.end(); f++)
  {
    if ((*f)->Intersect(p,v,true,dist,distFromSurface,normal))
     {
      if (distFromSurface > 0.0 && distFromSurface <= 0.5*fgTolerance &&
          (*f)->Distance(p,fgTolerance) <= 0.5*fgTolerance)
      {
        // We are on a surface. Return zero.
		  /*
        if (calcNorm) {
          *validNorm = extremeFacets.count(*f) ? true : false;
          *n         = SurfaceNormal(p);
        }  
		  */
        return 0.0;
      }
      if (dist >= 0.0 && dist < minDist)
      {
        minDist   = dist;
        minNormal = normal;
        isExtreme = extremeFacets.count(*f) ? true : false;
      }
    }
  }
  
  if (minDist < UUtils::kInfinity)
  {
	  /*
    if (calcNorm)
    {
      *validNorm = isExtreme;
      *n         = minNormal;
    }
	*/
    return minDist;
  }
  else
  {
    // No intersection found
	  /*
    if (calcNorm)
    {
      *validNorm = false;
      *n         = SurfaceNormal(p);
    }
	*/
    return 0.0;
  }
}








void UTessellatedSolid::DistanceToOutCandidates(const UVector3 &aPoint,
                    const UVector3 &direction, double &minDist, UVector3 &minNormal, bool &aConvex, double aPstep, vector<int > &candidates, UBits &bits) const
{
	int candidatesCount = candidates.size();
	double dist            = 0.0;
	double distFromSurface = 0.0;
	UVector3 normal;

	double minDistance = UUtils::kInfinity;   
	for (int i = 0 ; i < candidatesCount; ++i)
	{
		int candidate = candidates[i];
		bits.SetBitNumber(candidate);
		UFacet &facet = *facets[candidate];
		if (facet.Intersect(aPoint,direction,true,dist,distFromSurface,normal))
		{
		  if (distFromSurface > 0.0 && distFromSurface <= 0.5*fgTolerance &&
			  facet.Distance(aPoint,fgTolerance) <= 0.5*fgTolerance)
		  {
			// We are on a surface. Return zero.
			minDist = 0.0;
			return;
		  }
		  if (dist >= 0.0 && dist < minDist)
		  {
			minDist = dist;
			minNormal = normal;
		  }
		}
	}
}

double UTessellatedSolid::DistanceToOut(const UVector3 &aPoint, const UVector3 &aDirection,
	UVector3       &aNormalVector, bool           &aConvex,
	double aPstep) const
{
	double minDistance = UUtils::kInfinity;
	UVector3 currentPoint = aPoint;
	UVector3 direction = aDirection.Unit();
	double totalShift = 0;
	UBits exclusion(voxels.GetBitsPerSlice());
	vector<int> candidates, curVoxel(3);
	if (!voxels.Contains(aPoint)) return 0;

	voxels.GetVoxel(curVoxel, currentPoint);

    UVector3 minNormal;
	double shiftBonus = VUSolid::Tolerance()/10;

	do
	{
		if (!voxels.empty.GetNbits() || !voxels.empty[voxels.GetVoxelsIndex(curVoxel)])
		{
			if (voxels.GetCandidatesVoxelArray(curVoxel, candidates, &exclusion))
			{
				DistanceToOutCandidates(aPoint, direction, minDistance, minNormal,  aConvex, aPstep, candidates, exclusion); 
				if (minDistance < totalShift) break; 
			}
		}

		double shift = voxels.DistanceToNext(currentPoint, direction, curVoxel);
		if (shift == UUtils::kInfinity) break;

		totalShift += shift;
		if (minDistance < totalShift) break;

		currentPoint += direction * (shift + shiftBonus);
	}
	while (voxels.UpdateCurrentVoxel(currentPoint, direction, curVoxel));

	if (minDistance == UUtils::kInfinity) minDistance = 0;

	return minDistance;
}













double UTessellatedSolid::DistanceToInCandidates(const UVector3 &aPoint, const UVector3 &direction, double aPstep, std::vector<int> &candidates, UBits &bits) const
{
	int candidatesCount = candidates.size();
	double dist            = 0.0;
	double distFromSurface = 0.0;
	UVector3 normal;

	double minDistance = UUtils::kInfinity;   
	for (int i = 0 ; i < candidatesCount; ++i)
	{
		int candidate = candidates[i];
		bits.SetBitNumber(candidate);
		UFacet &facet = *facets[candidate];
		if (facet.Intersect(aPoint,direction,false,dist,distFromSurface,normal))
		{
			//
			// Set minDist to the new distance to current facet if distFromSurface is in
			// positive direction and point is not at surface.  If the point is within
			// 0.5*fgTolerance of the surface, then force distance to be zero and
			// leave member function immediately (for efficiency), as proposed by & credit
			// to Akira Okumura.
			//
		  if (distFromSurface > 0.5*fgTolerance && dist >= 0.0 && dist < minDistance)
		  {
			minDistance  = dist;
		  }
		  else if (-0.5*fgTolerance <= dist && dist <= 0.5*fgTolerance)
		  {
			return 0.0;
		  }
		}
	}
	return minDistance;
}


double UTessellatedSolid::DistanceToIn(const UVector3 &aPoint, 
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

	voxels.GetVoxel(curVoxel, currentPoint);

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

///////////////////////////////////////////////////////////////////////////////
//
// double DistanceToIn(const UVector3& p)
//
// Calculate distance to nearest surface of shape from an outside point p. The
// distance can be an underestimate.


struct VoxelBoxInfo
{
	double distance;
	int id;
};

bool compare( const VoxelBoxInfo &l, const VoxelBoxInfo &r)
{
	return l.distance < r.distance;
}

double UTessellatedSolid::SafetyFromOutside (const UVector3 &p, bool) const
{
  double minDist = UUtils::kInfinity;
  double dist    = 0.0;
  
#if USPECSDEBUG
  if ( Inside(p) == kInside )
  {
     std::ostringstream message;
     int oldprc = message.precision(16) ;
     message << "Point p is already inside!?" << endl
             << "Position:"  << endl << endl
             << "p.x() = "   << p.x()/mm << " mm" << endl
             << "p.y() = "   << p.y()/mm << " mm" << endl
             << "p.z() = "   << p.z()/mm << " mm" << endl
             << "DistanceToOut(p) == " << DistanceToOut(p);
     message.precision(oldprc) ;
     UException("UTriangularFacet::DistanceToIn(p)", "GeomSolids1002",
                 JustWarning, message);
  }
#endif

  if ( p.x < minExtent.x - fgTolerance ||
       p.x > maxExtent.x + fgTolerance ||
       p.y < minExtent.y - fgTolerance ||
       p.y > maxExtent.y + fgTolerance ||
       p.z < minExtent.z - fgTolerance ||
       p.z > maxExtent.z + fgTolerance )
  {

  }
  else
  {
    vector<int> candidates;
    int limit = voxels.GetCandidatesVoxelArray(p, candidates, NULL);
	if (limit == 0 && insides.GetNbits())
    {
		int index = voxels.GetPointIndex(p);
		if (insides[index]) return 0.0;
    }
  }

  int size = voxelBoxes.size();
  vector<VoxelBoxInfo> voxelsSorted(size);

  UBox safetyBox;

  for (int i = 0; i < size; i++)
  {
	  const UVoxelBox &voxelBox = voxelBoxes[i];

	  UVector3 pointShifted = p - voxelBox.pos;
	  const UVector3 &vec = voxelBox.hlen;
	  safetyBox.Set(vec);
	  double safety = safetyBox.SafetyFromOutside(pointShifted, true);
	  VoxelBoxInfo info;
	  info.distance = safety;
	  info.id = i;
	  voxelsSorted[i] = info;
  }
  std::sort(voxelsSorted.begin(), voxelsSorted.end(), compare);

  for (int i = 0; i < size; i++)
  {
	  const VoxelBoxInfo &info = voxelsSorted[i];
	  const UVoxelBox &voxelBox = voxelBoxes[info.id];
	  double dist = info.distance;
	  if (dist > minDist) break;

	  const vector<int> &candidates = voxelBoxesFaces[info.id];
	  int csize = candidates.size();
	  for (int i = 0; i < csize; i++)
      {
		  int candidate = candidates[i];
		 UFacet &facet = *facets[candidate];
         dist = facet.Distance(p,minDist,false);
         if (dist < minDist) minDist  = dist;
      }
  }
  return minDist;

  for (UFacetCI f=facets.begin(); f!=facets.end(); ++f)
  {
	  UFacet &facet = *(*f);
    dist = facet.Distance(p,minDist,false);
    if (dist < minDist)  { minDist  = dist; }
  }
  
  return minDist;

}

///////////////////////////////////////////////////////////////////////////////
//
// double DistanceToOut(const UVector3& p, const UVector3& v,
//                        const bool calcNorm=false,
//                        bool *validNorm=0, UVector3 *n=0);
//
// Return distance along the normalised vector v to the shape, from a
// point at an offset p inside or on the surface of the
// shape. Intersections with surfaces, when the point is not greater
// than fgTolerance/2 from a surface, must be ignored.
//     If calcNorm is true, then it must also set validNorm to either
//     * true, if the solid lies entirely behind or on the exiting
//        surface. Then it must set n to the outwards normal vector
//        (the Magnitude of the vector is not defined).
//     * false, if the solid does not lie entirely behind or on the
//       exiting surface.
// If calcNorm is false, then validNorm and n are unused.


/*
double UTessellatedSolid::DistanceToOut(const UVector3 &aPoint, const UVector3 &aDirection,
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
*/


///////////////////////////////////////////////////////////////////////////////
//
// double DistanceToOut(const UVector3& p)
//
// Calculate distance to nearest surface of shape from an inside
// point. The distance can be an underestimate.

double UTessellatedSolid::SafetyFromInside (const UVector3 &p, bool) const
{
  double minDist = UUtils::kInfinity;
  double dist    = 0.0;
  
#if USPECSDEBUG
  if ( Inside(p) == kOutside )
  {
     std::ostringstream message;
     int oldprc = message.precision(16) ;
     message << "Point p is already outside!?" << endl
             << "Position:"  << endl << endl
             << "p.x() = "   << p.x()/mm << " mm" << endl
             << "p.y() = "   << p.y()/mm << " mm" << endl
             << "p.z() = "   << p.z()/mm << " mm" << endl
             << "DistanceToIn(p) == " << DistanceToIn(p);
     message.precision(oldprc) ;
     UException("UTriangularFacet::DistanceToOut(p)", "GeomSolids1002",
                 JustWarning, message);
  }
#endif

  if ( p.x < minExtent.x - fgTolerance ||
       p.x > maxExtent.x + fgTolerance ||
       p.y < minExtent.y - fgTolerance ||
       p.y > maxExtent.y + fgTolerance ||
       p.z < minExtent.z - fgTolerance ||
       p.z > maxExtent.z + fgTolerance )
  {
    return 0.0;
  }  

  for (UFacetCI f=facets.begin(); f!=facets.end(); f++)
  {
	  UFacet &facet = *(*f);
    dist = facet.Distance(p,minDist,true);
    if (dist < minDist) minDist  = dist;
  }

  // the algorithm is based on create shell on current voxel ... than testing all surrounding voxels
  
  return minDist;
}

///////////////////////////////////////////////////////////////////////////////
//
// UGeometryType GetEntityType() const;
//
// Provide identification of the class of an object (required for
// persistency and STEP interface).
//
UGeometryType UTessellatedSolid::GetEntityType () const
{
  return geometryType;
}


/*
///////////////////////////////////////////////////////////////////////////////
//
void UTessellatedSolid::DescribeYourselfTo (UVGraphicsScene& scene) const
{
  scene.AddSolid (*this);
}
*/

///////////////////////////////////////////////////////////////////////////////
//
// Dispatch to parameterisation for replication mechanism dimension
// computation & modification.
//                                                                                
//void UTessellatedSolid::ComputeDimensions (UVPVParameterisation* p,
//  const int n, const UVPhysicalVolume* pRep) const
//{
//  UVSolid *ptr = 0;
//  ptr           = *this;
//  p->ComputeDimensions(ptr,n,pRep);
//}

///////////////////////////////////////////////////////////////////////////////
//
std::ostream &UTessellatedSolid::StreamInfo(std::ostream &os) const
{
  os << endl;
  os << "Geometry Type    = " << geometryType  << endl;
  os << "Number of facets = " << facets.size() << endl;
  
  for (UFacetCI f = facets.begin(); f != facets.end(); f++)
  {
    os << "FACET #          = " << f - facets.begin()+1 << endl;
	UFacet &facet = *(*f);
    facet.StreamInfo(os);
  }
  os <<endl;
  
  return os;
}

/*
///////////////////////////////////////////////////////////////////////////////
//
UPolyhedron *UTessellatedSolid::CreatePolyhedron () const
{
  int nVertices = vertexList.size();
  int nFacets   = facets.size();
  UPolyhedronArbitrary *polyhedron = new UPolyhedronArbitrary (nVertices, nFacets);
  for (UVector3List::const_iterator v = vertexList.begin();
        v!=vertexList.end(); v++) polyhedron->AddVertex(*v);
    
  for (UFacetCI f=facets.begin(); f != facets.end(); f++)
  {
    int v[4];
    for (int j=0; j<4; j++)
    {
      int i = (*f)->GetVertexIndex(j);
      if (i == 999999999) v[j] = 0;
      else                v[j] = i+1;
    }
    if ((*f)->GetEntityType() == "URectangularFacet")
    {
      int i = v[3];
      v[3]     = v[2];
      v[2]     = i;
    }
    polyhedron->AddFacet(v[0],v[1],v[2],v[3]);
  }
  polyhedron->SetReferences();  
 
  return (UPolyhedron*) polyhedron;
}

///////////////////////////////////////////////////////////////////////////////
//
UNURBS *UTessellatedSolid::CreateNURBS () const
{
  return 0;
}

///////////////////////////////////////////////////////////////////////////////
//
// GetPolyhedron
//
UPolyhedron* UTessellatedSolid::GetPolyhedron () const
{
  if (!fpPolyhedron ||
      fpPolyhedron->GetNumberOfRotationStepsAtTimeOfCreation() !=
      fpPolyhedron->GetNumberOfRotationSteps())
    {
      delete fpPolyhedron;
      fpPolyhedron = CreatePolyhedron();
    }
  return fpPolyhedron;
}
*/

// NOTE: USolid uses method with different arguments
//
///////////////////////////////////////////////////////////////////////////////
//
// CalculateExtent
//
// Based on correction provided by Stan Seibert, University of Texas.
//

/*
bool
UTessellatedSolid::CalculateExtent(const EAxis pAxis,
                                    const UVoxelLimits& pVoxelLimit,
                                    const UAffineTransform& pTransform,
                                          double& pMin, double& pMax) const
{
    UVector3List transVertexList(vertexList);

    // Put solid into transformed frame
    for (int i=0; i<vertexList.size(); i++)
      { pTransform.ApplyPointTransform(transVertexList[i]); }

    // Find min and max extent in each dimension
    UVector3 minExtent(UUtils::kInfinity, UUtils::kInfinity, UUtils::kInfinity);
    UVector3 maxExtent(-UUtils::kInfinity, -UUtils::kInfinity, -UUtils::kInfinity);
    for (int i=0; i<transVertexList.size(); i++)
    {
      for (int axis=UVector3::X; axis < UVector3::SIZE; axis++)
      {
        double coordinate = transVertexList[i][axis];
        if (coordinate < minExtent[axis])
          { minExtent[axis] = coordinate; }
        if (coordinate > maxExtent[axis])
          { maxExtent[axis] = coordinate; }
      }
    }

    // Check for containment and clamp to voxel boundaries
    for (int axis=UVector3::X; axis < UVector3::SIZE; axis++)
    {
      EAxis geomAxis = kXAxis; // U geom classes use different index type
      switch(axis)
      {
        case UVector3::X: geomAxis = kXAxis; break;
        case UVector3::Y: geomAxis = kYAxis; break;
        case UVector3::Z: geomAxis = kZAxis; break;
      }
      bool isLimited = pVoxelLimit.IsLimited(geomAxis);
      double voxelMinExtent = pVoxelLimit.GetMinExtent(geomAxis);
      double voxelMaxExtent = pVoxelLimit.GetMaxExtent(geomAxis);

      if (isLimited)
      {
        if ( minExtent[axis] > voxelMaxExtent+fgTolerance ||
             maxExtent[axis] < voxelMinExtent-fgTolerance    )
        {
          return false ;
        }
        else
        {
          if (minExtent[axis] < voxelMinExtent)
          {
            minExtent[axis] = voxelMinExtent ;
          }
          if (maxExtent[axis] > voxelMaxExtent)
          {
            maxExtent[axis] = voxelMaxExtent;
          }
        }
      }
    }

    // Convert pAxis into UVector3 index
    int vecAxis=0;
    switch(pAxis)
    {
      case kXAxis: vecAxis = UVector3::X; break;
      case kYAxis: vecAxis = UVector3::Y; break;
      case kZAxis: vecAxis = UVector3::Z; break;
      default: break;
    } 

    pMin = minExtent[vecAxis] - fgTolerance;
    pMax = maxExtent[vecAxis] + fgTolerance;

    return true;
}
*/

/*
void Extent(EAxisType aAxis, double &aMin, double &aMax)
{

}
  // Returns the minimum and maximum extent along the specified Cartesian axis
  virtual void Extent( UVector3 &aMin, UVector3 &aMax ) const = 0;
  */

/*
///////////////////////////////////////////////////////////////////////////////
//
double UTessellatedSolid::GetMinXExtent () const
{
	return minExtent.x;
}

///////////////////////////////////////////////////////////////////////////////
//
double UTessellatedSolid::GetMaxXExtent () const
{
	return maxExtent.x;
}

///////////////////////////////////////////////////////////////////////////////
//
double UTessellatedSolid::GetMinYExtent () const
  {return minExtent.y;}

///////////////////////////////////////////////////////////////////////////////
//
double UTessellatedSolid::GetMaxYExtent () const
  {return maxExtent.y;}

///////////////////////////////////////////////////////////////////////////////
//
double UTessellatedSolid::GetMinZExtent () const
  {return minExtent.z;}

///////////////////////////////////////////////////////////////////////////////
//
double UTessellatedSolid::GetMaxZExtent () const
  {return maxExtent.z;}
*/

void UTessellatedSolid::Extent (EAxisType aAxis, double &aMin, double &aMax) const
{
	// Returns extent of the solid along a given cartesian axis
	if (aAxis >= 0 && aAxis <= 2)
	{
		aMin = minExtent[aAxis]; aMax = maxExtent[aAxis];
	}
	else
         std::cout << "Extent: unknown axis" << aAxis << std::endl;
}

void UTessellatedSolid::Extent (UVector3 &aMin, UVector3 &aMax) const
{
   aMin = minExtent;
   aMax = maxExtent;
}

///////////////////////////////////////////////////////////////////////////////
//
/*
UVisExtent UTessellatedSolid::GetExtent () const
{
  return UVisExtent (minExtent.x, maxExtent.x, minExtent.y, maxExtent.y,
    minExtent.z, maxExtent.z);
}
*/

/*
///////////////////////////////////////////////////////////////////////////////
//
double UTessellatedSolid::GetCubicVolume ()
{
  if(cubicVolume != 0.) {;}
  else   { cubicVolume = VUSolid::GetCubicVolume(); }
  return cubicVolume;
}
*/

///////////////////////////////////////////////////////////////////////////////
//
double UTessellatedSolid::GetSurfaceArea ()
{
  if (surfaceArea != 0.) return surfaceArea;

  for (UFacetI f=facets.begin(); f!=facets.end(); f++)
  {
    surfaceArea += (*f)->GetArea();
  }
  return surfaceArea;
}

///////////////////////////////////////////////////////////////////////////////
//
UVector3 UTessellatedSolid::GetPointOnSurface() const
{
  // Select randomly a facet and return a random point on it

	int i = (int) UUtils::RandomUniform (0, facets.size());
  return facets[i]->GetPointOnFace();
}
///////////////////////////////////////////////////////////////////////////////
//
// SetRandomVectorSet
//
// This is a set of predefined random vectors (if that isn't a contradition
// in terms!) used to generate rays from a user-defined point.  The member
// function Inside uses these to determine whether the point is inside or
// outside of the tessellated solid.  All vectors should be unit vectors.
//
void UTessellatedSolid::SetRandomVectorSet()
{
	randir.resize(20);
	randir[0] = UVector3(-0.9577428892113370, 0.2732676269591740, 0.0897405271949221);
  randir[1]  = UVector3(-0.8331264504940770,-0.5162067214954600,-0.1985722492445700);
  randir[2]  = UVector3(-0.1516671651108820, 0.9666292616127460, 0.2064580868390110);
  randir[3]  = UVector3( 0.6570250350323190,-0.6944539025883300, 0.2933460081893360);
  randir[4]  = UVector3(-0.4820456281280320,-0.6331060000098690,-0.6056474264406270);
  randir[5]  = UVector3( 0.7629032554236800 , 0.1016854697539910,-0.6384658864065180);
  randir[6]  = UVector3( 0.7689540409061150, 0.5034929891988220, 0.3939600142169160);
  randir[7]  = UVector3( 0.5765188359255740, 0.5997271636278330,-0.5549354566343150);
  randir[8]  = UVector3( 0.6660632777862070,-0.6362809868288380, 0.3892379937580790);
  randir[9]  = UVector3( 0.3824415020414780, 0.6541792713761380,-0.6525243125110690);
  randir[10] = UVector3(-0.5107726564526760, 0.6020905056811610, 0.6136760679616570);
  randir[11] = UVector3( 0.7459135439578050, 0.6618796061649330, 0.0743530220183488);
  randir[12] = UVector3( 0.1536405855311580, 0.8117477913978260,-0.5634359711967240);
  randir[13] = UVector3( 0.0744395301705579,-0.8707110101772920,-0.4861286795736560);
  randir[14] = UVector3(-0.1665874645185400, 0.6018553940549240,-0.7810369397872780);
  randir[15] = UVector3( 0.7766902003633100, 0.6014617505959970,-0.1870724331097450);
  randir[16] = UVector3(-0.8710128685847430,-0.1434320216603030,-0.4698551243971010);
  randir[17] = UVector3( 0.8901082092766820,-0.4388411398893870, 0.1229871120030100);
  randir[18] = UVector3(-0.6430417431544370,-0.3295938228697690, 0.6912779675984150);
  randir[19] = UVector3( 0.6331124368380410, 0.6306211461665000, 0.4488714875425340);

  maxTries = 20;
}
