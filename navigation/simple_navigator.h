/*
 * simple_navigator.h
 *
 *  Created on: Mar 12, 2014
 *      Author: swenzel
 */

#ifndef SIMPLE_NAVIGATOR_H_
#define SIMPLE_NAVIGATOR_H_

#include "base/global.h"
#include "volumes/placed_volume.h"
#include "base/vector3d.h"
#include "navigation/navigationstate.h"
#include <iostream>
#include <cassert>
#include "base/soa3d.h"

#ifdef VECGEOM_ROOT
#include "management/root_manager.h"
#include "TGeoNode.h"
#include "TGeoMatrix.h"
#endif

namespace VECGEOM_NAMESPACE
{


class SimpleNavigator
{

public:
	/**
	 * function to locate a global point in the geometry hierarchy
	 * input: pointer to starting placed volume in the hierarchy and a global point, we also give an indication if we call this from top
	 * output: pointer to the deepest volume in hierarchy that contains the particle and the navigation state
	 *
	 * scope: function to be used on both CPU and GPU
	 */
	VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	VPlacedVolume const *
	LocatePoint( VPlacedVolume const * /* volume */,
				 Vector3D<Precision> const & /* globalpoint */,
				 NavigationState & /* state (volume path) to be returned */,
				 bool /*top*/) const;

	/**
	 * function to locate a global point in the geometry hierarchy
	 * input:  A local point in the referenceframe of the current deepest volume in the path,
	 * the path itself which gets modified
	 * output: path which may be modified
	 *
	 * scope: function to be used on both CPU and GPU
	 */
	VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	VPlacedVolume const *
	RelocatePointFromPath( Vector3D<Precision> const & /* localpoint */,
								 NavigationState & /* state to be modified */
	 	 	 	 	 	 ) const;

	/**
	 * function to check whether global point has same path as given by currentstate
	 * input:  A global point
	 *         the path itself
	 *         a new path object which is filled
	 * output: yes or no
	 *
	 * scope: function to be used on both CPU and GPU
	 */
	VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	bool
	HasSamePath(
				Vector3D<Precision> const & /* globalpoint */,
				NavigationState & /* currentstate */,
				NavigationState & /* newstate */
			   ) const;


	/**
	* A function to navigate ( find next boundary and/or the step to do )
	*/
	VECGEOM_CUDA_HEADER_BOTH
	void FindNextBoundaryAndStep( Vector3D<Precision> const & /* global point */,
								  Vector3D<Precision> const & /* global dir */,
								  NavigationState const & /* currentstate */,
								  NavigationState & /* newstate */,
								  Precision const & /* proposed physical step */,
								  Precision & /*step*/
								 ) const;

	/**
	 * A function to get back the safe distance; given a NavigationState object and a current global point
	 * point
	 */
	VECGEOM_CUDA_HEADER_BOTH
	Precision GetSafety( Vector3D<Precision> const & /*global_point*/,
					NavigationState const & /* currentstate */
	) const;

	/**
	 * Navigation interface for baskets; templates on Container3D which might be a SOA3D or AOS3D container
	 * Note that the user has to provide a couple of workspace memories; This is the easiest way to make the navigator fully
	 * threadsafe
	 */
	template <typename Container3D>
	void FindNextBoundaryAndStep(
			Container3D const & /*point*/,
			Container3D const & /*dir*/,
			Container3D & /*localpoint*/,
			Container3D & /*localdir*/,
			NavigationState **  /* this is interpreted as an array of pointers to NavigationStates */,
			NavigationState **  /* this is interpreted as an array of pointers to NavigationStates */,
			Precision const * /* pSteps -- proposed steps */,
			Precision * /* safeties */,
			Precision * /* distances; steps */,
			Precision * /* workspace */,
			int * /* for array of nextnodes */,
			int np) const;


	/**
	 * A verbose function telling about possible hit targets and steps; starting from a navigation state
	 * and a global point and direction ( we need to check for consistency ... ); mainly for debugging purposes
	 */
	void InspectEnvironmentForPointAndDirection(
			Vector3D<Precision> const & /* global point */,
			Vector3D<Precision> const & /* global direction */,
			NavigationState const & /* current state */
	) const;

}; // end of class declaration

VPlacedVolume const *
SimpleNavigator::LocatePoint( VPlacedVolume const * vol, Vector3D<Precision> const & point,
							  NavigationState & path, bool top ) const
{
	VPlacedVolume const * candvolume = vol;
	Vector3D<Precision> tmp(point);
	if( top )
	{
		candvolume = ( vol->UnplacedInside( point ) ) ? vol : 0;
	}
	if( candvolume )
	{
		path.Push( candvolume );
		Vector<Daughter> const * daughters = candvolume->logical_volume()->daughtersp();

		bool godeeper = true;
		while( godeeper && daughters->size() > 0)
		{
			godeeper = false;
			for(int i=0; i<daughters->size(); ++i)
			{
				VPlacedVolume const * nextvolume = (*daughters)[i];
				Vector3D<Precision> transformedpoint;

				if( nextvolume->Inside( tmp, transformedpoint ) )
				{
					path.Push( nextvolume );
					tmp = transformedpoint;
					candvolume =  nextvolume;
					daughters = candvolume->logical_volume()->daughtersp();
					godeeper=true;
					break;
				}
			}
		}
	}
	return candvolume;
}

VPlacedVolume const *
SimpleNavigator::RelocatePointFromPath( Vector3D<Precision> const & localpoint,
										NavigationState & path ) const
{
	// idea: do the following:
	// ----- is localpoint still in current mother ? : then go down
	// if not: have to go up until we reach a volume that contains the
	// localpoint and then go down again (neglecting the volumes currently stored in the path)
	VPlacedVolume const * currentmother = path.Top();
	if( currentmother != NULL )
	{
        Vector3D<Precision> tmp = localpoint;
		// go up iteratively
		while( currentmother && ! currentmother->UnplacedInside( tmp ) )
		{
			path.Pop();
			Vector3D<Precision> pointhigherup = currentmother->matrix()->InverseTransform( tmp );
			tmp=pointhigherup;
			currentmother=path.Top();
		}

		if(currentmother)
		{
			path.Pop();
			// may inline this
			return LocatePoint(currentmother, tmp, path, false);
		}
	}
	return currentmother;
}


VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool
SimpleNavigator::HasSamePath( Vector3D<Precision> const & globalpoint,
							  NavigationState & currentstate,
							  NavigationState & newstate ) const
{
	TransformationMatrix const & m = currentstate.TopMatrix();
	Vector3D<Precision> localpoint = m.Transform<1,0>(globalpoint);
	newstate = currentstate;
	RelocatePointFromPath( localpoint, newstate );
	return newstate.Top() == currentstate.Top();
}


void
SimpleNavigator::FindNextBoundaryAndStep( Vector3D<Precision> const & globalpoint,
		  	  	  	  	  	  	  	  	  Vector3D<Precision> const & globaldir,
		  	  	  	  	  	  	  	  	  NavigationState const & currentstate,
		  	  	  	  	  	  	  	  	  NavigationState & newstate,
		  	  	  	  	  	  	  	  	  Precision const & pstep,
		  	  	  	  	  	  	  	  	  Precision & step
) const
{
	// this information might have been cached in previous navigators??
	TransformationMatrix const & m = const_cast<NavigationState &> ( currentstate ).TopMatrix();
	Vector3D<Precision> localpoint=m.Transform<1,0>(globalpoint);
	Vector3D<Precision> localdir=m.Transform<0,0>(globaldir);

	VPlacedVolume const * currentvolume = currentstate.Top();
	int nexthitvolume = -1; // means mother

	step = currentvolume->DistanceToOut( localpoint, localdir, pstep );

	// iterate over all the daughter
	Vector<Daughter> const * daughters = currentvolume->logical_volume()->daughtersp();

	for(int d = 0; d<daughters->size(); ++d)
	{
		VPlacedVolume const * daughter = daughters->operator [](d);
		//	 previous distance becomes step estimate, distance to daughter returned in workspace
		Precision ddistance = daughter->DistanceToIn( localpoint, localdir, step );

		nexthitvolume = (ddistance < step) ? d : nexthitvolume;
		step 	  = (ddistance < step) ? ddistance  : step;
	}

	// now we have the candidates
	// try
	newstate=currentstate;
	assert( newstate.Top() == currentstate.Top() );

	// TODO: this is tedious, please provide operators in Vector3D!!
	// WE SHOULD HAVE A FUNCTION "TRANSPORT" FOR AN OPERATION LIKE THIS
	Vector3D<Precision> newpointafterboundary = localdir;
	newpointafterboundary*=(step + 1e-9);
	newpointafterboundary+=localpoint;

	if( nexthitvolume != -1 ) // not hitting mother
	{
		// continue directly further down
		VPlacedVolume const * nextvol = daughters->operator []( nexthitvolume );
		TransformationMatrix const *m = nextvol->matrix();

		// this should be inlined here
		LocatePoint( nextvol, m->Transform<1,0>(newpointafterboundary), newstate, false );
	}
	else
	{
		// continue directly further up
		//LocateLocalPointFromPath_Relative_Iterative( newpointafterboundary, newpointafterboundaryinnewframe, outpath, globalm );
		RelocatePointFromPath( newpointafterboundary, newstate );
	}
}

// this is just the brute force method; need to see whether it makes sense to combine it into
// the FindBoundaryAndStep function
Precision SimpleNavigator::GetSafety(Vector3D<Precision> const & globalpoint,
									 NavigationState const & currentstate) const
{
	// this information might have been cached already ??
	TransformationMatrix const & m = const_cast<NavigationState &>(currentstate).TopMatrix();
	Vector3D<Precision> localpoint=m.Transform<1,0>(globalpoint);

	// safety to mother
	VPlacedVolume const * currentvol = currentstate.Top();
	double safety = currentvol->SafetyToOut( localpoint );
//	std::cerr << "SafetyToOut "  << safety << std::endl;
	assert( safety > 0);

	// safety to daughters
	Vector<Daughter> const * daughters = currentvol->logical_volume()->daughtersp();
	int numberdaughters = daughters->size();
	for(int d = 0; d<numberdaughters; ++d)
	{
		VPlacedVolume const * daughter = daughters->operator [](d);
		double tmp = daughter->SafetyToIn( localpoint );
		std::cerr << "SafetyToIn "  << tmp << std::endl;
		safety = std::min(safety, tmp);
	}
	return safety;
}



/**
 * Navigation interface for baskets; templates on Container3D which might be a SOA3D or AOS3D container
 */

/*
template <typename Container3D>
void FindNextBoundaryAndStep(
		Container3D const & globalpoints,
		Container3D const & globaldirs,
		Container3D & localpoints,
		Container3D & localdirs,
		NavigationState * * currentstates   this is interpreted as an array of pointers to NavigationStates ,
		NavigationState * * newstates  this is interpreted as an array of pointers to NavigationStates ,
		Precision const * pSteps  -- proposed steps ,
		Precision * safeties  safeties ,
		Precision * workspace,
		Precision * distances  distances ,
		int * nextnode,
		int np) const
{
	// assuming that points and dirs are always global ones,
	// we need to transform to local coordinates first of all
	for( int i=0; i<np; ++i )
	{
		// TODO: we might be able to cache the matrices because some of the paths will be identical
		// need to have a quick way ( hash ) to compare paths
		TransformationMatrix m = currentstates[i]->TopMatrix();
		localpoints[i] = m.Transform<1,0>( globalpoints[i] );
		localdirs[i] = m.Transform<0,0>( globaldirs[i] );
		// initiallize next nodes
		nextnode[i]=-1; // -2 indicates that particle stays in the same logical volume
	}

	localpoints.SetFillSize(np);
	localdirs.SetFillSize(np);

	// attention here: the placed volume will of course differ for the particles;
	// however the distancetoout function and the daughtelist are the same for all particles
	VPlacedVolume const * currentvolume = currentstates[0]->Top();

	// calculate distance to Boundary of current volume
	currentvolume->DistanceToOut( localpoints, localdirs, pSteps, distances );

	//	nextnode[k]=-1;
	// this should be moved into the previous function
	//	for(int k=0;k<np;k++)
	//	{
	//		this->nextnode[k] = ( pSteps[k] < distance[k] )? -1 : nextnode[k];
	//	}

	// iterate over all the daughter
	Vector<Daughter> const * daughters = currentvolume->logical_volume()->daughtersp();
	for( int daughterindex=0; daughterindex < daughters->size(); ++daughterindex )
	{
		VPlacedVolume const * daughter = daughters->operator [](daughterindex);

		daughter->DistanceToIn( localpoints, localdirs, distances, workspace );

		// TODO: this has to be moved inside the above function
		for(int k=0; k<np; k++)
		{
			if(workspace[k]<distances[k])
			{
				distances[k] = workspace[k];
				nextnode[k] = daughterindex;
			}
		}
		//daughter->DistanceToIn_ANDUpdateCandidate( points, dirs, distance, nextnode, daughterindex );
		// at this moment, distance will contain the MINIMUM distance until now and nextnode will point to the candidate nextnode
	}
	VectorRelocateFromPaths( localpoints, localdirs,
			distances, nextnode, const_cast<NavigationState const **>(currentstates), newstates, np );
}
*/


#ifdef VECGEOM_ROOT
void SimpleNavigator::InspectEnvironmentForPointAndDirection
	(	Vector3D<Precision> const & globalpoint,
		Vector3D<Precision> const & globaldir,
		NavigationState const & state ) const
{
	TransformationMatrix m = const_cast<NavigationState &>(state).TopMatrix();
	Vector3D<Precision> localpoint = m.Transform<1,0>( globalpoint );
	Vector3D<Precision> localdir = m.Transform<0,0>( globaldir );

	// check that everything is consistent
	{
		NavigationState tmpstate( state );
		tmpstate.Clear();
		assert( LocatePoint( GeoManager::Instance().world(), globalpoint, tmpstate, true ) == state.Top() );
	}

	// now check mother and daughters
	VPlacedVolume const * currentvolume = state.Top();
	std::cout << "############################################ " << std::endl;
	std::cout << "Navigating in placed volume : " << RootManager::Instance().GetName( currentvolume ) << std::endl;

	int nexthitvolume = -1; // means mother
	double step = currentvolume->DistanceToOut( localpoint, localdir );

	std::cout << "DistanceToOutMother : " << step << std::endl;

	// iterate over all the daughters
	Vector<Daughter> const * daughters = currentvolume->logical_volume()->daughtersp();

	std::cout << "ITERATING OVER " << daughters->size() << " DAUGHTER VOLUMES " << std::endl;
	for(int d = 0; d<daughters->size(); ++d)
	{
		VPlacedVolume const * daughter = daughters->operator [](d);
		//	 previous distance becomes step estimate, distance to daughter returned in workspace
		Precision ddistance = daughter->DistanceToIn( localpoint, localdir, step );

		std::cout << "DistanceToDaughter : " << RootManager::Instance().GetName( daughter ) << " " << ddistance << std::endl;

		nexthitvolume = (ddistance < step) ? d : nexthitvolume;
		step 	  = (ddistance < step) ? ddistance  : step;
	}
	std::cout << "DECIDED FOR NEXTVOLUME " << nexthitvolume << std::endl;

	// same information from ROOT
	TGeoNode const * currentRootNode = RootManager::Instance().tgeonode( currentvolume );
	double lp[3]={localpoint[0],localpoint[1],localpoint[2]};
	double ld[3]={localdir[0],localdir[1],localdir[2]};
	double rootstep =  currentRootNode->GetVolume()->GetShape()->DistFromInside( lp, ld, 3, 1E30, 0 );
	std::cout << "---------------- CMP WITH ROOT ---------------------" << std::endl;
	std::cout << "DistanceToOutMother ROOT : " << rootstep << std::endl;
	std::cout << "ITERATING OVER " << currentRootNode->GetNdaughters() << " DAUGHTER VOLUMES " << std::endl;
	for( int d=0; d<currentRootNode->GetNdaughters();++d)
	{
		TGeoMatrix const * m = currentRootNode->GetMatrix();
		double llp[3], lld[3];
		m->MasterToLocal(lp, llp);
		m->MasterToLocalVect(ld, lld);
		TGeoNode const * daughter=currentRootNode->GetDaughter(d);
		Precision ddistance = daughter->GetVolume()->GetShape()->DistFromOutside(llp,llp,3,1E30,0);

		std::cout << "DistanceToDaughter ROOT : " << daughter->GetName() << " " << ddistance << std::endl;
	}
}
#endif




};

#endif /* SIMPLE_NAVIGATOR_H_ */
