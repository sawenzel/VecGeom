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

namespace vecgeom
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
	* A function to navigate ( find next boundary and/or the step to do )
	*/
	VECGEOM_CUDA_HEADER_BOTH
	void FindNextBoundaryAndStep( Vector3D<Precision> const & /* global point */,
								  Vector3D<Precision> const & /* global dir */,
								  NavigationState const & /* currentstate */,
								  NavigationState & /* newstate */,
								  double & /*step*/
								 ) const;

};

VPlacedVolume const *
SimpleNavigator::LocatePoint( VPlacedVolume const * vol, Vector3D<Precision> const & point,
							  NavigationState & path, bool top ) const
{
	VPlacedVolume const * candvolume = vol;
	Vector3D<Precision> tmp(point);
	if( top )
	{
		candvolume = ( vol->Inside( point ) ) ? vol : 0;
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
					break;
				}
			}
		}
	}
	return vol;
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
		while( currentmother && ! currentmother->Inside( tmp ) )
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


void
SimpleNavigator::FindNextBoundaryAndStep( Vector3D<Precision> const & globalpoint,
		  	  	  	  	  	  	  	  	  Vector3D<Precision> const & globaldir,
		  	  	  	  	  	  	  	  	  NavigationState const & currentstate,
		  	  	  	  	  	  	  	  	  NavigationState & newstate,
		  	  	  	  	  	  	  	  	  Precision & step
) const
{
	// this information might have been cached in previous navigators??
	TransformationMatrix const & m = const_cast<NavigationState &> ( currentstate ).TopMatrix();
	Vector3D<Precision> localpoint=m.Transform<1,0>(globalpoint);
	Vector3D<Precision> localdir=m.Transform<1,0>(globaldir);

	VPlacedVolume const * currentvolume = currentstate.Top();
	int nexthitvolume = -1; // means mother

	step = currentvolume->DistanceToOut( localpoint, localdir );

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

	// TODO: this is tedious, please provide operators in Vector3D!!
	// WE SHOULD HAVE A FUNCTION "TRANSPORT" FOR AN OPERATION LIKE THIS
	Vector3D<Precision> newpointafterboundary = localdir;
	newpointafterboundary*=(step + kGTolerance);
	newpointafterboundary+=localpoint;

	if( nexthitvolume != -1 ) // not hitting mother
	{
		// continue directly further down
		VPlacedVolume const * nextvol = currentstate.Top()->logical_volume()->daughtersp()->operator []( nexthitvolume );
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



};

#endif /* SIMPLE_NAVIGATOR_H_ */
