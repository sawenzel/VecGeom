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

};

#endif /* SIMPLE_NAVIGATOR_H_ */
