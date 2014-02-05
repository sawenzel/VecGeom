/*
 * SimpleVecNavigator.h
 *
 *  Created on: Dec 12, 2013
 *      Author: swenzel
 */

#ifndef SIMPLEVECNAVIGATOR_H_
#define SIMPLEVECNAVIGATOR_H_

#include "PhysicalVolume.h"
#include "Vector3D.h"

// the volume path class caches some important properties about where particles are in the geometry hierarchy
class VolumePath
{
	private:
		int fmaxlevel;
		int fcurrentlevel;
		PhysicalVolume const *  * path;
		TransformationMatrix const * * cache_of_globalmatrices;

	public:
		VolumePath(int);

		PhysicalVolume const * At(int i) const;
		void SetAt(int i, PhysicalVolume const *);

		// better to use pop and push
		void Push( PhysicalVolume const *);
		PhysicalVolume const * Top() const;
		void Pop();

		// clear all information
		void Clear();

		void Print() const;

		void GetGlobalMatrixFromPath( TransformationMatrix * m ) const;
};

inline
void VolumePath::GetGlobalMatrixFromPath( TransformationMatrix * m ) const
{
	for(int i=0; i<fcurrentlevel; i++ )
	{
		m->Multiply( path[i]->getMatrix() );
	}
}

inline
PhysicalVolume const * VolumePath::At(int i) const
{
	return path[i];
}


inline
void
VolumePath::SetAt(int i, PhysicalVolume const * v)
{
	path[i]=v;
}

inline
void
VolumePath::Pop()
{
	path[fcurrentlevel--]=0;
}

inline
void
VolumePath::Push( PhysicalVolume const * v)
{
	path[fcurrentlevel++]=v;
}


inline
PhysicalVolume const *
VolumePath::Top() const
{
	return path[fcurrentlevel];
}

inline
void
VolumePath::Clear()
{
	for(int i=0;i<fcurrentlevel;i++) path[0]=0;
	fcurrentlevel=0;
}


class SimpleVecNavigator {
private:
	double * workspace;
	// for transformed points and dirs
	Vectors3DSOA transformedpoints;
	Vectors3DSOA transformeddirs;

	PhysicalVolume const * top;

public:
	SimpleVecNavigator(int, PhysicalVolume const *);
	SimpleVecNavigator(int);
	virtual ~SimpleVecNavigator();


	void
	DistToNextBoundary( PhysicalVolume const *, Vectors3DSOA const & /*points*/,
												Vectors3DSOA const & /*dirs*/,
												double const * /*steps*/,
												double * /*distance*/,
												PhysicalVolume ** nextnode, int np ) const;

	void
	DistToNextBoundaryUsingUnplacedVolumes( PhysicalVolume const *, Vectors3DSOA const & /*points*/,
												Vectors3DSOA const & /*dirs*/,
												double const * /*steps*/,
												double * /*distance*/,
												PhysicalVolume ** nextnode, int np ) const;

	void
	DistToNextBoundaryUsingUnplacedVolumesButSpecializedMatrices( PhysicalVolume const *, Vectors3DSOA const & /*points*/,
													Vectors3DSOA const & /*dirs*/,
													double const * /*steps*/,
													double * /*distance*/,
													PhysicalVolume ** nextnode, int np ) const;


	void
	DistToNextBoundaryUsingROOT( PhysicalVolume const *,
								 double const * /*points*/,
								 double const * /*dirs*/,
								 double const * /*steps*/,
								 double * /*distance*/,
								 PhysicalVolume ** nextnode, int np ) const;

	void
	DistToNextBoundaryUsingUSOLIDS( PhysicalVolume const *, Vectors3DSOA const & /*points*/,
									Vectors3DSOA const & /*dirs*/,
									double const * /*steps*/,
									double * /*distance*/,
									PhysicalVolume ** nextnode, int np ) const;



	PhysicalVolume const *
	LocateGlobalPoint(PhysicalVolume const *, Vector3D const & globalpoint, Vector3D & localpoint, VolumePath &path, TransformationMatrix *, bool top=true) const;

	PhysicalVolume const *
	LocateGlobalPoint(PhysicalVolume const *, Vector3D const & globalpoint, Vector3D & localpoint, VolumePath &path, bool top=true) const;


	PhysicalVolume const *
	// this location starts from the a localpoint in the reference frame of inpath.Top() to find the new location ( if any )
	// we might need some more input here ( like the direction )
	LocateLocalPointFromPath(Vector3D const & localpoint, VolumePath const & inpath, VolumePath & newpath, TransformationMatrix * ) const;

	PhysicalVolume const *
	// this location starts from the a localpoint in the reference frame of inpath.Top() to find the new location ( if any )
	// we might need some more input here ( like the direction )
	LocateLocalPointFromPath_Relative(Vector3D const & localpoint, VolumePath & path, TransformationMatrix * ) const;


};

#endif /* SIMPLEVECNAVIGATOR_H_ */
