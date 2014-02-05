/*
 * SimpleVecNavigator.cpp
 *
 *  Created on: Dec 12, 2013
 *      Author: swenzel
 */

#include "SimpleVecNavigator.h"
#include "GlobalDefs.h"
#include "Vc/vector.h"
#include <list>
#include "TGeoShape.h"
#include "TGeoMatrix.h"
#include "UVector3.hh"
#include "VUSolid.hh"
#include "mm_malloc.h"
#include <iostream>
#include "TransformationMatrix.h"


VolumePath::VolumePath(int maxlevel) : fmaxlevel(maxlevel), fcurrentlevel(0)
{
	fmaxlevel = maxlevel;

	path= (PhysicalVolume const * *) _mm_malloc( sizeof(PhysicalVolume const *) * maxlevel, ALIGNMENT_BOUNDARY );
	cache_of_globalmatrices = ( TransformationMatrix const * *) _mm_malloc( sizeof(TransformationMatrix) * maxlevel, ALIGNMENT_BOUNDARY );
}


void VolumePath::Print() const
{
	std::cerr << "level" << fcurrentlevel << std::endl;
}


SimpleVecNavigator::SimpleVecNavigator(int np, PhysicalVolume const * t) : top(t)  {
	// TODO Auto-generated constructor stub
	workspace = (double * ) _mm_malloc( sizeof(double)*np, ALIGNMENT_BOUNDARY );
	transformeddirs.alloc(np);
	transformedpoints.alloc(np);
}


SimpleVecNavigator::SimpleVecNavigator(int np)  {
	// TODO Auto-generated constructor stub
	workspace = (double * ) _mm_malloc( sizeof(double)*np, ALIGNMENT_BOUNDARY );
	transformeddirs.alloc(np);
	transformedpoints.alloc(np);
}


SimpleVecNavigator::~SimpleVecNavigator() {
	// TODO Auto-generated destructor stub
}

static
inline
void
__attribute__((always_inline))
MIN_v ( double * __restrict__ step, double const * __restrict__ workspace, uint64_t * __restrict__ nextnodepointersasints,
		uint64_t __restrict__ curcandidateVolumePointerasint, unsigned int np )
{
	// this is not vectorizing !
	for(auto k=0;k<np;++k)
    {
      step[k] = (workspace[k]<step[k]) ? workspace[k] : step[k];
      nextnodepointersasints[k] = (workspace[k]< step[k]) ?  curcandidateVolumePointerasint : nextnodepointersasints[k];
    	  //	  std::cout << " hit some node " << curcandidateNode << std::endl;
    }
}

static
inline
void
__attribute__((always_inline))
MIN_v ( double * __restrict__ step, double const * __restrict__ workspace, PhysicalVolume ** __restrict__ nextnodepointersasints,
		PhysicalVolume const * __restrict__ curcandidateVolumePointerasint, unsigned int np )
{
	// this is not vectorizing !
	for(auto k=0;k<np;++k)
    {
		bool cond = workspace[k]<step[k];
		step[k] = (cond)? workspace[k] : step[k];
		nextnodepointersasints[k] = (cond)? curcandidateVolumePointerasint : nextnodepointersasints[k];
    }
}

static
inline
void
__attribute__((always_inline))
MINVc_v ( double * __restrict__ step, double const * __restrict__ workspace, uint64_t * __restrict__ nextnodepointersasints,
		uint64_t  __restrict__ curcandidateVolumePointerasint, unsigned int np )
{

	// this is not vectorizing !
	//for(auto k=0;k<np;++k)
    //{
//      step[k] = (workspace[k]<step[k]) ? workspace[k] : step[k];
      //nextnodepointersasints[k] = (workspace[k]< step[k]) ?  curcandidateVolumePointerasint : nextnodepointersasints[k];
    	  //	  std::cout << " hit some node " << curcandidateNode << std::endl;
    //}
	for(auto k=0;k<np;k+=Vc::double_v::Size)
	{
		Vc::double_v stepvec(&step[k]);
		Vc::double_v pointervec((double *) (&nextnodepointersasints[k]));
		Vc::double_v workspacevec(&workspace[k]);
		Vc::double_m m = stepvec < workspacevec;
		stepvec( m ) = workspacevec;
		pointervec( m ) = curcandidateVolumePointerasint;
		stepvec.store(&step[k]);
		pointervec.store((double *) (&nextnodepointersasints[k]));
	}
}

void SimpleVecNavigator::DistToNextBoundary( PhysicalVolume const * volume, Vectors3DSOA const & points,
											 Vectors3DSOA const & dirs,
											 double const * steps,
											 double * distance,
											 PhysicalVolume ** nextnode, int np ) const
{
// init nextnode ( maybe do a memcpy )
	for(auto k=0;k<np;++k)
	{
		nextnode[k]=0;
	}

	// calculate distance to Boundary of current volume
	volume->DistanceToOut( points, dirs, steps, distance );

	// iterate over all the daughter
	std::list<PhysicalVolume const *> const * daughters = volume->GetDaughterList();
	for( auto iter = daughters->begin(); iter!=daughters->end(); ++iter )
	{
		PhysicalVolume const * daughter = (*iter);
		// previous distance become step estimate, distance to daughter returned in workspace
		daughter->DistanceToIn( points, dirs, distance, workspace );

		// update dist and current nextnode candidate
		// maybe needs to be vectorized manually with Vc
		// MINVc_v(distance, workspace, reinterpret_cast<uint64_t *>(nextnode), reinterpret_cast<uint64_t> (daughter), np);
		// MIN_v(distance, workspace, reinterpret_cast<uint64_t *>(nextnode), reinterpret_cast<uint64_t> (daughter), np);
		MIN_v(distance, workspace, nextnode, daughter, np);
	}
}


void SimpleVecNavigator::DistToNextBoundaryUsingUnplacedVolumes( PhysicalVolume const * volume, Vectors3DSOA const & points,
											 Vectors3DSOA const & dirs,
											 double const * steps,
											 double * distance,
											 PhysicalVolume ** nextnode, int np ) const
{
// init nextnode ( maybe do a memcpy )
	for(auto k=0;k<np;++k)
	{
		nextnode[k]=0;
	}

	// calculate distance to Boundary of current volume
	volume->DistanceToOut( points, dirs, steps, distance );

	// iterate over all the daughter
	std::list<PhysicalVolume const *> const * daughters = volume->GetDaughterList();
	for( auto iter = daughters->begin(); iter!=daughters->end(); ++iter )
	{
		PhysicalVolume const * daughter = (*iter);

		PhysicalVolume const * unplacedvolume = daughter->GetAsUnplacedVolume();

		// previous distance become step estimate, distance to daughter returned in workspace

		// now need to do matrix transformations outside
		TransformationMatrix const * tm = unplacedvolume->getMatrix();
		tm->MasterToLocalCombined( points, const_cast<Vectors3DSOA &>(transformedpoints),
				dirs, const_cast<Vectors3DSOA &>(transformeddirs ));

		unplacedvolume->DistanceToIn( transformedpoints, transformeddirs, distance, workspace );

		// update dist and current nextnode candidate
		// maybe needs to be vectorized manually with Vc
		for( auto k=0; k < np; ++k )
		{
			if( workspace[k] < distance[k] )
			{
				distance[k]=workspace[k];
				nextnode[k]=const_cast<PhysicalVolume *>(daughter);
			}
		}
	}
}


void SimpleVecNavigator::DistToNextBoundaryUsingROOT(  PhysicalVolume const * volume, double const * points,
		 	 	 	 	 	 	 	 	 	 	 	   double const * dirs,
		 	 	 	 	 	 	 	 	 	 	 	   double const * steps,
		 	 	 	 	 	 	 	 	 	 	 	   double * distance,
		 	 	 	 	 	 	 	 	 	 	 	   PhysicalVolume ** nextnode, int np ) const
{
	for(auto k=0;k<np;k++)
	{
		nextnode[k]=0;

		distance[k]=(volume->GetAsUnplacedROOTSolid())->DistFromInside(const_cast<double *>( &points[3*k] ),
																	const_cast<double *>(&dirs[3*k]),
																		3,	steps[k], 0 );

		std::list<PhysicalVolume const *> const * daughters = volume->GetDaughterList();
		for( auto iter = daughters->begin(); iter!=daughters->end(); ++iter )
			{
				PhysicalVolume const * daughter = (*iter);
				TGeoShape const * unplacedvolume = daughter->GetAsUnplacedROOTSolid();

				// previous distance become step estimate, distance to daughter returned in workspace

				// now need to do matrix transformations outside
				TGeoMatrix const * rm = daughter->getMatrix()->GetAsTGeoMatrix();
				double localpoint[3];
				double localdir[3];
				rm->MasterToLocal( &points[3*k], localpoint );
				rm->MasterToLocalVect( &dirs[3*k], localdir );

				double distd = unplacedvolume->DistFromOutside( localpoint, localdir, 3, distance[k], 0 );

				if( distd < distance[k] )
				{
					distance[k] = distd;
					nextnode[k] = const_cast<PhysicalVolume *>(daughter);
				}
			}
	}
}



void SimpleVecNavigator::DistToNextBoundaryUsingUSOLIDS( PhysicalVolume const * volume, Vectors3DSOA const & points,
		 	 	 	 	 	 	 	 	 	 	 	 	 Vectors3DSOA const & dirs,
		 	 	 	 	 	 	 	 	 	 	 	 	 double const * steps,
		 	 	 	 	 	 	 	 	 	 	 	 	 double * distance,
		 	 	 	 	 	 	 	 	 	 	 	 	 PhysicalVolume ** nextnode, int np ) const
{
	for(auto k=0;k<np;k++)
	{
		nextnode[k]=0;
		UVector3 normal;
		Vector3D p = points.getAsVector(k);
		Vector3D d = dirs.getAsVector(k);
		bool conv;

		distance[k]=(volume->GetAsUnplacedUSolid())->DistanceToOut( reinterpret_cast<UVector3 const &>( p ),
																	reinterpret_cast<UVector3 const &>( d ),
																	normal,
																	conv,
																	steps[k] );

		std::list<PhysicalVolume const *> const * daughters = volume->GetDaughterList();
		for( auto iter = daughters->begin(); iter!=daughters->end(); ++iter )
			{
				PhysicalVolume const * daughter = (*iter);
				VUSolid const * unplacedvolume = daughter->GetAsUnplacedUSolid();

				// previous distance become step estimate, distance to daughter returned in workspace

				// now need to do matrix transformations outside / here we inline them
				Vector3D localpoint, localdir;
				(*iter)->getMatrix()->MasterToLocal<1,-1>( p, localpoint );
				(*iter)->getMatrix()->MasterToLocalVec<-1>( d, localdir );

				double distd = unplacedvolume->DistanceToIn( reinterpret_cast<UVector3 const &> (localpoint),
															 reinterpret_cast<UVector3 const &>(localdir), distance[k] );

				if( distd < distance[k] )
				{
					distance[k] = distd;
					nextnode[k] = const_cast<PhysicalVolume *>(daughter);
				}
			}
	}
}

// assumptions:
// global point is given in coordinates of the frame of reference of vol ( normally vol will be the world )

// we could specialize this function on whether top or not
// this is very special at the moment; no treatment of boundary information
// we might have to pass the directions as well -- but then we would have to transform them as well ( bugger )
PhysicalVolume const * SimpleVecNavigator::LocateGlobalPoint(PhysicalVolume const * vol,
		Vector3D const & globalpoint, Vector3D & localpoint, VolumePath & path, TransformationMatrix * globalm, bool top=true) const
{
	PhysicalVolume const * candvolume = vol;
	if( top ) candvolume = ( vol->UnplacedContains( globalpoint ) )? vol : 0;
	if( candvolume )
	{
		path.Push( candvolume );
		std::list< PhysicalVolume const * > * dlist = vol->GetDaughterList();
		std::list< PhysicalVolume const * >::iterator iter;
		for(iter=dlist->begin(); iter!=dlist->end(); iter++)
		{
			PhysicalVolume const * nextvol=(*iter);
			Vector3D transformedpoint;
			if(nextvol->Contains(globalpoint, transformedpoint, globalm))
			{
				// this is no longer the top ( so setting top to false )
				localpoint.x = transformedpoint.x;
				localpoint.y = transformedpoint.y;
				localpoint.z = transformedpoint.z;
				candvolume = LocateGlobalPoint(nextvol, transformedpoint, localpoint, path, globalm, false);
				break;
			}
		}
	}
	return candvolume;

	// TODO: fill the path while going down
	// at the very end: do the matrix multiplications and caching of global matrices -- doing this in a loop will be icache friendly
}

PhysicalVolume const * SimpleVecNavigator::LocateGlobalPoint(PhysicalVolume const * vol,
		Vector3D const & globalpoint, Vector3D & localpoint, VolumePath & path, bool top=true) const
{
	PhysicalVolume const * candvolume = vol;
	if( top ) candvolume = ( vol->UnplacedContains( globalpoint ) )? vol : 0;
	if( candvolume )
	{
		path.Push( candvolume );
		std::list< PhysicalVolume const * > * dlist = vol->GetDaughterList();
		std::list< PhysicalVolume const * >::iterator iter;
		for(iter=dlist->begin(); iter!=dlist->end(); iter++ )
		{
			PhysicalVolume const * nextvol=(*iter);
			Vector3D transformedpoint;
			if( nextvol->Contains( globalpoint, transformedpoint ))
			{
				// this is no longer the top ( so setting top to false )
				localpoint.x = transformedpoint.x;
				localpoint.y = transformedpoint.y;
				localpoint.z = transformedpoint.z;
				candvolume = LocateGlobalPoint(nextvol, transformedpoint, localpoint, path, false);
				break;
			}
		}
	}
	return candvolume;

	// TODO: fill the path while going down
	// at the very end: do the matrix multiplications and caching of global matrices -- doing this in a loop will be icache friendly
}


PhysicalVolume const * SimpleVecNavigator::LocateLocalPointFromPath(
		Vector3D const & localpoint, VolumePath const & oldpath, VolumePath & newpath, TransformationMatrix * globalm ) const
{
 // very simple at the moment
	TransformationMatrix globalmatrix;
	oldpath.GetGlobalMatrixFromPath( &globalmatrix );
	Vector3D globalpoint;
	globalmatrix.LocalToMaster( localpoint, globalpoint );
	Vector3D newlocalpoint;
	return LocateGlobalPoint( top, globalpoint, newlocalpoint, newpath, globalm );
}


PhysicalVolume const * SimpleVecNavigator::LocateLocalPointFromPath_Relative(
		Vector3D const & localpoint, VolumePath & path, TransformationMatrix * globalm) const
{
	// idea: do the following:
	// ----- is localpoint still in current mother ? : then go down
	// if not: have to go up until we reach a volume that contains the localpoint and then go down again (neglecting the volumes currently stored in the path)
	PhysicalVolume const * currentmother = path.Top();
	if( currentmother->UnplacedContains(localpoint) )
	{
		// go further down -- can use LocateGlobalPoint function for this
		Vector3D newlocalpoint;

		// at this moment would have to retrieve global matrix from path
		path.GetGlobalMatrixFromPath( globalm );

		return LocateGlobalPoint(currentmother, localpoint, newlocalpoint, path, globalm, false);
	}
	else
	{
		// convert localpoint to reference frame of mother

		// get rid of current volume in path
		path.Pop();

		Vector3D pointhigherup;
		currentmother->getMatrix()->LocalToMaster(localpoint, pointhigherup);
		return LocateLocalPointFromPath_Relative(pointhigherup, path, globalm);
	}
}



