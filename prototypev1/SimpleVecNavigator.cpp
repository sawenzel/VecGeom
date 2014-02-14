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
#include "Vector3DFast.h"

class PhysicalVolume;
VolumePath::VolumePath(int maxlevel) : fmaxlevel(maxlevel), fcurrentlevel(0)
{
	fmaxlevel = maxlevel;

	path= (PhysicalVolume const * *) _mm_malloc( sizeof(PhysicalVolume const *) * maxlevel, ALIGNMENT_BOUNDARY );
	cache_of_globalmatrices = ( TransformationMatrix const * *) _mm_malloc( sizeof(TransformationMatrix) * maxlevel, ALIGNMENT_BOUNDARY );
	Clear();
}


void VolumePath::Print() const
{
	std::cerr << "level" << fcurrentlevel << std::endl;
	for( int i=0;i<fcurrentlevel;i++ )
	{
		std::cerr << " " << i << " " << path[i] << std::endl;
	}


}

// function compares two paths and calculated distance along shortest path connecting them on the tree
int VolumePath::Distance( VolumePath const & other ) const
{
    int lastcommonlevel;
    int maxlevel = std::max( GetCurrentLevel(), other.GetCurrentLevel() );

    //  algorithm: start on top and go down until paths split
    for(int i=0; i< maxlevel; i++)
	{
    	PhysicalVolume const *v1=this->At(i);
		PhysicalVolume const *v2=other.At(i);
		if( v1 == v2 )
		{
			lastcommonlevel = i;
		}
		else
		{
			break;
		}
	}
	return (GetCurrentLevel()-lastcommonlevel) + ( other.GetCurrentLevel() - lastcommonlevel ) - 2;
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
MIN_v ( double * __restrict__ step, double const * __restrict__ workspace, const PhysicalVolume ** __restrict__ nextnodepointersasints,
		PhysicalVolume const * __restrict__ curcandidateVolumePointerasint, unsigned int np )
{
	// this is not vectorizing !
	for(auto k=0;k<np;++k)
    {
		bool cond = workspace[k]<step[k];
		step[k] = (cond)? workspace[k] : step[k];
		nextnodepointersasints[k] = (cond)? const_cast<PhysicalVolume *>(curcandidateVolumePointerasint) : nextnodepointersasints[k];
    }
}



void SimpleVecNavigator::DistToNextBoundary( PhysicalVolume const * volume, Vectors3DSOA const & points,
											 Vectors3DSOA const & dirs,
											 double const * steps,
											 double * distance,
											 const PhysicalVolume ** nextnode, int np )
{
// init nextnode ( maybe do a memcpy )
	for(auto k=0;k<np;++k)
	{
		nextnode[k]=0;
	}

	// calculate distance to Boundary of current volume
	volume->DistanceToOut( points, dirs, steps, distance );

	// iterate over all the daughter
	PhysicalVolume::DaughterContainer_t const * daughters = volume->GetDaughters();
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
											 const PhysicalVolume ** nextnode, int np )
{
// init nextnode ( maybe do a memcpy )
	for(auto k=0;k<np;++k)
	{
		nextnode[k]=0;
	}

	// calculate distance to Boundary of current volume
	volume->DistanceToOut( points, dirs, steps, distance );

	// iterate over all the daughter
	PhysicalVolume::DaughterContainer_t const * daughters = volume->GetDaughters();
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
		 	 	 	 	 	 	 	 	 	 	 	   const PhysicalVolume ** nextnode, int np )
{
	for(auto k=0;k<np;k++)
	{
		nextnode[k]=0;

		distance[k]=(volume->GetAsUnplacedROOTSolid())->DistFromInside(const_cast<double *>( &points[3*k] ),
																	const_cast<double *>(&dirs[3*k]),
																		3,	steps[k], 0 );


		PhysicalVolume::DaughterContainer_t const * daughters = volume->GetDaughters();
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
		 	 	 	 	 	 	 	 	 	 	 	 	 const PhysicalVolume ** nextnode, int np )
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

		PhysicalVolume::DaughterContainer_t const * daughters = volume->GetDaughters();
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
PhysicalVolume const * SimpleVecNavigator::LocatePoint(PhysicalVolume const * vol,
		Vector3D const & point, Vector3D & localpoint, VolumePath & path, TransformationMatrix * globalm, bool top) const
{
	PhysicalVolume const * candvolume = vol;
	if( top ) candvolume = ( vol->UnplacedContains( point ) )? vol : 0;
	if( candvolume )
	{
		path.Push( candvolume );
		PhysicalVolume::DaughterContainer_t const * dlist = candvolume->GetDaughters();
		PhysicalVolume::DaughterContainerConstIterator_t iter;
		for(auto iter=dlist->begin(); iter!=dlist->end(); iter++)
		{
			PhysicalVolume const * nextvol=(*iter);
			Vector3D transformedpoint;
			if(nextvol->Contains(point, transformedpoint, globalm))
			{
				// this is no longer the top ( so setting top to false )
				localpoint = transformedpoint;
				candvolume = LocatePoint(nextvol, transformedpoint, localpoint, path, globalm, false);
				break;
			}
		}
	}
	return candvolume;

	// TODO: fill the path while going down
	// at the very end: do the matrix multiplications and caching of global matrices -- doing this in a loop will be icache friendly
}






PhysicalVolume const * SimpleVecNavigator::LocatePoint(PhysicalVolume const * vol,
		Vector3D const & point, Vector3D & localpoint, VolumePath & path, bool top) const
{
	PhysicalVolume const * candvolume = vol;
	if( top ) candvolume = ( vol->UnplacedContains( point ) )? vol : 0;
	if( candvolume )
	{
		path.Push( candvolume );
		PhysicalVolume::DaughterContainer_t const * dlist = candvolume->GetDaughters();
		PhysicalVolume::DaughterContainerConstIterator_t iter;
		for(iter=dlist->begin(); iter!=dlist->end(); iter++ )
		{
			PhysicalVolume const * nextvol=(*iter);
			Vector3D transformedpoint;
			if( nextvol->Contains( point, transformedpoint ))
			{
				// this is no longer the top ( so setting top to false )
				localpoint = transformedpoint;
				candvolume = LocatePoint(nextvol, transformedpoint, localpoint, path, false);
				break;
			}
		}
	}
	return candvolume;

	// TODO: fill the path while going down
	// at the very end: do the matrix multiplications and caching of global matrices -- doing this in a loop will be icache friendly
}


PhysicalVolume const * SimpleVecNavigator::LocateLocalPointFromPath(
		Vector3D const & point, Vector3D & localpoint, VolumePath const & oldpath, VolumePath & newpath, TransformationMatrix * globalm ) const
{
 // very simple at the moment
	newpath.Clear();
	TransformationMatrix globalmatrix;
	oldpath.GetGlobalMatrixFromPath( &globalmatrix );
	Vector3D globalpoint;
	globalmatrix.LocalToMaster( point, globalpoint );
	localpoint = globalpoint;
	return LocatePoint( top, globalpoint, localpoint, newpath, globalm );
}



PhysicalVolume const * SimpleVecNavigator::LocateLocalPointFromPath_Relative(
		Vector3D const & point, Vector3D & localpoint, VolumePath & path, TransformationMatrix * globalm) const
{
	// idea: do the following:
	// ----- is localpoint still in current mother ? : then go down
	// if not: have to go up until we reach a volume that contains the localpoint and then go down again (neglecting the volumes currently stored in the path)
	PhysicalVolume const * currentmother = path.Top();
	if( currentmother != NULL )
	{
		localpoint = point;
		path.Pop();
		if( currentmother->UnplacedContains(point) )
		{
			// go further down -- can use LocatePoint function for this
			// at this moment would have to retrieve global matrix from path
			path.GetGlobalMatrixFromPath( globalm );
			globalm->Multiply( currentmother->getMatrix() );

			//std::cerr << "going further down " << std::endl;

			// return LocatePoint(currentmother, point, localpoint, path, globalm, false);
			return LocatePoint(currentmother, point, localpoint, path, globalm, false);
		}
		else
		{
			// convert localpoint to reference frame of mother

			Vector3D pointhigherup;
			currentmother->getMatrix()->LocalToMaster(point, pointhigherup);
			return LocateLocalPointFromPath_Relative(pointhigherup, localpoint, path, globalm);
		}
	}
	else
	{
		// particle is not even within world volume
		return 0;
	}
}

// this function tries to locate a point relative to a current geometry path
// the point is given in the reference frame of the last volume in that path
// this function should be called when we know that we are not entering a daughter
PhysicalVolume const * SimpleVecNavigator::LocateLocalPointFromPath_Relative_Iterative(
		Vector3D const & point,
		Vector3D & localpoint,
		VolumePath & path,
		TransformationMatrix * globalm) const
{
	// idea: do the following:
	// ----- is localpoint still in current mother ? : then go down
	// if not: have to go up until we reach a volume that contains the localpoint and then go down again (neglecting the volumes currently stored in the path)
	PhysicalVolume const * currentmother = path.Top();
	if( currentmother != NULL )
	{
        Vector3D tmp = point;
		// go up iteratively
		while( currentmother && ! currentmother->UnplacedContains( tmp ) )
		{
			path.Pop();
			Vector3D pointhigherup;
			currentmother->getMatrix()->LocalToMaster( tmp, pointhigherup );
			tmp=pointhigherup;
			currentmother=path.Top();
		}

		if(currentmother)
		{
			path.Pop();
			// then go down
			path.GetGlobalMatrixFromPath( globalm );
			globalm->Multiply( currentmother->getMatrix() );

			// may inline this
			return LocatePoint_iterative(currentmother, tmp, localpoint, path, globalm, false);
		}
	}
	return currentmother;
}


// this function tries to locate a point relative to a current geometry path
// the point is given in the reference frame of the last volume in that path
// this function should be called when we know that we are not entering a daughter
PhysicalVolume const * SimpleVecNavigator::LocateLocalPointFromPath_Relative_Iterative(
		Vector3DFast const & point,
		Vector3DFast & localpoint,
		VolumePath & path,
		FastTransformationMatrix * globalm) const
{
	// idea: do the following:
	// ----- is localpoint still in current mother ? : then go down
	// if not: have to go up until we reach a volume that contains the localpoint and then go down again (neglecting the volumes currently stored in the path)
	PhysicalVolume const * currentmother = path.Top();
	if( currentmother != NULL )
	{
        Vector3DFast tmp = point;
		// go up iteratively
		while( currentmother && ! currentmother->UnplacedContains( tmp ) )
		{
			path.Pop();
			Vector3DFast pointhigherup;
			currentmother->getFastMatrix()->LocalToMaster( tmp, pointhigherup );
			tmp=pointhigherup;
			currentmother=path.Top();
		}

		if(currentmother)
		{
			path.Pop();
			// then go down
			path.GetGlobalMatrixFromPath( globalm );
			globalm->Multiply<1,-1>( currentmother->getFastMatrix() );

			// may inline this
			return LocatePoint_iterative(currentmother, tmp, localpoint, path, globalm, false);
		}
	}
	return currentmother;
}



void SimpleVecNavigator::FindNextBoundaryAndStep(
		TransformationMatrix * globalm,
		Vector3D const & point,
		Vector3D const & dir,
		VolumePath const & inpath,
		VolumePath & outpath,
		Vector3D & newpoint,
		double &step ) const
{
  // question: in what coordinate system do we have the point? global or local
  // same for direction?
  Vector3D localpoint;
  globalm->MasterToLocal<1,-1>(point, localpoint);
  Vector3D localdir;
  globalm->MasterToLocalVec<-1>(dir, localdir);
  
  PhysicalVolume const * currentvolume = inpath.Top();
  int nexthitvolume = -1; // means mother
  
  step = currentvolume->DistanceToOut(localpoint, localdir, Utils::kInfinity);
  
  // iterate over all the daughters
  PhysicalVolume::DaughterContainer_t const * daughters = currentvolume->GetDaughters();
  PhysicalVolume::DaughterContainerConstIterator_t iter;
  int counter=0;
  for(iter = daughters->begin(); iter!=daughters->end(); ++iter)
    {
      double ddistance;
      PhysicalVolume const * daughter = (*iter);
      ddistance = daughter->DistanceToIn( localpoint, localdir, step );
      
      nexthitvolume = (ddistance < step) ? counter : nexthitvolume;
      step 	  = (ddistance < step) ? ddistance  : step;
      counter++;
    }
  
  // now we have the candidates
  // try
  outpath=inpath;
  Vector3D newpointafterboundary = localpoint + (step + Utils::frTolerance)*localdir;
  Vector3D newpointafterboundaryinnewframe;
  

  // this step is only necessary if nexthitvolume daughter or mother;
  // LocateLocalPointFromPath_Relative( newpointafterboundary, newpointafterboundaryinnewframe, outpath, globalm );
  // this step is not necessary ( this should be anyway: point + step*dir; )
  // ideas for improvement: improve

  if( nexthitvolume != -1 ) // not mother
  {
	  // continue directly further down
	  Vector3D tmp;
	  TransformationMatrix const *m;
	  PhysicalVolume const * nextvol = currentvolume->GetNthDaughter( nexthitvolume );
	  m = nextvol->getMatrix();
	  m->MasterToLocal<1,-1>( newpointafterboundary, tmp );
	  LocatePoint( nextvol, tmp, newpointafterboundaryinnewframe, outpath, globalm );
  }
  else
  {
	  // continue directly further up
	  LocateLocalPointFromPath_Relative( newpointafterboundary, newpointafterboundaryinnewframe, outpath, globalm );
  }

  globalm->LocalToMaster( newpointafterboundaryinnewframe, newpoint );
  //newpoint = point + ( step + Utils::frHalfTolerance )*localdir;

}


void SimpleVecNavigator::FindNextBoundaryAndStep_iterative(
		TransformationMatrix * globalm,
		Vector3D const & point,
		Vector3D const & dir,
		VolumePath const & inpath,
		VolumePath & outpath,
		Vector3D & newpoint,
		double &step ) const
{
  // question: in what coordinate system do we have the point? global or local
  // same for direction?

	// this information might have been cached in previous navigators??
	Vector3D localpoint;
	globalm->MasterToLocal<1,-1>(point, localpoint);
	Vector3D localdir;
	globalm->MasterToLocalVec<-1>(dir, localdir);

  PhysicalVolume const * currentvolume = inpath.Top();
  int nexthitvolume = -1; // means mother

  step = currentvolume->DistanceToOut(localpoint, localdir, Utils::kInfinity);

  // iterate over all the daughter
  PhysicalVolume::DaughterContainer_t const * daughters = currentvolume->GetDaughters();
  int counter=0;
  for(auto iter = daughters->begin(); iter!=daughters->end(); ++iter)
    {
      double ddistance;
      PhysicalVolume const * daughter = (*iter);
      // previous distance becomes step estimate, distance to daughter returned in workspace
      ddistance = daughter->DistanceToIn( localpoint, localdir, step );

      nexthitvolume = (ddistance < step) ? counter : nexthitvolume;
      step 	  = (ddistance < step) ? ddistance  : step;
      counter++;
    }

  // now we have the candidates
  // try
  outpath=inpath;
  Vector3D newpointafterboundary = localpoint + (step + Utils::frTolerance)*localdir;
  Vector3D newpointafterboundaryinnewframe;

  // this step is only necessary if nexthitvolume daughter or mother;
  // LocateLocalPointFromPath_Relative_Iterative( newpointafterboundary, newpointafterboundaryinnewframe, outpath, globalm );
  // globalm->LocalToMaster( newpointafterboundaryinnewframe, newpoint );


  if( nexthitvolume != -1 ) // not hitting mother
  {
	 // continue directly further down
	 // continue directly further down
	 Vector3D tmp;
	 TransformationMatrix const *m;
	 PhysicalVolume const * nextvol = currentvolume->GetNthDaughter( nexthitvolume );
	 m = nextvol->getMatrix();
	 m->MasterToLocal<1,-1>( newpointafterboundary, tmp );
	 globalm->Multiply( m );
	 LocatePoint_iterative( nextvol, tmp, newpointafterboundaryinnewframe, outpath, globalm );
  }
  else
  {
	 // continue directly further up
  	 LocateLocalPointFromPath_Relative_Iterative( newpointafterboundary, newpointafterboundaryinnewframe, outpath, globalm );
  }


  //globalm->LocalToMaster( newpointafterboundaryinnewframe, newpoint );
  newpoint = point + (step+Utils::frTolerance)*dir;
}



void SimpleVecNavigator::FindNextBoundaryAndStep_iterative(
		FastTransformationMatrix * globalm,
		Vector3DFast const & point,
		Vector3DFast const & dir,
		VolumePath const & inpath,
		VolumePath & outpath,
		Vector3DFast & newpoint,
		double &step ) const
{
  // question: in what coordinate system do we have the point? global or local
  // same for direction?

	// this information might have been cached in previous navigators??
	Vector3DFast localpoint;
	globalm->MasterToLocal<1,-1>(point, localpoint);
	Vector3DFast localdir;
	globalm->MasterToLocalVec<-1>(dir, localdir);

	PhysicalVolume const * currentvolume = inpath.Top();
	int nexthitvolume = -1; // means mother

	step = currentvolume->DistanceToOut(localpoint, localdir, Utils::kInfinity);

	// iterate over all the daughter
	PhysicalVolume::DaughterContainer_t const * daughters = currentvolume->GetDaughters();
	int counter=0;
	for(auto iter = daughters->begin(); iter!=daughters->end(); ++iter)
	{
      double ddistance;
      PhysicalVolume const * daughter = (*iter);
      // previous distance becomes step estimate, distance to daughter returned in workspace
      ddistance = daughter->DistanceToIn( localpoint, localdir, step );

      nexthitvolume = (ddistance < step) ? counter : nexthitvolume;
      step 	  = (ddistance < step) ? ddistance  : step;
      counter++;
    }

	// now we have the candidates
	// try
	outpath=inpath;
	Vector3DFast newpointafterboundary = localpoint + (step + Utils::frTolerance)*localdir;
	Vector3DFast newpointafterboundaryinnewframe;

	if( nexthitvolume != -1 ) // not hitting mother
	{
		// continue directly further down
		// continue directly further down
		Vector3DFast tmp;
		FastTransformationMatrix const *m;
		PhysicalVolume const * nextvol = currentvolume->GetNthDaughter( nexthitvolume );
		m = nextvol->getFastMatrix();
		m->MasterToLocal<1,-1>( newpointafterboundary, tmp );
		globalm->Multiply<1,-1>( m );
		LocatePoint_iterative( nextvol, tmp, newpointafterboundaryinnewframe, outpath, globalm );
	}
	else
	{
		// continue directly further up
		LocateLocalPointFromPath_Relative_Iterative( newpointafterboundary, newpointafterboundaryinnewframe, outpath, globalm );
	}

	//globalm->LocalToMaster( newpointafterboundaryinnewframe, newpoint );
	newpoint = point + (step+Utils::frTolerance)*dir;
}

