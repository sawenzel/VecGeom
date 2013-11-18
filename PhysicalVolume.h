/*
 * PhysicalVolume.h
 *
 *  Created on: Nov 8, 2013
 *      Author: swenzel
 */

#ifndef PHYSICALVOLUME_H_
#define PHYSICALVOLUME_H_

#include <list>
#include "TransformationMatrix.h"
#include "LogicalVolume.h"

// pure abstract class
class PhysicalVolume
{
	private:
		PlacedBox<1,0> * bbox; // this is the bounding box, it is placed
			// only translation necessary with respect to the local coordinate system of the PhysicalVolume

	protected:
		TransformationMatrix const *matrix; // placement matrix with respect to containing volume
		//std::list<PhysicalVolume> daughterVolumes; // list or vector?

		// something like a logical volume id

		// I am not sure that this is appropriate
		// maybe we should do PhysicalShape and PhysicalVolume as different classes
		LogicalVolume  *logicalvol;

		// crucial thing: we keep a POINTER to a list of volumes ( and not a list )
		std::list<PhysicalVolume const *> * daughters;

		bool ExclusiveContains( Vector3D const & ) const;


	public:
		PhysicalVolume( TransformationMatrix const *m ) : matrix(m), logicalvol(0), daughters(0) { };
		virtual double DistanceToIn( Vector3D const &, Vector3D const &, double ) const = 0;
		virtual double DistanceToOut( Vector3D const &, Vector3D const &, double ) const = 0;
		virtual bool   Contains( Vector3D const & ) const = 0;
		virtual bool   UnplacedContains( Vector3D const & ) const = 0;

		// for basket treatment
		virtual void DistanceToIn( Vectors3DSOA const &, Vectors3DSOA const &, double, double * /*result*/ ) const = 0;

		// this is
		// virtual void DistanceToOut(Vectors3DSOA const &, Vectors3DSOA const &, double , int, double * /*result*/) const = 0;
		// delete all explicit constructors etc...
		//
		virtual ~PhysicalVolume( ){ };

		void AddDaughter( PhysicalVolume const * vol ){ daughters->push_back(vol); }

		// this function fills the physical volume with random points and directions such that the points are
		// contained within the volume but not within the daughters
		// it returns the points in points and the directions in dirs
		void fillWithRandomPoints( Vectors3DSOA &points, Vectors3DSOA & dirs, int number ) const;



		LogicalVolume const * getLogicalVolume(){ return logicalvol; }
};


#endif /* PHYSICALVOLUME_H_ */
