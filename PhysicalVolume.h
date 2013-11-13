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
	protected:

		TransformationMatrix const *matrix; // placement matrix with respect to containing volume
		//std::list<PhysicalVolume> daughterVolumes; // list or vector?

		// something like a logical volume id

		// I am not sure that this is appropriate
		// maybe we should do PhysicalShape and PhysicalVolume as different classes
		LogicalVolume  *logicalvol;


	public:
		PhysicalVolume(TransformationMatrix const *m) : matrix(m), logicalvol(0) {};


		virtual double DistanceToIn(Vector3D const &, Vector3D const &, double ) const = 0;
		virtual double DistanceToOut(Vector3D const &, Vector3D const &, double ) const = 0;


		// this is
		virtual void DistanceToOut(Vectors3DSOA const &, Vectors3DSOA const &, double , int, double * /*result*/) const = 0;
		// delete all explicit constructors etc...
		
		
		//
		virtual ~PhysicalVolume(){};

		// add factory methods
		void AddDaughter( PhysicalVolume const * vol ){ } ;
		LogicalVolume const * getLogicalVolume(){return logicalvol;}
};


#endif /* PHYSICALVOLUME_H_ */
