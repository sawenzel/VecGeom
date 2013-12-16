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
#include <iostream>
#include <vector>


//#include "TGeoShape.h"
class VUSolid;
class TGeoShape;

// pure abstract class
class PhysicalVolume
{
	protected:
		PhysicalVolume * bbox;
		// this is the bounding box, it is placed
		// only translation necessary with respect to the local coordinate system of the PhysicalVolume
		// this is a bit nasty because we need to cast bbox to PlacedBox<1,0>

		TransformationMatrix const *matrix; // placement matrix with respect to containing volume
		//std::list<PhysicalVolume> daughterVolumes; // list or vector?

		// something like a logical volume id

		// I am not sure that this is appropriate
		// maybe we should do PhysicalShape and PhysicalVolume as different classes
		LogicalVolume  *logicalvol;

		// crucial thing: we keep a POINTER to a list of volumes ( and not a list )
		// is this a protected member ??? NO!!
		std::list<PhysicalVolume const *> * daughters;

		bool ExclusiveContains( Vector3D const & ) const;

		//** this is for benchmarking and conversion purposes **//
		VUSolid * analogoususolid;
		TGeoShape * analogousrootsolid;

	public:
		PhysicalVolume( TransformationMatrix const *m ) : matrix(m),
			logicalvol(0), daughters(new std::list<PhysicalVolume const *>), bbox(0), analogoususolid(0), analogousrootsolid(0) {};



		virtual double DistanceToIn( Vector3D const &, Vector3D const &, double ) const = 0;
		virtual double DistanceToOut( Vector3D const &, Vector3D const &, double ) const = 0;
		virtual bool   Contains( Vector3D const & ) const = 0;
		virtual bool   UnplacedContains( Vector3D const & ) const = 0;
	//	virtual GlobalTypes::SurfaceEnumType   UnplacedContains( Vector3D const & ) const {};
		virtual double SafetyToIn( Vector3D const & ) const = 0;
		virtual double SafetyToOut( Vector3D const & ) const = 0;

		// for basket treatment (supposed to be dispatched to particle parallel case)
		virtual void DistanceToIn( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const = 0;
		virtual void DistanceToOut( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const {};


		// for basket treatment (supposed to be dispatched to loop case over (SIMD) optimized 1-particle function)
		virtual void DistanceToInIL( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const = 0;
		//	virtual void DistanceToInIL( std::vector<Vector3D> const &, std::vector<Vector3D> const &, double const * /*steps*/, double * /*result*/ ) const;
		virtual void DistanceToInIL( Vector3D const *, Vector3D const *, double const * /*steps*/, double * /*result*/, int /*size*/ ) const =0;

		VUSolid const * GetAsUnplacedUSolid() const
		{
			return analogoususolid;
		}
		void SetUnplacedUSolid( VUSolid *  solid ) { analogoususolid = solid ;}
		TGeoShape const * GetAsUnplacedROOTSolid() const
		{
			return analogousrootsolid;
		}


		// providing an interface to an UNPLACED version of this shape
		// this should return a pointer to a shape with identity transformation = stripping any matrix information from it
		virtual PhysicalVolume const * GetAsUnplacedVolume() const = 0;
		void printInfo() const;

		void PrintDistToEachDaughter( Vector3D const &, Vector3D const &) const;
		void PrintDistToEachDaughterROOT( Vector3D const &, Vector3D const &) const;
		void PrintDistToEachDaughterUSOLID( Vector3D const &, Vector3D const &) const;
		// this is
		// virtual void DistanceToOut(Vectors3DSOA const &, Vectors3DSOA const &, double , int, double * /*result*/) const = 0;
		// delete all explicit constructors etc...
		//

		virtual ~PhysicalVolume( ){ };

		void AddDaughter( PhysicalVolume const * vol )
		{
			// this is a delicate issue
			// since the daughter list will be shared between multiple placed volumes
			// we should have a reference counter on the daughters list

			// if reference > 1 we should either refuse addition or at least issue a warning

			if( ! daughters )
			{
				daughters = new std::list<PhysicalVolume const *>;
			}
			if( daughters )
			{
				daughters->push_back(vol);
			}
			else
			{
				std::cerr << "WARNING: no daughter list found" << std::endl;
			}
		}

		int GetNumberOfDaughters() const
		{
			if( daughters )
			{
				return daughters->size();
			}
			return 0;
		}

		std::list<PhysicalVolume const *> const * GetDaughterList() const
		{
			return daughters;
		}

		// this function fills the physical volume with random points and directions such that the points are
		// contained within the volume but not within the daughters
		// it returns the points in points and the directions in dirs
		// this will be in the local reference frame
		void fillWithRandomPoints( Vectors3DSOA & points, int number ) const;

		// random directions
		static
		void fillWithRandomDirections( Vectors3DSOA & dirs, int number );

		// give random directions satisfying the constraint that fraction of them hits a daughter boundary
		// needs the positions as inputs
		// we consider fraction as the LOWER BOUND
		void fillWithBiasedDirections( Vectors3DSOA const & points, Vectors3DSOA & dirs, int number, double fraction ) const;

		// to access information about Matrix
		TransformationMatrix const * getMatrix() const {return matrix;}

		LogicalVolume const * getLogicalVolume() const { return logicalvol; }

		// for debugging purposes ( can be overridden by concrete shapes )
		virtual
		void
		DebugPointAndDirDistanceToIn( Vector3D const & x, Vector3D const & y ) const
		{}

};


#endif /* PHYSICALVOLUME_H_ */
