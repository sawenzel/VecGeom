/*
 * PhysicalVolume.h
 *
 *  Created on: Nov 8, 2013
 *      Author: swenzel
 */

#ifndef PHYSICALVOLUME_H_
#define PHYSICALVOLUME_H_

#include <vector>
#include "TransformationMatrix.h"
#include "LogicalVolume.h"
#include <iostream>
#include <vector>
#include "GlobalDefs.h"


//#include "TGeoShape.h"
class VUSolid;
class TGeoShape;

class BoxParameters;

// pure abstract class
class PhysicalVolume
{
public:
		typedef std::vector< PhysicalVolume const *> DaughterContainer_t;
		typedef DaughterContainer_t::iterator DaughterContainerIterator_t;
		typedef DaughterContainer_t::const_iterator DaughterContainerConstIterator_t;

protected:
		PhysicalVolume * bbox;
		// this is the bounding box, it is placed
		// only translation necessary with respect to the local coordinate system of the PhysicalVolume
		// this is a bit nasty because we need to cast bbox to PlacedBox<1,1296>

		TransformationMatrix const *matrix; // placement matrix with respect to containing volume

		// just for comparision, we also keep a fastmatrix
		FastTransformationMatrix *fastmatrix;

		// something like a logical volume id

		// I am not sure that this is appropriate
		// maybe we should do PhysicalShape and PhysicalVolume as different classes
		LogicalVolume  *logicalvol;

		// crucial thing: we keep a POINTER to a list of volumes ( and not a list )
		// is this a protected member ??? NO!!
		DaughterContainer_t * daughters;

		bool ExclusiveContains( Vector3D const & ) const;

		//** this is for benchmarking and conversion purposes **//
		VUSolid * analogoususolid;
		TGeoShape * analogousrootsolid;

		// setting the daughter list
		void SetDaughterList( DaughterContainer_t * l)
		{
			if( daughters->size() > 0 )
			{
				std::cerr << " trying to set a new daughterlist while old one is already set " << std::endl;
				std::cerr << " THIS IS LIKELY AN ERROR AND I AM REFUSING " << std::endl;
			}
			else
			{
				daughters= l;
			}
		}

	public:
		PhysicalVolume( TransformationMatrix const *m ) : matrix(m),
			logicalvol(0), daughters(new DaughterContainer_t), bbox(0), analogoususolid(0), analogousrootsolid(0),
			fastmatrix(new FastTransformationMatrix())
			{
				fastmatrix->SetTrans( &matrix->trans[0] );
				fastmatrix->SetRotation( &matrix->rot[0] );
			};

		virtual double DistanceToIn( Vector3D const &, Vector3D const &, double ) const = 0;
		virtual double DistanceToOut( Vector3D const &, Vector3D const &, double ) const = 0;
		virtual bool   Contains( Vector3D const & ) const = 0;
		virtual bool   UnplacedContains( Vector3D const & ) const = 0;

		// same as Contains but returning the transformed point for further processing
		// this function is a "specific" version for locating points in a volume hierarchy
		// it also modifies the global matrix
		virtual bool   Contains( Vector3D const &, Vector3D & ) const { return false; }

		// this version modifies the global matrix additionally
		virtual bool   Contains( Vector3D const &, Vector3D &, TransformationMatrix * ) const { return false; }

		//** ------------------------------------------------------------------------------------------------------------------------------**//
		//*****  the following methods are introduced temporirly to study speed/ influence of the new fast ThreeVector (Vector3DFast) *****//
		virtual double DistanceToIn( Vector3DFast const &, Vector3DFast const &, double /*step*/ ) const {return 0.;}
		virtual double DistanceToOut( Vector3DFast const &, Vector3DFast const &, double /*step*/ ) const {return 0.;}
		virtual double DistanceToInAndSafety( Vector3DFast const &, Vector3DFast const &, double /*step*/, double & ) const {return 0.;}
		virtual double DistanceToOutAndSafety( Vector3DFast const &, Vector3DFast const &, double /*step*/, double & ) const {return 0.;}

		virtual bool   Contains( Vector3DFast const & ) const {return 0.;}
		virtual bool   UnplacedContains( Vector3DFast const & ) const {return 0.;}

		// same as Contains but returning the transformed point for further processing
		// this function is a "specific" version for locating points in a volume hierarchy
		// it also modifies the global matrix
		virtual bool   Contains( Vector3DFast const &, Vector3DFast & ) const { return false; }
		// this version modifies the global matrix additionally
		virtual bool   Contains( Vector3DFast const &, Vector3DFast &, FastTransformationMatrix * ) const { return false; }
		// ** -------------------------------------------------------------------------------------------------------------------------------//

		// the contains function knowing about the surface
		virtual GlobalTypes::SurfaceEnumType   UnplacedContains_WithSurface( Vector3D const & ) const { return GlobalTypes::kOutside; };
		virtual GlobalTypes::SurfaceEnumType   Contains_WithSurface( Vector3D const & ) const { return GlobalTypes::kOutside; };

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
				daughters = new DaughterContainer_t;
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

		// method which places a placed(or unplaced) box inside this volume
		// this method is going to analyse the box and matrix combination ( using the existing
		// shape factory functionality to add an appropriate specialised list of
		// TODO: ideally we should give in an "abstract placed box" and not boxparameters
		PhysicalVolume const * PlaceDaughter( PhysicalVolume * newdaughter, DaughterContainer_t const * const d = 0)
		{
			if( ! daughters ){
				daughters = new DaughterContainer_t;
			}
			if( daughters ){
				// does not compile here; cyclic dependency
				// PhysicalVolume * newdaughter = GeoManager::MakePlacedBox(b,m);
				newdaughter->SetDaughterList( const_cast< DaughterContainer_t *> (d) );
				daughters->push_back( newdaughter );
			}
			else{
				std::cerr << "WARNING: no daughter list found" << std::endl;
			}
			return newdaughter;
		}

		inline
		int GetNumberOfDaughters() const
		{
			if( daughters )
			{
				return daughters->size();
			}
			return 0;
		}

		inline
		DaughterContainer_t const * GetDaughters() const
		{
			return daughters;
		}

		inline
		PhysicalVolume const * GetNthDaughter( int n ) const
		{
			// will not work with lists
			return daughters->operator[](n);
		}

		// this function fills the physical volume with random points and directions such that the points are
		// contained within the volume but not within the daughters
		// it returns the points in points and the directions in dirs
		// this will be in the local reference frame
		void fillWithRandomPoints( Vectors3DSOA & points, int number ) const;

		// random directions
		static
		void fillWithRandomDirections( Vectors3DSOA & dirs, int number );

		static
		void samplePoint( Vector3D & point, double dx, double dy, double dz, double scale );

		// give random directions satisfying the constraint that fraction of them hits a daughter boundary
		// needs the positions as inputs
		// we consider fraction as the LOWER BOUND
		void fillWithBiasedDirections( Vectors3DSOA const & points, Vectors3DSOA & dirs, int number, double fraction ) const;

		// to access information about Matrix
		TransformationMatrix const * getMatrix() const {return matrix;}
		FastTransformationMatrix const * getFastMatrix() const {return fastmatrix;}

		LogicalVolume const * getLogicalVolume() const { return logicalvol; }

		// for debugging purposes ( can be overridden by concrete shapes )
		virtual
		void
		DebugPointAndDirDistanceToIn( Vector3D const & x, Vector3D const & y ) const
		{}

};


#endif /* PHYSICALVOLUME_H_ */
