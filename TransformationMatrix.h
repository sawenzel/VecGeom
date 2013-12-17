/*
 * TransformationMatrix.h
 *
 *  Created on: Nov 8, 2013
 *      Author: swenzel
 */

#ifndef TRANSFORMATIONMATRIX_H_
#define TRANSFORMATIONMATRIX_H_

#include "Vector3D.h"
#include <cmath>
#include <type_traits>
#include "Vc/vector.h"

#include "TGeoMatrix.h"

typedef int TranslationIdType;
typedef int RotationIdType;

class TGeoMatrix;

// the idea is to have StorageClass being a
//template <typename StorageClass>
class TransformationMatrix
{
private:
    // this can be varied depending on template specialization
	double trans[3];
	double rot[9];
	bool identity;
	bool hasRotation;
	bool hasTranslation;

	static const double kDegToRad;
	static const double kRadToDeg;

	// the equivalent ROOT matrix (for convenience)
	TGeoMatrix * rootmatrix;

	template<RotationIdType rid>
	inline
	void
	emitrotationcode(Vector3D const &, Vector3D &) const;

// the idea is to provide one general engine which works for both scale ( T = double ) as well as vector backends ( T = Vc::double_v )
	template<RotationIdType rid, typename T>
	inline
	void
	emitrotationcodeT( T const & mx, T const & my, T const & mz, T & lx, T & ly, T & lz ) const;

	template<RotationIdType rid, typename T>
	static
	inline
	void
	  emitrotationcodeT( T const & mx, T const & my, T const & mz, T & lx, T & ly, T & lz, double const *rot );

	void SetTrans(const Vector3D /*trans_*/);
	void SetTrans(const double tx, const double ty, const double tz);
	void SetAngles(const Vector3D /*euler*/);
	void SetAngles(const double phi, const double theta, const double psi);
	void SetProperties();
	void InitEquivalentTGeoMatrix(double, double, double);
	void InitEquivalentTGeoMatrix();

public:
	bool isIdentity() const {return identity;}
	bool isRotation() const {return hasRotation;}
	bool isTranslation() const {return hasTranslation;}
	void print() const;

	TGeoMatrix const * GetAsTGeoMatrix() const { return rootmatrix;}

	Vector3D getTrans() const
	{
		return Vector3D(trans[0], trans[1], trans[2]);
	}

	Vector3D GetEulerAngles() const;

	// a factory method
	static
	TransformationMatrix const *
	createSpecializedMatrix( double, double, double, double, double, double );


	// a static matrix classification method
	static
	void
	classifyMatrix( double, double, double, double, double, double, int &, int &);

	static
	inline
	RotationIdType
	getRotationFootprintS(double const *r){
		int footprint=0;
		// count zero entries and give back footprint which classifies them
		for(int i=0;i<9;i++)
		{
			if( std::abs(r[i]) < 1E-12 )
				footprint+=i*i*i; // cubic power identifies cases uniquely
		}

		// that's a diagonal matrix.
		// have to check if this is trivial case
		if( footprint == 720 )
		{
			if( r[0]==1. && r[4]==1. && r[8]== 1. )
			{
				return 1296; // this will be trivial rotation == no rotation
			}
		}
		return footprint;
	}

	// same as a member
	inline
	RotationIdType
	getRotationFootprint() const
	{
		return getRotationFootprintS(this->rot);
	}

	unsigned int getNumberOfZeroEntries() const;

	// constructor
	TransformationMatrix(double const *t, double const *r);

	// more general constructor ala ROOT ( with Euler Angles )
	TransformationMatrix(double tx, double ty, double tz, double phi, double theta, double psi);

	virtual
	~TransformationMatrix();

	TransformationMatrix(TransformationMatrix const &other);

	inline
	static TranslationIdType GetTranslationIdTypeS(double const *t)
	{
		if( t[0]==0 && t[1]==0 && t[2]==0 )
			return 0;
		else
			return 1;
	}

	inline
	TranslationIdType GetTranslationIdType() const
	{
		return GetTranslationIdTypeS(trans);

	}


	template <TranslationIdType tid, RotationIdType rid>
	inline
	void
	MasterToLocal(Vector3D const &, Vector3D &) const;


	// T is the internal type to be used ( can be scalar or vector )
	// these interfaces are for the vector treatments
	template <TranslationIdType tid, RotationIdType rid, typename T>
		inline
		void
		MasterToLocal(Vectors3DSOA const &, Vectors3DSOA &) const;
	template <TranslationIdType tid, RotationIdType rid, typename T>
		inline
		void
		__attribute__((always_inline))
		MasterToLocalVec(Vectors3DSOA const &, Vectors3DSOA &) const;
	template <TranslationIdType tid, RotationIdType rid, typename T>
		inline
		void
		MasterToLocalCombinedT(Vectors3DSOA const &, Vectors3DSOA &, Vectors3DSOA const &, Vectors3DSOA &) const;


	/*
	 *  we should provide the versions for a Vc vector or more general for a T vector
	 */
	template<TranslationIdType tid, RotationIdType rid, typename T>
		inline
		void
		MasterToLocal(T const & masterx, T const & mastery, T const & masterz, T  & localx , T & localy , T & localz ) const;

	template<TranslationIdType tid, RotationIdType rid, typename T>
		inline
		void
		__attribute__((always_inline))
		MasterToLocalVec(T const & masterx, T const & mastery, T const & masterz, T  & localx , T & localy , T & localz ) const
		{
			MasterToLocal<0,rid,T>(masterx, mastery,masterz,localx,localy,localz);
		}


	// to transform real vectors, we don't care about translation
	template <RotationIdType rid>
	inline
	void
	__attribute__((always_inline))
	MasterToLocalVec(Vector3D const &, Vector3D &) const;


	// define some virtual functions (which should dispatch to specialized templated functions)
	virtual
		void
		MasterToLocal(Vector3D const & master, Vector3D & local) const {
		// inline general case here
		MasterToLocal<-1,-1>(master,local);
	};

	virtual
		void
		MasterToLocalVec(Vector3D const & master, Vector3D & local) const {
		MasterToLocalVec<-1>(master,local);
	};


	virtual
	void
	MasterToLocal(Vectors3DSOA const & master, Vectors3DSOA & local) const
	{
// this is not nice: we bind ourselfs to Vc here
		MasterToLocal<-1,-1, Vc::double_v>(master, local);
	}

	virtual
	void
	MasterToLocalVec(Vectors3DSOA const & master, Vectors3DSOA & local) const
	{
// this is not nice: we bind ourselfs to Vc here
		MasterToLocal<0,-1, Vc::double_v>(master, local);
	}

	virtual
	void
	MasterToLocalCombined( Vectors3DSOA const & masterpoint, Vectors3DSOA & localpoint,
			Vectors3DSOA const & mastervec, Vectors3DSOA & localvec ) const
	{
		//mapping v
		MasterToLocalCombinedT<1,-1,Vc::double_v>( masterpoint, localpoint, mastervec, localvec );
	}

	friend class PhysicalVolume;
	friend class GeoManager;
};

template <TranslationIdType tid, RotationIdType rid, typename T>
inline
void
__attribute__((always_inline))
TransformationMatrix::MasterToLocal(T const & masterx, T const & mastery, T const & masterz, T  & localx , T & localy , T & localz ) const
{
  if( tid==0 && rid == 1296 ) // this means identity
  {
	  localx = masterx;
	  localy = mastery;
	  localz = masterz;
  }
 else if( tid == 1 && rid == 1296 ) // tid == 1 means we have only translation
  {
	  localx = masterx - trans[0];
	  localy = mastery - trans[1];
	  localz = masterz - trans[2];
  }
 else if( tid == 0 && rid!=0 ) // pure rotation
  {
	  emitrotationcodeT<rid,T>(masterx, mastery, masterz, localx, localy, localz);
  }
 else if ( tid == 1 && rid!=0 ) // both rotation and translation
 {
	  //	 T mtx, mty, mtz;
	  //	 mtx = masterx - trans[0];
	  // mty=  mastery - trans[1];
	  // mtz = masterz - trans[2];
	 emitrotationcodeT<rid,T>(masterx-trans[0], mastery-trans[1], masterz-trans[2], localx, localy, localz);
 }
}

template <TranslationIdType tid, RotationIdType rid>
inline
void
__attribute__((always_inline))
TransformationMatrix::MasterToLocal(Vector3D const & master, Vector3D & local) const
{
	MasterToLocal<tid, rid, double>(master.x, master.y, master.z, local.x, local.y, local.z );
}


template <RotationIdType rid>
void
TransformationMatrix::MasterToLocalVec(Vector3D const & master, Vector3D & local ) const
{
	MasterToLocal<0, rid>(master, local);
}

template <RotationIdType rid, typename T>
inline
void
TransformationMatrix::emitrotationcodeT( T const & mtx, T const & mty, T const & mtz, T & localx, T & localy, T & localz ) const
{
  	if(rid==252){
	     localx=mtx*rot[0];
	     localy=mty*rot[4]+mtz*rot[7];
	     localz=mty*rot[5]+mtz*rot[8];
	     return;
	}
	if(rid==405){
	     localx=mty*rot[3];
	     localy=mtx*rot[1]+mtz*rot[7];
	     localz=mtx*rot[2]+mtz*rot[8];
	     return;
	}
	if(rid==882){
	     localx=mtz*rot[6];
	     localy=mtx*rot[1]+mty*rot[4];
	     localz=mtx*rot[2]+mty*rot[5];
	     return;
	}
	if(rid==415){
	     localx=mty*rot[3]+mtz*rot[6];
	     localy=mtx*rot[1];
	     localz=mty*rot[5]+mtz*rot[8];
	     return;
	}
	if(rid==496){
	     localx=mtx*rot[0]+mtz*rot[6];
	     localy=mty*rot[4];
	     localz=mtx*rot[2]+mtz*rot[8];
	     return;
	}
	if(rid==793){
	     localx=mtx*rot[0]+mty*rot[3];
	     localy=mtz*rot[7];
	     localz=mtx*rot[2]+mty*rot[5];
	     return;
	}
	if(rid==638){
	     localx=mty*rot[3]+mtz*rot[6];
	     localy=mty*rot[4]+mtz*rot[7];
	     localz=mtx*rot[2];
	     return;
	}
	if(rid==611){
	     localx=mtx*rot[0]+mtz*rot[6];
	     localy=mtx*rot[1]+mtz*rot[7];
	     localz=mty*rot[5];
	     return;
	}
	if(rid==692){
	     localx=mtx*rot[0]+mty*rot[3];
	     localy=mtx*rot[1]+mty*rot[4];
	     localz=mtz*rot[8];
	     return;
	}
	if(rid==720){
	     localx=mtx*rot[0];
	     localy=mty*rot[4];
	     localz=mtz*rot[8];
	     return;
	}
	if(rid==828){
	     localx=mtx*rot[0];
	     localy=mtz*rot[7];
	     localz=mty*rot[5];
	     return;
	}
	if(rid==756){
	     localx=mty*rot[3];
	     localy=mtx*rot[1];
	     localz=mtz*rot[8];
	     return;
	}
	if(rid==918){
	     localx=mty*rot[3];
	     localy=mtz*rot[7];
	     localz=mtx*rot[2];
	     return;
	}
	if(rid==954){
	     localx=mtz*rot[6];
	     localy=mtx*rot[1];
	     localz=mty*rot[5];
	     return;
	}
	if(rid==1008){
	     localx=mtz*rot[6];
	     localy=mty*rot[4];
	     localz=mtx*rot[2];
	     return;
	}
	if(rid==1296){
		localx=mtx;
		localy=mty;
		localz=mtz;
		return;
	}
 // localx=mtx*rot[0]+mty*rot[3]+mtz*rot[6];
  //localy=mtx*rot[1]+mty*rot[4]+mtz*rot[7];
  //localz=mtx*rot[2]+mty*rot[5]+mtz*rot[8];
	// this is better for inlining purposes
	localx=mtx*rot[0];
	localy=mtx*rot[1];
	localz=mtx*rot[2];
	localx+=mty*rot[3];
	localy+=mty*rot[4];
	localz+=mty*rot[5];
	localx+=mtz*rot[6];
	localy+=mtz*rot[7];
	localz+=mtz*rot[8];
	return;
}



template <RotationIdType rid>
inline
void
TransformationMatrix::emitrotationcode(Vector3D const & mt, Vector3D & local) const
{
  emitrotationcode<rid, double>( mt.x, mt.y, mt.z, local.x, local.y, local.z );
}


template <TranslationIdType tid, RotationIdType rid, typename T >
inline
void
TransformationMatrix::MasterToLocal(Vectors3DSOA const & master_v, Vectors3DSOA & local_v) const
{
	// here we are getting a vector of points in SOA form and need to return a vector of points in SOA form
	// this code is specific to Vc but we could use type traits (is_same or something)
	for( int i=0; i < master_v.size ; i += T::Size )
	{
		T x( &master_v.x[i] );
		T y( &master_v.y[i] );
		T z( &master_v.z[i] );
		T lx, ly, lz;
		MasterToLocal<tid,rid,T>(x,y,z,lx,ly,lz);
		// store back result
		lx.store( &local_v.x[i] );
		ly.store( &local_v.y[i] );
		lz.store( &local_v.z[i] );
	}
	// need to treat tail part still
}

/*
template <TranslationIdType tid, RotationIdType rid, typename T>
inline
void
  TransformationMatrix;<tid,rid>::MasterToLocalVec(Vectors3DSOA const & master_v, Vector3DSOA & local_v) const
{
	MasterToLocal<0,tid,T>(master_v, local_v);
}
*/

template <TranslationIdType tid, RotationIdType rid, typename T>
inline
void
TransformationMatrix::MasterToLocalCombinedT( Vectors3DSOA const & master_v, Vectors3DSOA & local_v,
		Vectors3DSOA const & masterd_v, Vectors3DSOA & locald_v ) const
{
	// here we are getting a vector of points in SOA form and need to return a vector of points in SOA form
	// this code is specific to Vc but we could use type traits (is_same or something)
	for( int i=0; i < master_v.size; i += T::Size )
	{
		T x( &master_v.x[i] );
		T y( &master_v.y[i] );
		T z( &master_v.z[i] );
		T lx, ly, lz;
		MasterToLocal<tid,rid,T>(x,y,z,lx,ly,lz);
		// store back result
		lx.store( &local_v.x[i] );
		ly.store( &local_v.y[i] );
		lz.store( &local_v.z[i] );
		T xd( &masterd_v.x[i] );
		T yd( &masterd_v.y[i] );
		T zd( &masterd_v.z[i] );
		T lxd, lyd, lzd;
		MasterToLocal<0,rid,T>(xd,yd,zd,lxd,lyd,lzd);
		// store back result
		lxd.store( &locald_v.x[i] );
		lyd.store( &locald_v.y[i] );
		lzd.store( &locald_v.z[i] );
	}
	// need to treat tail part still
}


// -----------------------------------------------------------------------------------------------------------------------------------------------------------------

// now we define subclasses to TransformationMatrix as specialized classes
template <int tid, int rid>
class SpecializedTransformation : public TransformationMatrix
{
public:
	SpecializedTransformation(double tx, double ty, double tz, double phi, double theta, double psi) : TransformationMatrix(tx, ty, tz, phi, theta, psi) {};

	// check this more carefully
	virtual
	~SpecializedTransformation(){};

	virtual
	void
	MasterToLocal(Vector3D const &, Vector3D &) const;

	virtual
	void
	MasterToLocalVec(Vector3D const &, Vector3D &) const;

	virtual
	void
	MasterToLocal(Vectors3DSOA const &, Vectors3DSOA & ) const;

	virtual
	void
	MasterToLocalVec(Vectors3DSOA const &, Vectors3DSOA & ) const;

	virtual
	void
	MasterToLocalCombined(Vectors3DSOA const &, Vectors3DSOA &, Vectors3DSOA const &, Vectors3DSOA &) const;

};


template <int tid, int rid>
void
SpecializedTransformation<tid,rid>::MasterToLocal(Vector3D const & master, Vector3D & local) const
{
	TransformationMatrix::MasterToLocal<tid,rid>(master,local);
}

template <int tid, int rid>
void
SpecializedTransformation<tid,rid>::MasterToLocalVec(Vector3D const & master, Vector3D & local) const
{
	TransformationMatrix::MasterToLocalVec<rid>(master,local);
}


template <int tid, int rid>
void
SpecializedTransformation<tid,rid>::MasterToLocal(Vectors3DSOA const & master, Vectors3DSOA & local) const
{
	TransformationMatrix::MasterToLocal<tid, rid, Vc::double_v >(master,local);
}


template <int tid, int rid>
void
SpecializedTransformation<tid,rid>::MasterToLocalVec(Vectors3DSOA const & master, Vectors3DSOA & local) const
{
	TransformationMatrix::MasterToLocal<0, rid, Vc::double_v >(master,local);
}

template <int tid, int rid>
void
SpecializedTransformation<tid,rid>::MasterToLocalCombined(Vectors3DSOA const & masterp, Vectors3DSOA & localp, Vectors3DSOA const & masterd, Vectors3DSOA & locald) const
{
	TransformationMatrix::MasterToLocalCombinedT<tid,rid, Vc::double_v>( masterp, localp, masterd, locald);
}

#endif /* TRANSFORMATIONMATRIX_H_ */
