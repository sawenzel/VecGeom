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
#include <cstring>

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

	void setAngles(double phi, double theta, double psi);
	void setProperties();
	void InitEquivalentTGeoMatrix(double, double, double);

public:
	void SetToIdentity();
	bool isIdentity() const {return identity;}
	bool isRotation() const {return hasRotation;}
	bool isTranslation() const {return hasTranslation;}
	void print() const;

	TGeoMatrix const * GetAsTGeoMatrix() const { return rootmatrix;}

	Vector3D getTrans() const
	{
		return Vector3D(trans[0], trans[1], trans[2]);
	}

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

	// will create identity matrix
	TransformationMatrix();

	// more general constructor ala ROOT ( with Euler Angles )
	TransformationMatrix(double tx, double ty, double tz, double phi, double theta, double psi);

	virtual
	~TransformationMatrix(){}

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
		MasterToLocal<1,-1>(master,local);
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
		MasterToLocal<1,-1, Vc::double_v>(master, local);
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

    // ---------------------------------------------
	// for the other way around: LocalToMaster

	inline
	void
	__attribute__((always_inline))
	LocalToMaster(Vector3D const &, Vector3D &) const;



	/*
	 *  we should provide the versions for a Vc vector or more general for a T vector
	 */
	template<TranslationIdType tid, typename T>
	inline
	void
	LocalToMaster(T const & masterx, T const & mastery, T const & masterz, T  & localx, T & localy, T & localz) const;

	template<typename T>
	inline
	void
	__attribute__((always_inline))
	LocalToMasterVec(T const & masterx, T const & mastery, T const & masterz, T  & localx, T & localy, T & localz) const
	{
		LocalToMaster<0,T>(masterx,mastery,masterz,localx,localy,localz);
	}

	// to transform real vectors, we don't care about translation
	inline
	void
	__attribute__((always_inline))
	LocalToMasterVec(Vector3D const & local, Vector3D & master) const
	{
		LocalToMasterVec<double>(local.x,local.y,local.z,master.x,master.y,master.z);
	}
	// END LOCAL TO MASTER


	inline
	void
	// multiplication of a matrix from the right; storing of result in left-hand matrix
	Multiply( TransformationMatrix const * rhs );

	// multiplication of a right matrix where we template specialize on the properties of the right hand matrix
	template<TranslationIdType tid, RotationIdType rid>
	inline
	void
	__attribute__((always_inline))
	MultiplyT( TransformationMatrix const * rhs );


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


inline
void
TransformationMatrix::LocalToMaster(Vector3D const & local, Vector3D & master) const
{
	// this is treated totally unspecialized for the moment
	LocalToMaster<1, double>( local.x, local.y, local.z, master.x, master.y, master.z );
}

template<TranslationIdType tid, typename T>
inline
__attribute__((always_inline))
void
TransformationMatrix::LocalToMaster(T const & local_x, T const & local_y, T const & local_z, T & master_x, T & master_y, T & master_z ) const
{
	// we are just doing the full stuff here ( LocalToMaster is less critical than other way round )
	if(tid == 0)
	{
		master_x = local_x*rot[0];
		master_x += local_y*rot[1];
		master_x += local_z*rot[2];
		master_y = local_x*rot[3];
		master_y += local_y*rot[4];
		master_y += local_z*rot[5];
		master_z = local_x*rot[6];
		master_z += local_y*rot[7];
		master_z += local_z*rot[8];
	}
	else
	{
		master_x = trans[0];
		master_x += local_x*rot[0];
		master_x += local_y*rot[1];
		master_x += local_z*rot[2];
		master_y =  trans[1];
		master_y += local_x*rot[3];
		master_y += local_y*rot[4];
		master_y += local_z*rot[5];
		master_z =  trans[2];
		master_z += local_x*rot[6];
		master_z += local_y*rot[7];
		master_z += local_z*rot[8];
	}
}

// this function will likely be template specialized
inline
__attribute__((always_inline))
void
TransformationMatrix::Multiply( TransformationMatrix const * rhs )
{
	// brute force multiplication of matrices just to get started

	// do nothing if identity
	if(rhs->identity) return;

	// transform translation part ( should reuse a mastertolocal transformation here )
	trans[0] += rot[0]*rhs->trans[0] + rot[1]*rhs->trans[1] + rot[2]*rhs->trans[2];
	trans[1] += rot[3]*rhs->trans[0] + rot[4]*rhs->trans[1] + rot[5]*rhs->trans[2];
	trans[2] += rot[6]*rhs->trans[0] + rot[7]*rhs->trans[1] + rot[8]*rhs->trans[2];

	// transform rotation part
	double newrot[9];
	// this can certainly be optimized
	for(int i=0;i<3;i++)
	{
		newrot[3*i] = rot[3*i]*rhs->rot[0] + rot[3*i+1]*rhs->rot[3]+rot[3*i+2]*rot[6];
		newrot[3*i+1] = rot[3*i]*rhs->rot[1] + rot[3*i+1]*rhs->rot[4]+rot[3*i+2]*rot[7];
		newrot[3*i+2] = rot[3*i]*rhs->rot[2] + rot[3*i+1]*rhs->rot[5]+rot[3*i+2]*rot[8];
	}
	memcpy(rot,newrot,sizeof(double)*9);
}

template<RotationIdType rid>
inline
__attribute__((always_inline))
void
multiplyrotationpart( double * __restrict__ lhsm, double const * __restrict__ rhsm )
{
	if(rid==252){
		double rm[9];
		for(int i=0;i<3;i++)
				{
					rm[3*i]   = lhsm[3*i] * rhsm[0];
					rm[3*i+1] = lhsm[3*i+1] * rhsm[4]  +  lhsm[3*i+2] * rhsm[7];
					rm[3*i+2] = lhsm[3*i+1] * rhsm[5]  +  lhsm[3*i+2] * rhsm[8];
				}
			memcpy(lhsm, rm, sizeof(double)*9);
			return;
		}
		if(rid==405){
			double rm[9];
			for(int i=0;i<3;i++)
					{
						rm[3*i]   = lhsm[3*i+1] * rhsm[3];
						rm[3*i+1] = lhsm[3*i] * rhsm[1] + lhsm[3*i+2] * rhsm[7];
						rm[3*i+2] = lhsm[3*i] * rhsm[2] + lhsm[3*i+2] * rhsm[8];
					}
			memcpy(lhsm, rm, sizeof(double)*9);
			return;
		}
		if(rid==882){
			double rm[9];
			for(int i=0;i<3;i++)
					{
						rm[3*i]   = lhsm[3*i+2] * rhsm[6];
						rm[3*i+1] = lhsm[3*i] * rhsm[1] + lhsm[3*i+1] * rhsm[4];
						rm[3*i+2] = lhsm[3*i] * rhsm[2] + lhsm[3*i+1] * rhsm[5];
					}
			memcpy(lhsm, rm, sizeof(double)*9);
			return;
		}
		if(rid==415){
			double rm[9];
			for(int i=0;i<3;i++)
					{
						rm[3*i]   = lhsm[3*i+1] * rhsm[3]  +  lhsm[3*i+2] * rhsm[6];
						rm[3*i+1] = lhsm[3*i] * rhsm[1];
						rm[3*i+2] = lhsm[3*i+1] * rhsm[5]  +  lhsm[3*i+2] * rhsm[8];
					}
			memcpy(lhsm, rm, sizeof(double)*9);
			return;
		}
		if(rid==496){
			double rm[9];
			for(int i=0;i<3;i++)
					{
						rm[3*i]   = lhsm[3*i] * rhsm[0] + lhsm[3*i+2] * rhsm[6];
						rm[3*i+1] = lhsm[3*i+1] * rhsm[4];
						rm[3*i+2] = lhsm[3*i] * rhsm[2] + lhsm[3*i+2] * rhsm[8];
					}
			memcpy(lhsm, rm, sizeof(double)*9);
			return;
		}
		if(rid==793){
			double rm[9];
			for(int i=0;i<3;i++)
					{
						rm[3*i]   = lhsm[3*i] * rhsm[0] + lhsm[3*i+1] * rhsm[3];
						rm[3*i+1] = lhsm[3*i+2] * rhsm[7];
						rm[3*i+2] = lhsm[3*i] * rhsm[2] + lhsm[3*i+1] * rhsm[5];
					}
			memcpy(lhsm, rm, sizeof(double)*9);
			return;
		}
		if(rid==638){
			double rm[9];
			for(int i=0;i<3;i++)
					{
						rm[3*i]   = lhsm[3*i+1] * rhsm[3]  +  lhsm[3*i+2] * rhsm[6];
						rm[3*i+1] = lhsm[3*i+1] * rhsm[4]  +  lhsm[3*i+2] * rhsm[7];
						rm[3*i+2] = lhsm[3*i] * rhsm[2];
					}
			memcpy(lhsm, rm, sizeof(double)*9);
			return;
		}
		if(rid==611){
			double rm[9];
			for(int i=0;i<3;i++)
					{
						rm[3*i]   = lhsm[3*i] * rhsm[0] + lhsm[3*i+2] * rhsm[6];
						rm[3*i+1] = lhsm[3*i] * rhsm[1] + lhsm[3*i+2] * rhsm[7];
						rm[3*i+2] = lhsm[3*i+1] * rhsm[5];
					}
			memcpy(lhsm, rm, sizeof(double)*9);
			return;
		}
		if(rid==692){
			double rm[9];
			for(int i=0;i<3;i++)
					{
						rm[3*i]   = lhsm[3*i] * rhsm[0] + lhsm[3*i+1] * rhsm[3];
						rm[3*i+1] = lhsm[3*i] * rhsm[1] + lhsm[3*i+1] * rhsm[4];
						rm[3*i+2] = lhsm[3*i+2] * rhsm[8];
					}
			memcpy(lhsm, rm, sizeof(double)*9);
			return;
		}
		if(rid==720){
			for(int i=0;i<3;i++)
					{
						lhsm[3*i]   = lhsm[3*i] * rhsm[0];
						lhsm[3*i+1] = lhsm[3*i+1] * rhsm[4];
						lhsm[3*i+2] = lhsm[3*i+2] * rhsm[8];
					}
			return;
		}
		if(rid==828){
			for(int i=0;i<3;i++)
					{
						lhsm[3*i]   = lhsm[3*i] * rhsm[0];
						lhsm[3*i+1] = lhsm[3*i+2] * rhsm[7];
						lhsm[3*i+2] = lhsm[3*i+1] * rhsm[5];
					}
		     return;
		}
		if(rid==756){
			for(int i=0;i<3;i++)
					{
						lhsm[3*i]   = lhsm[3*i+1] * rhsm[3] ;
					    lhsm[3*i+1] = lhsm[3*i] * rhsm[1];
						lhsm[3*i+2] = lhsm[3*i+2] * rhsm[8];
					}
			 return;
		}
		if(rid==918){
			for(int i=0;i<3;i++)
					{
						lhsm[3*i]   = lhsm[3*i+1] * rhsm[3];
						lhsm[3*i+1] = lhsm[3*i+2] * rhsm[7];
						lhsm[3*i+2] = lhsm[3*i] * rhsm[2];
					}
			 return;
		}
		if(rid==954){
			for(int i=0;i<3;i++)
					{
						lhsm[3*i]   = lhsm[3*i+2] * rhsm[6];
						lhsm[3*i+1] = lhsm[3*i] * rhsm[1];
						lhsm[3*i+2] = lhsm[3*i+1] * rhsm[5];
					}
		     return;
		}
		if(rid==1008){
			for(int i=0;i<3;i++)
					{
						lhsm[3*i]   = lhsm[3*i+2] * rhsm[6];
						lhsm[3*i+1] = lhsm[3*i+1] * rhsm[4];
						lhsm[3*i+2] = lhsm[3*i] * rhsm[2];
					}
		     return;
		}
		if(rid==1296){
			// diagonal case ( nothing to do )
			return;
		}
	// generic case
		else
		{
			double rm[9];
			for(int i=0;i<3;i++)
			{
				rm[3*i]   = lhsm[3*i] * rhsm[0] + lhsm[3*i+1] * rhsm[3]  +  lhsm[3*i+2] * rhsm[6];
				rm[3*i+1] = lhsm[3*i] * rhsm[1] + lhsm[3*i+1] * rhsm[4]  +  lhsm[3*i+2] * rhsm[7];
				rm[3*i+2] = lhsm[3*i] * rhsm[2] + lhsm[3*i+1] * rhsm[5]  +  lhsm[3*i+2] * rhsm[8];
			}
			memcpy(lhsm, rm, sizeof(double)*9);
		}
}

// the tid and rid are properties of the right hand matrix
template<TranslationIdType tid, RotationIdType rid>
inline
__attribute__((always_inline))
void
TransformationMatrix::MultiplyT( TransformationMatrix const * rhs )
{
	// transform translation part ( should reuse a mastertolocal transformation here )
	if( tid != 0)
	{
		trans[0] += rot[0] * rhs->trans[0] + rot[1] * rhs->trans[1] + rot[2] * rhs->trans[2];
		trans[1] += rot[3] * rhs->trans[0] + rot[4] * rhs->trans[1] + rot[5] * rhs->trans[2];
		trans[2] += rot[6] * rhs->trans[0] + rot[7] * rhs->trans[1] + rot[8] * rhs->trans[2];
	}
	multiplyrotationpart<rid>( rot, &rhs->rot[0] );
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
