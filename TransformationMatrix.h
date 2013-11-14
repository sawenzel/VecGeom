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

typedef int TranslationIdType;
typedef int RotationIdType;

class TransformationMatrix
{
private:
    // this can be varied depending on template specialization
	double trans[3];
	double rot[9];

	template<RotationIdType rid>
	inline
	void
	emitrotationcode(Vector3D const &, Vector3D &) const;

// the idea is to provide one general engine which works for both scale ( T = double ) as well as vector backends ( T = Vc::double_v )
	template<RotationIdType rid, typename T>
	inline
	void
	emitrotationcodeT( T const & mx, T const & my, T const & mz, T & lx, T & ly, T & lz ) const;

	void setAngles(double phi, double theta, double psi);

public:
	static
	inline
	RotationIdType
	getRotationFootprintS(double const *r){
		int footprint=0;
		// count zero entries and give back footprint which classifies them
		for(int i=0;i<9;i++)
		{
			if(r[i]==0.)
				footprint+=i*i*i; // cubic power identifies cases uniquely
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

	// constructor
	TransformationMatrix(double const *t, double const *r)
	{
	    trans[0]=t[0];
	    trans[1]=t[1];
		trans[2]=t[2];
		for(int i=0;i<9;i++)
			rot[i]=r[i];
		// we need to check more stuff ( for instance that product along diagonal is +1)
	}

	// more general constructor ala ROOT ( with Euler Angles )
	TransformationMatrix(double tx, double ty, double tz, double phi, double theta, double psi)
	{
		trans[0]=tx;
		trans[1]=ty;
		trans[2]=tz;
		setAngles(phi, theta, psi);
	}


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
	template <TranslationIdType tid, RotationIdType rid, typename T>
	inline
	void
	MasterToLocal(Vectors3DSOA const &, Vectors3DSOA &) const;


	/*
	 *  we should provide the versions for a Vc vector
	 */
	template<TranslationIdType tid, RotationIdType rid, typename T>
	inline
	void
	MasterToLocal(T const & masterx, T const & mastery, T const & masterz, T  & localx , T & localy , T & localz ) const;


	// to transform real vectors, we don't care about translation
	template <RotationIdType rid>
	inline
	void
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


	friend class PhysicalVolume;
	friend class GeoManager;
};

template <TranslationIdType tid, RotationIdType rid, typename T>
inline
void
TransformationMatrix::MasterToLocal(T const & masterx, T const & mastery, T const & masterz, T  & localx , T & localy , T & localz ) const
{
	if( tid==0 && rid ==0 ) // this means identity
		{
			localx = masterx;
			localy = mastery;
			localz = masterz;
		}
	else if( tid != 0 && rid == 0 ) // tid == 1 means we have
		{
			localx = masterx + trans[0];
			localy = mastery + trans[1];
			localz = masterz + trans[2];
		}
	else if( tid == 0 && rid!=0 ) // pure rotation
		{
			emitrotationcodeT<rid,T>(masterx, mastery, masterz, localx, localy, localz);
		}
	else if ( tid != 0 && rid!=0 ) // both rotation and translation
		{
			T mtx, mty, mtz;
			mtx = masterx + trans[0];
			mty=  mastery + trans[1];
			mtz = masterz + trans[2];
			emitrotationcodeT<rid,T>(mtx, mty, mtz, localx, localy, localz);
		}
}

template <TranslationIdType tid, RotationIdType rid>
inline
void
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
	localx=mtx*rot[0]+mty*rot[3]+mtz*rot[6];
	localy=mtx*rot[1]+mty*rot[4]+mtz*rot[7];
	localz=mtx*rot[2]+mty*rot[5]+mtz*rot[8];
}


template <RotationIdType rid>
inline
void
TransformationMatrix::emitrotationcode(Vector3D const & mt, Vector3D & local) const
{
	emitrotationcode<rid, double>( mt.x, mt.y, mt.z, local.x, local.y, local.z );
}


template <TranslationIdType tid, RotationIdType rid, typename T>
inline
void
TransformationMatrix<tid,rid>::MasterToLocal(Vectors3DSOA const & master_v, Vector3DSOA & local_v) const
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
		lx.store(&local_v.x[i]);
		ly.store(&local_v.y[i]);
		lz.store(&local_v.x[i]);
	}
	// need to treat tail part still

}


// -----------------------------------------------------------------------------------------------------------------------------------------------------------------

// now we define subclasses to TransformationMatrix as specialized classes
template <int tid, int rid>
class SpecializedTransformation : TransformationMatrix
{
public:
	virtual
	void
	MasterToLocal(Vector3D const &, Vector3D &) const;

	virtual
	void
	MasterToLocalVec(Vector3D const &, Vector3D &) const;
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



#endif /* TRANSFORMATIONMATRIX_H_ */
