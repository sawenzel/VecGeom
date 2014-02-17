/*
 * Vector3DFast.h
 *
 *  Created on: Feb 7, 2014
 *      Author: swenzel
 */

#ifndef VECTOR3DFAST_H_
#define VECTOR3DFAST_H_

#include <Vc/Vc>
#include "Vc/vector.h"
#include "Vc/Memory"
#include <iostream>
#include <cassert>
#include "mm_malloc.h"

class Vector3DFast
{
private:
	typedef Vc::Vector<double> base_t;
	typedef typename base_t::Mask mask_t;
	Vc::Memory<base_t, 3 > internalVcmemory;

	inline
	void SetVector(int i, Vc::Vector<double > const & v)
	{
		internalVcmemory.vector(i)=v;
	}

public:
	// references providing a "view" to the real memory ( just for convenience and compatibility with old vector usage )
	double & x;
	double & y;
	double & z;

	// for proper memory allocation on the heap
	static 
	void * operator new(std::size_t sz)
	{
//	  std::cerr  << "overloaded new called" << std::endl;
	  void *aligned_buffer=_mm_malloc( sizeof(Vector3DFast), 32 );
	  return ::operator new(sz, aligned_buffer);
	}

	static
	  void operator delete(void * ptr)
	{
	  // std::cerr << "overloaded delete" << std::endl;
	  _mm_free(ptr);
	}


	inline
	Vector3DFast( ) : internalVcmemory() , x(internalVcmemory[0]), 	y(internalVcmemory[1]), z(internalVcmemory[2])
	{
		// assert alignment
		void * a =  &internalVcmemory[0];
	//	std::cerr << a << " " << ((long long) a) % 32L << std::endl;
		assert( ((long long) a) % 32L == 0 );
	}

	inline
	Vector3DFast( Vector3DFast const & rhs ) : internalVcmemory(), x(internalVcmemory[0]), 	y(internalVcmemory[1]), z(internalVcmemory[2]) {
		//long long a = (long long) &internalVcmemory[0];
		//std::cerr << a << a/32. << std::endl;
		//	assert( a / 32. == 0 );
		void * a =  &internalVcmemory[0];
		//		std::cerr << a << " " << ((long long) a) % 32L << std::endl;
		assert( ((long long) a) % 32L == 0 );
		for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
			{
				this->internalVcmemory.vector(i)=rhs.internalVcmemory.vector(i);
			}
	}


	inline
	Vector3DFast( double x,  double y,  double z) : internalVcmemory(), x(internalVcmemory[0]),	y(internalVcmemory[1]), z(internalVcmemory[2]) {
		void * a =  &internalVcmemory[0];
	//	std::cerr << a << " " << ((long long) a) % 32L << std::endl;
		assert( ((long long) a) % 32L == 0 );
		//long long a = (long long) &internalVcmemory[0];
			//std::cerr << a/32. << std::endl;
			//assert( a / 32. == 0 );
			SetX(x);SetY(y);SetZ(z);
		}


	inline
	__attribute__((always_inline))
	Vector3DFast & operator+=( Vector3DFast const & rhs )
	{
		for( int i=0; i< 1 + 3/Vc::Vector<double>::Size; i++ )
		//for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
		{
		    base_t v1,v2;
		    v1 = rhs.internalVcmemory.vector(i);
		    v2 = this->internalVcmemory.vector(i);
#ifdef VECTORDEBUG
		    std::cerr << "adding " << v1 << " " << v2 << std::endl;
#endif
		    v2 += v1;
		    this->internalVcmemory.vector(i)=v2;
		}
		return *this;
	}


	inline
	__attribute__((always_inline))
	Vector3DFast & operator*=( double s )
	{
		this->internalVcmemory*=s;
		return *this;
	}


	inline
		__attribute__((always_inline))
		Vector3DFast & operator-=( Vector3DFast const & rhs )
		{
			this->internalVcmemory-=rhs.internalVcmemory;
			return *this;
		}



	// assignment operator
	inline
	__attribute__((always_inline))
	Vector3DFast & operator=( Vector3DFast const & rhs )
	{
		for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
		{
			base_t v1 = rhs.internalVcmemory.vector(i);
#ifdef VECTORDEBUG
			std::cerr << "assigning " << v1 << std::endl;
#endif
			this->internalVcmemory.vector(i)=rhs.internalVcmemory.vector(i);
		}
		return *this;
	}


	inline
	__attribute__((always_inline))
	Vector3DFast & operator*=( Vector3DFast const & rhs )
	{
		/*
			for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
			{
				//		std::cerr << "adding vector " << i <<  std::endl;
				this->internalVcmemory.vector(i) *= rhs.internalVcmemory.vector(i);
			}
				*/
			this->internalVcmemory*=rhs.internalVcmemory;
			return *this;
	}


	inline
	__attribute__((always_inline))
	Vector3DFast & operator/=( Vector3DFast const & rhs )
	{
			this->internalVcmemory/= rhs.internalVcmemory;
			return *this;
	}

	inline
	__attribute__((always_inline))
	double ScalarProduct( Vector3DFast const & rhs ) const
	{
		double sum=0.;
		Vc::Vector<double> s(Vc::Zero);
		for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
		{
			//	std::cerr << "adding vector " << i <<  std::endl;
			Vc::Vector<double> tmp1 = this->internalVcmemory.vector(i);
			Vc::Vector<double> tmp2 = rhs.internalVcmemory.vector(i);
			s+=tmp1*tmp2;
		}
		return s.sum();
	}


	inline double GetX() const {
		return internalVcmemory[ 0 ];
	}

	inline double GetY() const {
		return internalVcmemory[ 1 ];
	}

	inline double GetZ() const {
		return internalVcmemory[ 2 ];

		//return internalVcmemory.vector( 2 / Vc::Vector<double>::Size  )[ 2 % Vc::Vector<double>::Size ];
	}

	inline void SetX(double x)  {
		internalVcmemory[0]=x;}

	inline void SetY(double y)  {
		internalVcmemory[1]=y;}

	inline void SetZ(double z)  {
		internalVcmemory[2]=z;}


	// operator[] in an unchecked option ( checked in debug mode )
	inline
	double & operator[](int index)
	{
#ifdef DEBUG
		assert(index >0 );
		assert(index <3 );
#endif
		return internalVcmemory[index];
	}


	inline
	double const operator[](int index) const
	{
#ifdef DEBUG
		assert(index >0 );
		assert(index <3 );
#endif
		return internalVcmemory[index];
	}

	inline
	__attribute__((always_inline))
	Vector3DFast Abs() const
	{
		Vector3DFast tmp;
		for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
		{
			base_t v = this->internalVcmemory.vector(i);
			tmp.internalVcmemory.vector(i) = Vc::abs( v );
		}
		return tmp;
	}

	inline
	__attribute__((always_inline))
	double Sum() const
	{
		double s=0.;
		for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
		{
			base_t v = this->internalVcmemory.vector(i);
			s+= v.sum( );
		}
		return s;
	}

	inline
	__attribute__((always_inline))
	// tricky operation since we have to care for the padded component
	double Min() const
	{
		double d = internalVcmemory[0];
/*
		for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
		{
			base_t v = this->internalVcmemory.vector(i);
			double m = v.min();
			d = (d > m) ? m : d;
		}
*/
		d = (internalVcmemory[1] < d )? internalVcmemory[1] : d;
		d = (internalVcmemory[2] < d )? internalVcmemory[2] : d;
		return d;
	}


	inline
	__attribute__((always_inline))
	// tricky operation since we have to care for the padded component
	double MinButNotNegative() const
	{
		double d = internalVcmemory[0];
	/*
			for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
			{
					base_t v = this->internalVcmemory.vector(i);
				double m = v.min();
				d = (d > m) ? m : d;
			}
	*/
		d = ( internalVcmemory[1]<d )? internalVcmemory[1] : d;
		d = ( internalVcmemory[2]<d )? internalVcmemory[2] : d;
		return (d < 0)? 0 : d;
	}

	inline
	static
	__attribute__((always_inline))
	// this should be made faster ( possibly with a vector multiplication and addition and using a mask vector )
	Vector3DFast ChooseComponentsBasedOnCondition( Vector3DFast const & v1,
			Vector3DFast const & v2, Vector3DFast const & conditionvector )
	{
		Vector3DFast tmp;
		tmp.internalVcmemory[0] = (conditionvector.internalVcmemory[0] < 0)? v1.internalVcmemory[0] : v2.internalVcmemory[0];
		tmp.internalVcmemory[1] = (conditionvector.internalVcmemory[1] < 0)? v1.internalVcmemory[1] : v2.internalVcmemory[1];
		tmp.internalVcmemory[2] = (conditionvector.internalVcmemory[2] < 0)? v1.internalVcmemory[2] : v2.internalVcmemory[2];
		return tmp;
	}

	inline
	static
	__attribute__((always_inline))
	// this should be made faster ( possibly with a vector multiplication and addition and using a mask vector )
	Vector3DFast ChooseComponentsBasedOnConditionFast( Vector3DFast const & v1,
			Vector3DFast const & v2, Vector3DFast const & conditionvector )
		{
			Vector3DFast tmp;
			for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
			{
				base_t targetv = tmp.internalVcmemory.vector(i);
				base_t v1c = v1.internalVcmemory.vector(i);
				base_t v2c = v2.internalVcmemory.vector(i);

				base_t condv = conditionvector.internalVcmemory.vector(i);
				targetv = v2c;
				targetv( condv < 0 ) = v1c;
				tmp.internalVcmemory.vector(i)=targetv;
			}
			return tmp;
		}


	inline
	static
	__attribute__((always_inline))
	Vector3DFast Min( Vector3DFast const & v1, Vector3DFast const & v2)
	{
		Vector3DFast tmp;
		for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
			{
				base_t v1c = v1.internalVcmemory.vector(i);
				base_t v2c = v2.internalVcmemory.vector(i);
				tmp.internalVcmemory.vector(i)=Vc::min(v1c,v2c);
			}
		return tmp;
	}

	inline
	static
	__attribute__((always_inline))
	// this should be templated and made more general ( for instance generalizing on condition )
	bool ExistsIndexWhereBothComponentsPositive( Vector3DFast const & v1, Vector3DFast const & v2 )
	{
		// returns true of there exists an index i such that v1(i)>0 and v2(i)>0
		// used for instance in box distancetoin
		for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
			{
				base_t v1c = v1.internalVcmemory.vector(i);
				base_t v2c = v2.internalVcmemory.vector(i);
				mask_t m = v1c>0 && v2c>0;
				if( ! m.isEmpty( ) ) return true;
			}
		return false;
	}

	inline
	void
	print() const
	{
		Vector3DFast tmp;
		for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
		{
			std::cerr << "vector " << i << std::endl;
		}
	}


	inline
	__attribute__((always_inline))
	// a starting comparison operator
	bool IsAnySmallerZero( ) const
	{
		bool result = false;
		Vector3DFast tmp;
		for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
		{
			mask_t m = this->internalVcmemory.vector(i) < 0;
			result |= ! m.isEmpty();
		}
		return result;
	}

	inline
	__attribute__((always_inline))
	bool IsAnyLargerThan( double x ) const
	{
		bool result = false;
		Vector3DFast tmp;
		for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
		{
			base_t vec(x); // broadcast double to vec
			mask_t m = this->internalVcmemory.vector(i) > vec;
			if( !m.isEmpty() ) return true;
		}
		return result;
	}


	inline
	__attribute__((always_inline))
	// a starting comparison operator
	bool IsAnySmallerThan( Vector3DFast const & rhs ) const
	{
		bool result = false;
		for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
		{
			base_t v1,v2;
			v1 = this->internalVcmemory.vector(i);
			v2 = rhs.internalVcmemory.vector(i);
			mask_t m =  v1 < v2;
			result |= ! m.isEmpty();
// could early return here ...

		}
		return result;
	}

	inline
	__attribute__((always_inline))
	// a starting comparison operator
	bool IsAnyLargerThan( Vector3DFast const & rhs ) const
	{
			bool result = false;
			for( int i=0; i < 1 + 3/Vc::Vector<double>::Size; i++ )
			{
				base_t v1,v2;
				v1 = this->internalVcmemory.vector(i);
				v2 = rhs.internalVcmemory.vector(i);
				mask_t m =  v1 > v2;
				result |= ! m.isEmpty();
// could early return here
			}
			return result;
	}


};


inline
Vector3DFast const operator-(Vector3DFast const & lhs, Vector3DFast const & rhs)
{
	Vector3DFast tmp(lhs);
	tmp-=rhs;
	return tmp;
}


inline
Vector3DFast const operator+(Vector3DFast const & lhs, Vector3DFast const & rhs)
{
	Vector3DFast tmp(lhs);
	tmp+=rhs;
	return tmp;
}

inline
Vector3DFast const operator*(Vector3DFast const & lhs, Vector3DFast const & rhs)
{
	Vector3DFast tmp(lhs);
	tmp*=rhs;
	return tmp;
}


inline
Vector3DFast const operator*(double lhs, Vector3DFast const & rhs)
{
	Vector3DFast tmp(rhs);
	tmp*=lhs;
	return tmp;
}


inline
Vector3DFast const operator*(Vector3DFast const & lhs, double rhs)
{
	Vector3DFast tmp(lhs);
	tmp*=rhs;
	return tmp;
}


inline
Vector3DFast const operator/(Vector3DFast const & lhs, Vector3DFast const & rhs)
{
	Vector3DFast tmp(lhs);
	tmp/=rhs;
	return tmp;
}



inline
static
std::ostream& operator<<( std::ostream& stream, Vector3DFast const & vec )
{
	stream << "{ " << vec.GetX() << " , " << vec.GetY() << " , " << vec.GetZ() << " } ";
	return stream;
}

// to try something with Vc memory
// note: this is a class which should work optimally with vector instructions sets >= AVX
// at the moment this might produce worse performance on SSE
// this might be solved with some more template magic or fallback to the old code
class FastTransformationMatrix
{
// storage
private:
    // the translation
	Vector3DFast trans;

	// the columns of the rotation matrix ( as vectors )
	Vector3DFast rotcol1;
	Vector3DFast rotcol2;
	Vector3DFast rotcol3;

	// the rows of the rotation matrix ( as vectors )
	Vector3DFast rotrow1;
	Vector3DFast rotrow2;
	Vector3DFast rotrow3;

	bool identity;

	inline
	void
	columnupdate( )
	{
		rotcol1.SetX( rotrow1.GetX() ); rotcol1.SetY( rotrow2.GetX() ); rotcol1.SetZ( rotrow3.GetX() );
		rotcol2.SetX( rotrow1.GetY() ); rotcol2.SetY( rotrow2.GetY() ); rotcol2.SetZ( rotrow3.GetY() );
		rotcol3.SetX( rotrow1.GetZ() ); rotcol3.SetY( rotrow2.GetZ() ); rotcol3.SetZ( rotrow3.GetZ() );
	}


public:
	// for proper memory allocation on the heap
	static 
	void * operator new(std::size_t sz)
	{
	  std::cerr  << "overloaded new called for matrix" << std::endl; 
	  void *aligned_buffer=_mm_malloc( sizeof(FastTransformationMatrix), 32 );
	  return ::operator new(sz, aligned_buffer);
	}

	static
	  void operator delete(void * ptr)
	{
	  // std::cerr << "overloaded delete" << std::endl;
	  _mm_free(ptr);
	}


	FastTransformationMatrix() : trans(), rotcol1(), rotcol2(), rotcol3(),
		rotrow1(), rotrow2(), rotrow3(), identity(false)
	{};


	FastTransformationMatrix(double tx, double ty, double tz, double phi, double theta, double psi)
	{
		trans.SetX(tx);
		trans.SetY(ty);
		trans.SetZ(tz);

		setAngles(phi, theta, psi);
	}

	FastTransformationMatrix(double tx, double ty, double tz,
			double r0, double r1, double r2, double r3, double r4, double r5, double r6, double r7,
			double r8 )
	{
		trans.SetX(tx);
		trans.SetY(ty);
		trans.SetZ(tz);
		rotrow1.SetX(r0);
		rotrow1.SetY(r1);
		rotrow1.SetZ(r2);
		rotrow2.SetX(r3);
		rotrow2.SetY(r4);
		rotrow2.SetZ(r5);
		rotrow3.SetX(r6);
		rotrow3.SetY(r7);
		rotrow3.SetZ(r8);
		columnupdate();
	}

	void SetTrans( double const * t)
	{
		trans.SetX(t[0]);
		trans.SetY(t[1]);
		trans.SetZ(t[2]);
	}
	void SetRotation( double const * r)
	{
		rotrow1.SetX(r[0]);
		rotrow1.SetY(r[1]);
		rotrow1.SetZ(r[2]);
		rotrow2.SetX(r[3]);
		rotrow2.SetY(r[4]);
		rotrow2.SetZ(r[5]);
		rotrow3.SetX(r[6]);
		rotrow3.SetY(r[7]);
		rotrow3.SetZ(r[8]);
		columnupdate();
	}


	inline
	void SetToIdentity(){
		// this has to be done eventually with some global const vectors to reduce the number of moves
		trans.SetX(0);
		trans.SetY(0);
		trans.SetZ(0);

		rotrow1.SetX(1.);
		rotrow1.SetY(0.);
		rotrow1.SetZ(0.);

		rotrow2.SetX(0.);
		rotrow2.SetY(1.);
		rotrow2.SetZ(0.);

		rotrow3.SetX(0.);
		rotrow3.SetY(0.);
		rotrow3.SetZ(1.);

		columnupdate();

		identity=true;
	};




	// we have far less specializations here ( since optimized for 1particle case )
	template <int tid, int rid>
	inline
	void
	__attribute__((always_inline))
	MasterToLocal(Vector3DFast const & master, Vector3DFast & local) const
	{
		if(tid == 0) // no translation
		{
			if(rid == 1296) // identity
			{
				local=master;
			}
			else // rotation
			{
				local=master.GetX()*rotrow1 + master.GetY()*rotrow2 + master.GetZ()*rotrow3;
			}
		}
		else // with translation
		{
			if(rid==1296)
			{
				local = master - trans;
			}
			else
			{
				Vector3DFast tmp=master - trans;
				local = tmp.GetX()*rotrow1 + tmp.GetY()*rotrow2 + tmp.GetZ()*rotrow3;
			}
		}
	}

	template <int rid>
	inline
	void
	__attribute__((always_inline))
	MasterToLocalVec(Vector3DFast const & master, Vector3DFast & local) const
	{
		return MasterToLocal<0,rid>(master,local);
	}


    // not templated since less important
	inline
	__attribute__((always_inline))
	void
	LocalToMaster(Vector3DFast const & local, Vector3DFast & master) const
	{
		master = trans + local.GetX()*rotcol1 + local.GetY()*rotcol2 + local.GetZ()*rotcol3;
	}


	void setAngles(double phi, double theta, double psi)
	{
		double degrad = M_PI/180.;
		double sinphi = sin(degrad*phi);
		double cosphi = cos(degrad*phi);
		double sinthe = sin(degrad*theta);
		double costhe = cos(degrad*theta);
		double sinpsi = sin(degrad*psi);
		double cospsi = cos(degrad*psi);

		// this is setting the rowvectors
		rotrow1.SetX( cospsi*cosphi - costhe*sinphi*sinpsi );
		rotrow1.SetY( -sinpsi*cosphi - costhe*sinphi*cospsi );
		rotrow1.SetZ( sinthe*sinphi );

		rotrow2.SetX( cospsi*sinphi + costhe*cosphi*sinpsi );
		rotrow2.SetY( -sinpsi*sinphi + costhe*cosphi*cospsi );
		rotrow2.SetZ( -sinthe*cosphi );

		rotrow3.SetX( sinpsi*sinthe );
		rotrow3.SetY( cospsi*sinthe );
		rotrow3.SetZ( costhe );

		columnupdate();
	}




	void print() const
	{
		// print the matrix in 4x4 format
		printf("%12.11f\t%12.11f\t%12.11f    Tx = %10.6f\n", rotcol1.GetX(), rotcol2.GetX(), rotcol3.GetX(), trans.GetX());
		printf("%12.11f\t%12.11f\t%12.11f    Ty = %10.6f\n", rotcol1.GetY(), rotcol2.GetY(), rotcol3.GetY(), trans.GetY());
		printf("%12.11f\t%12.11f\t%12.11f    Tz = %10.6f\n", rotcol1.GetZ(), rotcol2.GetZ(), rotcol3.GetZ(), trans.GetZ());
	}

	// the tid and rid are properties of the right hand matrix
	template<int tid, int rid>
	inline
	__attribute__((always_inline))
	void Multiply( FastTransformationMatrix const * rhs )
	{
		// do nothing if identity
	    // if(rhs->identity) return;
		// transform translation part ( should reuse a mastertolocal transformation here )
		if(tid>0)
		{
			trans += rotcol1*rhs->trans.GetX();
			trans += rotcol2*rhs->trans.GetY();
			trans += rotcol3*rhs->trans.GetZ();
		}

		if(rid!=1296) // if not identity
		{
			double tmpx = rotrow1.GetX();
			double tmpy = rotrow1.GetY();
			double tmpz = rotrow1.GetZ();
			rotrow1 = tmpx*rhs->rotrow1 + tmpy*rhs->rotrow2 + tmpz*rhs->rotrow3;

			tmpx = rotrow2.GetX();
			tmpy = rotrow2.GetY();
			tmpz = rotrow2.GetZ();
			rotrow2 = tmpx*rhs->rotrow1 + tmpy*rhs->rotrow2 + tmpz*rhs->rotrow3;

			tmpx = rotrow3.GetX();
			tmpy = rotrow3.GetY();
			tmpz = rotrow3.GetZ();
			rotrow3 = tmpx*rhs->rotrow1 + tmpy*rhs->rotrow2 + tmpz*rhs->rotrow3;

			// update rotcols
			// TOBE DONE BUT WE COULD ALSO JUST FORBID TO USE LOCALTOMASTER ON A GLOBAL MATRIX
		}
	}
};

#endif /* VECTOR3DFAST_H_ */
