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

class Vector3DFast
{
private:
	Vc::Memory<Vc::Vector<double >, 3 > internalVcmemory;

public:
	Vector3DFast( double x,  double y,  double z) : internalVcmemory() {
		SetX(x);SetY(y);SetZ(z);
	}

	inline
		__attribute__((always_inline))
		Vector3DFast & operator+=( Vector3DFast const & rhs )
			{
				this->internalVcmemory+=rhs.internalVcmemory;
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
};



inline
static
std::ostream& operator<<( std::ostream& stream, Vector3DFast const & vec )
{
	stream << "{ " << vec.GetX() << " , " << vec.GetY() << " , " << vec.GetZ() << " } ";
	return stream;
}



#endif /* VECTOR3DFAST_H_ */
