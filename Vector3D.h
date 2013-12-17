/*
 * Vector3D.h
 *
 *  Created on: Nov 10, 2013
 *      Author: swenzel
 */

#ifndef VECTOR3D_H_
#define VECTOR3D_H_

#include "GlobalDefs.h"
#include "mm_malloc.h"
#include <vector>
#include <iostream>

struct Vector3D
{
	double x,y,z;
	Vector3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {};
	Vector3D() : x(0), y(0), z(0) {};

	/*
	double operator[](int i) const
	{
		return &x+i;
	}
*/

	Vector3D & operator=(Vector3D const & rhs)
	{
		this->x=rhs.x;
		this->y=rhs.y;
		this->z=rhs.z;
		return *this;
	}

	static
	inline
	double scalarProduct( Vector3D const & left, Vector3D const & right)
	{
		return left.x*right.x + left.y*right.y + left.z*right.z;
	}

	static
	inline
	double scalarProductInXYPlane( Vector3D const & left, Vector3D const & right)
	{
		return left.x*right.x + left.y*right.y;
	}

	/*
	void print() const
	{
		std::cout << " 3DVector " << this << std::endl;
		std::cout << " x : " << x << std::endl;
		std::cout << " y : " << y << std::endl;
		std::cout << " z : " << z << std::endl;
	}
*/
	void print() const
	{
	//	std::cout << "3DVector (instance ) " << (*this) << std::endl;
	}

};

static
std::ostream& operator<<( std::ostream& stream, Vector3D const & vec )
{
	stream << "{ " << vec.x << " , " << vec.y << " , " << vec.z << " } ";
	return stream;
}

struct Vectors3DSOA
{
private:
  double * xvec;
  double * yvec;
  double * zvec;

public:
  int size; // the size
  double *x; // these are just "views" to the real data in xvec ( with the idea that we can set x to point to different start locations in xvec )
  double *y;
  double *z;

  Vectors3DSOA() : size(0), xvec(nullptr), yvec(nullptr), zvec(nullptr),
                   x(nullptr), y(nullptr), z(nullptr) {}

  Vectors3DSOA(Vectors3DSOA const &other) {
    const int size = other.size;
    alloc(size);
    for (int i = 0; i < size; ++i) {
      xvec[i] = other.xvec[i];
      yvec[i] = other.yvec[i];
      zvec[i] = other.zvec[i];
    }
  }

  // Creates the SOA Vector as a view into another SOA Vector
  Vectors3DSOA(Vectors3DSOA const &other, const unsigned index,
               const unsigned size_) {
    size = size_;
    xvec = other.xvec;
    yvec = other.yvec;
    zvec = other.zvec;
    setstartindex(index);
  }

  ~Vectors3DSOA() {
    // dealloc();
  }

  Vector3D getAsVector(int index) const
  {
	  return Vector3D(x[index],y[index],z[index]);
  }

  void getAsVector(int index, Vector3D & v) const
  {
	  v.x=x[index];
  	  v.y=y[index];
  	  v.z=z[index];
  }

  void toStructureOfVector3D( std::vector<Vector3D> & v ) const
  {
	  for(auto i=0;i<size;i++)
	  {
		  getAsVector(i, v[i]);
	  }
  }


  void toPlainArray( double * v, int np )
  {
	  for(auto i=0;i<np;i++)
	  {
		  v[3*i]=x[i];
		  v[3*i+1]=y[i];
		  v[3*i+2]=z[i];
	  }
  }

  void toStructureOfVector3D( Vector3D * v )
    {
  	  for(auto i=0;i<size;i++)
  	  {
  		  getAsVector(i, v[i]);
  	  }
    }

  void setFromVector(int index, Vector3D const &v) const
  {
  	  x[index]=v.x;
  	  y[index]=v.y;
  	  z[index]=v.z;
  }

  void setstartindex(int index)
  {
    x=&xvec[index];
    y=&yvec[index];
    z=&zvec[index];
  }

  // checks if index will be aligned and if not go back to an aligned address
  void setstartindex_aligned(int /*index*/)
  {
    // TOBEDONE
  }

  void fill(double const * onedvec)
  {
    for(int i=0;i<size;++i)
      {
    	xvec[i]=onedvec[3*i];
    	yvec[i]=onedvec[3*i+1];
    	zvec[i]=onedvec[3*i+2];
      }
  }

  void alloc(int n)
  {
    this->size=n;
    xvec=(double*)_mm_malloc(sizeof(double)*size,ALIGNMENT_BOUNDARY); // aligned malloc (32 for AVX )
    yvec=(double*)_mm_malloc(sizeof(double)*size,ALIGNMENT_BOUNDARY);
    zvec=(double*)_mm_malloc(sizeof(double)*size,ALIGNMENT_BOUNDARY);
    x=xvec;y=yvec;z=zvec;
  }

  void dealloc()
  {
    if (xvec) _mm_free(xvec);
    if (yvec) _mm_free(yvec);
    if (zvec) _mm_free(zvec);
    x=0;y=0;z=0;
  }
};

 #endif /* VECTOR3D_H_ */
