/**
 * @file box_kernel.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_VOLUMES_KERNEL_BOXKERNEL_H_
#define VECGEOM_VOLUMES_KERNEL_BOXKERNEL_H_

#include "base/global.h"
#include "base/vector3d.h"
#include "base/transformation_matrix.h"
#include <ostream>

namespace VECGEOM_NAMESPACE {

/** the core inside function with matrix stripped; it expects a local point ) **/
template<typename Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void BoxUnplacedInside( Vector3D<Precision> const & dimensions,
						Vector3D<typename Backend::precision_v> const &localpoint,
						typename Backend::bool_v *const inside )
{
	Vector3D<typename Backend::bool_v> inside_dim(Backend::kFalse);
	  for (int i = 0; i < 3; ++i) {
	    inside_dim[i] = Abs(localpoint[i]) < dimensions[i];
	    if (Backend::early_returns) {
	      if (!inside_dim[i]) {
	        *inside = Backend::kFalse;
	        return;
	      }
	    }
	  }

	  if (Backend::early_returns) {
	    *inside = Backend::kTrue;
	  } else {
	    *inside = inside_dim[0] && inside_dim[1] && inside_dim[2];
	  }
}

/**
 *  a C-like function that returns if a particle is inside the box
 *   given specified by boxdimensions and placed with matrix **/
template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void BoxInside(Vector3D<Precision> const & boxdimensions,
               TransformationMatrix const &matrix,
               Vector3D<typename Backend::precision_v> const &point,
               typename Backend::bool_v *const inside) {

 // probably better like this:
  BoxUnplacedInside<Backend>(boxdimensions, matrix.Transform<trans_code,
                            rot_code>(point), inside);
}

/**
 * a C-like function that returns if a particle is inside the box
 * given specified by boxdimensions and placed with matrix
 *
 * this function also makes the transformed local point available to the caller
 **/
template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void BoxInside(Vector3D<Precision> const & boxdimensions,
               TransformationMatrix const &matrix,
               Vector3D<typename Backend::precision_v> const &point,
               Vector3D<typename Backend::precision_v> & localpoint,
               typename Backend::bool_v *const inside) {

  localpoint = matrix.Transform<trans_code, rot_code>(point);
  BoxUnplacedInside<Backend>( boxdimensions, localpoint, inside );
}

template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void BoxDistanceToIn(
    Vector3D<Precision> const &dimensions,
    TransformationMatrix const &matrix,
    Vector3D<typename Backend::precision_v> const &pos,
    Vector3D<typename Backend::precision_v> const &dir,
    typename Backend::precision_v const &step_max,
    typename Backend::precision_v *const distance) {

  typedef typename Backend::precision_v Float;
  typedef typename Backend::bool_v Bool;

  Vector3D<Float> safety;
  Vector3D<Float> pos_local;
  Vector3D<Float> dir_local;
  Bool hit(false);
  Bool done(false);
  *distance = kInfinity;

  matrix.Transform<trans_code, rot_code>(pos, pos_local);
  matrix.TransformRotation<rot_code>(dir, dir_local);

  safety[0] = Abs(pos_local[0]) - dimensions[0];
  safety[1] = Abs(pos_local[1]) - dimensions[1];
  safety[2] = Abs(pos_local[2]) - dimensions[2];

  done |= (safety[0] >= step_max ||
           safety[1] >= step_max ||
           safety[2] >= step_max);
  if (done == true) return;

  Float next, coord1, coord2;

  // x
  next = safety[0] / Abs(dir_local[0] + kTiny);
  coord1 = pos_local[1] + next * dir_local[1];
  coord2 = pos_local[2] + next * dir_local[2];
  hit = safety[0] > 0 &&
        pos_local[0] * dir_local[0] < 0 &&
        Abs(coord1) <= dimensions[1] &&
        Abs(coord2) <= dimensions[2];
  MaskedAssign(!done && hit, next, distance);
  done |= hit;
  if (done == true) return;

  // y
  next = safety[1] / Abs(dir_local[1] + kTiny);
  coord1 = pos_local[0] + next * dir_local[0];
  coord2 = pos_local[2] + next * dir_local[2];
  hit = safety[1] > 0 &&
        pos_local[1] * dir_local[1] < 0 &&
        Abs(coord1) <= dimensions[0] &&
        Abs(coord2) <= dimensions[2];
  MaskedAssign(!done && hit, next, distance);
  done |= hit;
  if (done == true) return;

  // z
  next = safety[2] / Abs(dir_local[2] + kTiny);
  coord1 = pos_local[0] + next * dir_local[0];
  coord2 = pos_local[1] + next * dir_local[1];
  hit = safety[2] > 0 &&
        pos_local[2] * dir_local[2] < 0 &&
        Abs(coord1) <= dimensions[0] &&
        Abs(coord2) <= dimensions[1];
  MaskedAssign(!done && hit, next, distance);

}

template <typename Backend>
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void BoxDistanceToOut(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &pos,
    Vector3D<typename Backend::precision_v> const &dir,
    typename Backend::precision_v const &step_max,
    typename Backend::precision_v & distance) {

	typedef typename Backend::precision_v Float;
	typedef typename Backend::bool_v Bool;

	Float big(1E30);

	Float saf[3];
	saf[0] = Abs(pos[0])-dimensions[0];
	saf[1] = Abs(pos[1])-dimensions[1];
	saf[2] = Abs(pos[2])-dimensions[2];

	// TODO: check this
	Bool inside = saf[0]< Float(0.) && saf[1] < Float(0.) && saf[2]< Float(0.);
	MaskedAssign( !inside, big, &distance );

	// TODO: could make the code more compact by looping over dir
	Float invdirx = 1.0/dir[0];
	Float invdiry = 1.0/dir[1];
	Float invdirz = 1.0/dir[2];

	Bool mask;
	Float distx = (dimensions[0]-pos[0]) * invdirx;
	mask = dir[0]<0;
	MaskedAssign( mask, (-dimensions[0]-pos[0]) * invdirx , &distx);


	Float disty = (dimensions[1]-pos[1]) * invdiry;
	mask = dir[1]<0;
	MaskedAssign( mask, (-dimensions[1]-pos[1]) * invdiry , &disty);


	Float distz = (dimensions[2]-pos[2]) * invdirz;
	mask = dir[2]<0;
	MaskedAssign( mask, (-dimensions[2]-pos[2]) * invdirz , &distz);

	distance = distx;
	mask = distance>disty;
	MaskedAssign( mask, disty, &distance);
	mask = distance>distz;
	MaskedAssign( mask, distz, &distance);

	return;
}


template< TranslationCode trans_code,
		  RotationCode rot_code,
		  typename Backend >
void BoxSafetyToIn( Vector3D<Precision> const &dimensions,
					TransformationMatrix const & matrix,
					Vector3D<typename Backend::precision_v> const & point,
					typename Backend::precision_v & safety
				  )
{
   typedef typename Backend::precision_v Float;
   typedef typename Backend::bool_v Bool;

   Vector3D<Float> localpoint
   	   	   = matrix.Transform<trans_code,rot_code>(point);
   safety = -dimensions[0] + Abs(localpoint[0]);
   Float safy = -dimensions[1] + Abs(localpoint[1]);
   Float safz = -dimensions[2] + Abs(localpoint[2]);
   // check if we should use MIN/MAX here instead
   MaskedAssign( Bool( safy > safety ), safy, &safety );
   MaskedAssign( Bool( safz > safety ), safz, &safety );
}

template< typename Backend >
void BoxSafetyToOut(Vector3D<Precision> const &dimensions,
					Vector3D<typename Backend::precision_v> const & point,
					typename Backend::precision_v & safety
				  )
{
   typedef typename Backend::precision_v Float;
   typedef typename Backend::bool_v Bool;

   safety = dimensions[0] - Abs( point[0] );
   Float safy = dimensions[1] - Abs( point[1] );
   Float safz = dimensions[2] - Abs( point[2] );
   // check if we should use MIN here instead
   MaskedAssign( Bool( safy < safety ), safy, &safety );
   MaskedAssign( Bool( safz < safety ), safz, &safety );
}


} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_BOXKERNEL_H_
