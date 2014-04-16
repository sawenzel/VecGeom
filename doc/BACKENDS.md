Backends
========

In order to be independent of SIMD implementations, VecGeom introduces the concept of *backends*. These are modular header files that provide types along with operations that work on these types. By doing this, the types themselves govern the instructions generated to perform operations.

A backend is a struct containing a number of typedefs and static variables:

    struct kVc {
      typedef Vc::int_v                   int_v;
      typedef Vc::Vector<Precision>       precision_v;
      typedef Vc::Vector<Precision>::Mask bool_v;
      constexpr static bool early_returns = false;
      const static precision_v kOne;
      const static precision_v kZero;
      const static bool_v kTrue;
      const static bool_v kFalse;
    };

The types defined in the backend struct will then have to implement operations necessary by the algorithms that employ them. The struct is defined in the file `backend/<backend>/backend.h`.

In addition, backends will have to implement a number of patterns to loop over input data and dispatch batches to the vector kernels. This is done in the file `backend/<backend>/implementation.h`.

Abstracted algorithms
=====================

To make use of the modular backends, core algorithms are implemented in a generic way which templates on the backend type:

    template<typename Backend>
    void BoxUnplacedInside(...)

These algorithms expect a number of operations to be implemented for the types specified in the backend struct, including arithmetic operators, mathematical operations and vector-style mask operations.

Implementation example
----------------------

The following example will demonstrate the vector interface call tree by using the example of the `Inside` method of the `SpecializedBox` class. A triplet of periods (`...`) means code is left out for readability purposes. The specialized box templates on its placement through a translational and rotational id to generate specialized transformation code.

The vector interface exposed to the user takes an input container and an output array, defined in `volumes/specialized_box.h`:

    template <TranslationCode trans_code, RotationCode rot_code>
    void SpecializedBox<trans_code, rot_code>::Inside(...) const {
      Inside_Looper<trans_code, rot_code>(*this, ...);
    }

This call is dispatched to the free function `Inside_Looper`, which is located in the `implementation.h` header file of the backend used.

Assuming Vc is used as the SIMD backend, the `Inside_Looper` function is implemented as:

    template <TranslationCode trans_code, RotationCode rot_code,
              typename VolumeType, typename ContainerType>
    void VPlacedVolume::Inside_Looper(...) {
      for (...) {
        const VcBool result =
            volume.template InsideDispatch<trans_code, rot_code, kVc>(
              Vector3D<VcPrecision>(...)
            );
        ...
      }
    }

When called from the `Inside` method of the specialized box, the last two template arguments can be inferred by the function arguments. The looper then *loops* over the input in steps of the SIMD vector size in question, and calls the `InsideDispatch` method of the calling volume.

The dispatch function of the volume is the one finally responsible of dispatching to the abstracted code performing the work, and is located in the unspecialized version of the volume, in this case `volumes/placed_box.h`:

    template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
    typename Backend::bool_v PlacedBox::InsideDispatch(...) const {
      typename Backend::bool_v output;
      BoxInside<trans_code, rot_code, Backend>(..., output);
      return output;
    }

As all template parameters and function arguments are known at this stage, the kernel for the box inside method in `volumes/kernel/box_kernel.h` can be called:

    template <TranslationCode trans_code, RotationCode rot_code, typename Backend>
    void BoxInside(...)

As all the above is inlined, this template stack will result in transformationally specialized SIMD instructions as implemented by the backend, looping over the user input and populating the output array.