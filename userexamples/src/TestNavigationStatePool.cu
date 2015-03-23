#include "base/Global.h"
#include "navigation/NavigationState.h"
#include "navigation/NavStatePool.h"
#include "management/CudaManager.h"
#include "navigation/SimpleNavigator.h"
#include "backend/cuda/Backend.h"

#include <stdio.h>

__global__
void ProcessNavStates( void* gpu_ptr /* a pointer to buffer of navigation states */, int depth, int n )
{
  using vecgeom::cuda::NavigationState;
  using vecgeom::cuda::NavStatePool;

  const int i = vecgeom::cuda::ThreadIndex();
  if( i >= n ) return;

  if(i==0){
    printf("*** Size of NavigationState on the GPU: %d bytes (SizeOf=%d) at gpu_ptr=%p\n", sizeof(vecgeom::cuda::NavigationState), NavigationState::SizeOf(depth), gpu_ptr);
    // dump memory from GPU side
    NavigationState* dumper = reinterpret_cast<NavigationState*>(gpu_ptr);
    dumper->Dump();
  }

  // // // get the navigationstate for this thread/lane
  // NavigationState *states  = reinterpret_cast<NavigationState*>(gpu_ptr);
  // NavigationState *stateA = &(states[i]);
  // // Alternative: forcing size=160 to get to next state
  // NavigationState *state = reinterpret_cast<NavigationState*>(gpu_ptr+i*NavigationState::SizeOf(depth));
  // printf("Alternative state addresses: stateA=%p  |  state=%p\n", stateA, state);

  // get the navigationstate for this thread/lane
  // Warning: arithmetic on pointer to void or function type.
  vecgeom::cuda::NavigationState * state = reinterpret_cast<vecgeom::cuda::NavigationState*>( gpu_ptr +
        vecgeom::cuda::NavigationState::SizeOf(depth)*i ); 

  // actually do something to the states; here just popping off the top volume
  printf("From GPU: "); state->Print();

  // state->Pop();
  // state->SetPathPointer( oldpathpointer );
}

void LaunchNavigationKernel( void* gpu_ptr, int depth, int n )
{
  vecgeom::cuda::LaunchParameters launch =
      vecgeom::cuda::LaunchParameters(n);

  int gsize = launch.grid_size.x;
  int bsize = launch.block_size.x;
  printf("Launching GPU kernels: ProcessNavStates<<<%i,%i>>>...\n", gsize, bsize);
  ProcessNavStates<<< launch.grid_size , launch.block_size >>>( gpu_ptr, depth, n );
  printf("Returning from LaunchNavigationKernel now.\n");
}
