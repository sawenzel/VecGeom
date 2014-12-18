#include "navigation/NavigationState.h"
#include "base/Global.h"
#include "management/CudaManager.h"
#include "navigation/SimpleNavigator.h"
#include "backend/cuda/Backend.h"

#include <stdio.h>

__global__ 
void ProcessNavStates( void* gpu_ptr /* a pointer to buffer of navigation states */, int depth, int n )
{
  const int i = vecgeom::cuda::ThreadIndex();
  if( i >= n ) return;

  if(i==0){
	printf("SIZEOF NAVSTATE ON THE GPU %d\n", sizeof(vecgeom::cuda::NavigationState));
  }

  // get the navigationstate for this thread/lane
  vecgeom::cuda::NavigationState * state = reinterpret_cast<vecgeom::cuda::NavigationState*>( gpu_ptr +
        vecgeom::cuda::NavigationState::SizeOf(depth)*i ); 

  //void * oldpathpointer = state->GetPathPointer();
  // need to fix path pointer
  //state->CorrectPathPointer();

//  if(i==5){
 //   printf("thread %i state %d\n", i, state->GetCurrentLevel());
  //  for(int j=0;j<state->GetCurrentLevel();++j)
  //  	    printf("volumep %p", state->At(j));
 // }

  // actually do something to the states; here just popping off the top volume
 // if( state->GetCurrentLevel()>0 )
    state->Pop();

   // state->SetPathPointer( oldpathpointer );
}

void LaunchNavigationKernel( void* gpu_ptr, int depth, int n )
{
  vecgeom::cuda::LaunchParameters launch =
      vecgeom::cuda::LaunchParameters(n);

  ProcessNavStates<<< launch.grid_size , launch.block_size >>>( gpu_ptr, depth, n );
}


