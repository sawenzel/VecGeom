#
# GEANT4 SBT Script to test utrd
# DCW 19/3/99 
#
/test/maxPoints 10000
#
# --- utrd.a1.log
# Start with a nice and easy case
#
/solid/UTrd 30 10 40 15 60
	
/test/errorFileName log/utrd.a1.log
/test/run
#
/voxel/errorFileName log/utrdv.a1.log
/voxel/run
#
# --- utrd.a2.log
# Up the ante and generate points on a grid
#
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/utrd.a2.log
/test/run
#
# --- utrd.b1.log
# Try an odd case
#
/solid/UTrd 30 10 40 15 60
/test/errorFileName  log/utrd.b1.log
/test/run
#
/voxel/errorFileName log/utrdv.b1.log
/voxel/run
#
exit
