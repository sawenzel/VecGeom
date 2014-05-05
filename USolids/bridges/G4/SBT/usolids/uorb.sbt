#
# GEANT4 SBT Script to test UOrb
# DCW 19/3/99 
#
/test/maxPoints 10000
#
# --- uorb.a1.log
# Start with a nice and easy case
#
/solid/UOrb 0.5

/test/errorFileName log/uorb.a1.log
/test/run
#
/voxel/errorFileName log/uorbv.a1.log
/voxel/run

#
# --- uorb.a2.log
# Up the ante and generate points on a grid
#
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/uorb.a2.log
/test/run
#
# --- uorb.b1.log
# Try an odd case
#
/solid/UOrb 0.5
/test/errorFileName  log/uorb.b1.log
/test/run
#
/voxel/errorFileName log/uorbv.b1.log
/voxel/run

/performance/errorFileName log/uorbp.b1.log
/performance/run
#
exit
