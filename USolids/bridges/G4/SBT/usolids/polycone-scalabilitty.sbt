
#
# GEANT4 SBT Script to test G4Polycone
# DCW 19/3/99 First try
#
# Increment the number below to **really** waste CPU time
#

#<<<<<<< polycone.geant4
#test/maxPoints 5000
#test/maxVoxels 100
#=======
#test/maxPoints 1000
#>>>>>>> 1.2

#
# --- polycone.a1.log
# Start with a nice and easy case
#

# r = [1, 1, 1, 0, 0, 1.5,1.5, 1.2];
# z = [1,2,3,4,4,3,2,1];
# figure; plot(z,r);

/performance/maxPoints 10000
/performance/repeat 1


# Geant4 had problems with this, some points in enclosing cylinder were reported as inside
/solid/G4Polycone2 0 360 3 (0, 1, 2) (1, 1, 1) (2, 3, 2)

/performance/errorFileName log/polycone-3s-360-perf/polyconep.a1.log
/performance/maxPoints 1000
/performance/repeat 1
/control/execute usolids/performance.sbt

/solid/G4Polycone2 0 360 10 (1, 2, 2, 3, 3, 4, 10, 15, 15, 18) (0.5, 1, 2, 2, 3, 3, 1, 4, 1, 1)  (1, 2, 3, 3, 4, 4, 5, 5, 4, 2)
/performance/maxPoints 1
/performance/repeat 100

/performance/errorFileName log/polycone-10s-360-perf/polyconep.a1.log

#/control/execute usolids/performance.sbt

/performance/maxInsidePercent 25
/performance/maxOutsidePercent 50
/performance/method Inside
/performance/run


# ## Polycone 100 #######################################################################
#
/performance/maxPoints 10000
/solid/G4Polycone2 0 360 100 (1,3,3,4,10,15,15,18,18,19,19,20,20,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,34,34,35,35,36,36,37,37,38,38,39,39,40,40,41,41,42,42,43,43,44,44,45,45,46,46,47,47,48,48,49,49,50,50,51,51,52,52,53,53,54,54,55,55,56,56,57,57,58,58,59,59,60,60,61,61,62,62,63,125,126) (0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5) (1,1,3,3,1,4,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1)

/performance/errorFileName log/polycone-100s-360-perf/polyconep.a1.log
/performance/repeat 100
/control/execute usolids/performance.sbt

exit



/solid/G4Polycone 0 180 10 (0, 1, 2, 2, 3, 3, 1, 4, 1, 0) (1, 2, 2, 3, 3, 4, 10, 15, 15, 18) 
/performance/maxPoints 10000

/performance/errorFileName log/polycone-10-180-perf/polyconep.a1.log
/performance/repeat 100
/control/execute usolids/performance.sbt

exit

/solid/G4Polycone 0 180 3 (1, 1, 0) (1, 2, 3) 

/performance/errorFileName log/polycone-3-180-test/polyconep.a1.log
/performance/maxPoints 10000
/performance/repeat 1000
/control/execute usolids/performance.sbt

exit



# ## Polycone 100 #######################################################################
#
/performance/maxPoints 100000
/solid/G4Polycone2 0 180 100(1,3,3,4,10,15,15,18,18,19,19,20,20,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,34,34,35,35,36,36,37,37,38,38,39,39,40,40,41,41,42,42,43,43,44,44,45,45,46,46,47,47,48,48,49,49,50,50,51,51,52,52,53,53,54,54,55,55,56,56,57,57,58,58,59,59,60,60,61,61,62,62,63,125,126) (0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5) (1,1,3,3,1,4,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1)

/performance/errorFileName log/polycone-100s-180-perf/polyconep.a1.log
/performance/repeat 1
/control/execute usolids/performance.sbt

#exit



/solid/G4Polycone2 0 180 10 (1, 2, 2, 3, 3, 4, 10, 15, 15, 18) (0.5, 1, 2, 2, 3, 3, 1, 4, 1, 1)  (1, 2, 3, 3, 4, 4, 5, 5, 4, 2)
/performance/maxPoints 100000

/performance/errorFileName log/polycone-10s-180-perf/polyconep.a1.log
/performance/repeat 1
/control/execute usolids/performance.sbt


exit

# Geant4 had problems with this, some points in enclosing cylinder were reported as inside
/solid/G4Polycone2 0 180 3 (0, 1, 2) (1, 1, 1) (2, 3, 2)

/performance/errorFileName log/polycone-3s-180-perf/polyconep.a1.log
/performance/maxPoints 10000
/performance/repeat 1
/control/execute usolids/performance.sbt

exit




#exit
/polyconep.a1.log
/performance/repeat 100
/control/execute usolids/performance.sbt

#exit



/solid/G4Polycone2 0 180 10 (1, 2, 2, 3, 3, 4, 10, 15, 15, 18) (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  (1, 2, 3, 3, 4, 4, 5, 5, 4, 2)
/performance/maxPoints 100000

/performance/errorFileName log/polycone-10-180-perf/polyconep.a1.log
/performance/repeat 100
/control/execute usolids/performance.sbt

#exit


/solid/G4Polycone2 0 360 10 (1, 2, 2, 3, 3, 4, 10, 15, 15, 18) (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  (1, 2, 3, 3, 4, 4, 5, 5, 4, 2)
/performance/maxPoints 100000

/performance/errorFileName log/polycone-10-360-perf/polyconep.a1.log
/performance/repeat 100
/control/execute usolids/performance.sbt

#exit



/solid/G4Polycone2 0 180 3 (0, 1, 2) (1, 2, 1) (2, 3, 2)
/performance/errorFileName log/polycone-3-180-perf/polyconep.a1.log
/performance/maxPoints 100000
/performance/repeat 100
/control/execute usolids/performance.sbt

#exit










/performance/maxPoints 10000

# ## Polycone 100 #######################################################################
#
/solid/G4Polycone 0 180 100 (0,1,3,3,1,4,1,0,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,0) (1,3,3,4,10,15,15,18,18,19,19,20,20,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,34,34,35,35,36,36,37,37,38,38,39,39,40,40,41,41,42,42,43,43,44,44,45,45,46,46,47,47,48,48,49,49,50,50,51,51,52,52,53,53,54,54,55,55,56,56,57,57,58,58,59,59,60,60,61,61,62,62,63,125,126)
/performance/errorFileName log/polycone-100-perf/polyconep.a1.log
/performance/repeat 100
/control/execute usolids/performance.sbt
















# ## Polycone 20 #######################################################################
#
/solid/G4Polycone 0 180 20 (0,1,3,3,1,4,1,0,2,2,1,1,2,2,1,1,2,2,1,0) (1,3,3,4,10,15,15,18,18,19,19,20,20,21,21,22,22,23,45,46)
/performance/errorFileName log/polycone-20-180-p10k/polyconep.a1.log
/performance/repeat 100
/control/execute usolids/performance.sbt

exit

/performance/maxPoints 10000

/solid/G4Polycone 0 180 3 (0, 1, 0) (1, 2, 3) 
/performance/maxPoints 10000

/performance/errorFileName log/polycone-3-180-perf/polyconep.a1.log
/performance/maxPoints 10000
/performance/repeat 1000
/control/execute usolids/performance.sbt

exit






#
# ## Polycone 50 #######################################################################
#
/solid/G4Polycone 0 180 50 (0,1,3,3,1,4,1,0,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,1,0) (1,3,3,4,10,15,15,18,18,19,19,20,20,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,32,32,33,33,34,34,35,35,36,36,37,37,38,75,76)
/performance/errorFileName log/polycone-50-p10k/polyconep.a1.log
/performance/repeat 100
/control/execute usolids/performance.sbt
#



