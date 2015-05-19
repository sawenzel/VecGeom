#!/bin/bash 
#declare filenames


fname1=$1 
#"cms2015.root"

fname2=$2 
#"Sandro_tubes.txt"

fname3="Geom_Err_Report.log"

# Declare time out time for each Geometry diagnosys
declare -i timeout=10

# Make a directory in present directory as dir.Shape.Diag
logDIR="dir.Shape.Diag"
if [ ! -d "$logDIR" ]; then
    mkdir ${logDIR}
	echo ${logDIR} " -- a new directory has been created to store all log files"
else	
	echo ${logDIR} " -- directory already exists and all log files are being stored here."
fi

# here declare how many tests are required e.g. 20 in batch
testNum=2
# declare array in which all geometries are stored
declare -a Geometry
# link file descriptor 10 with stdin
exec 10<&0
exec < $fname2

let count=1

while read LINE; do
	Geometry[$count]=$LINE
	((count++))
done

((count--))
#echo ${Geometry[@]}
echo Number of Elements: ${#Geometry[@]}


# restoring stdin from file descriptor and closing descriptor
exec 0<&10 10<&-

# Now generate random serial number of geometry and test
# We perform maximum $testNum numbers of test per slot
for ((index=1; index <= testNum; index++)); do 
	cue=$((RANDOM % count+1))
	echo " Index = " ${index} "  cue = " ${cue} " Geometry = " ${Geometry[$cue]} "<<<<<<<"
	timeout --preserve-status --kill-after=${timeout}s ${timeout} BenchmarkShapeFromROOTFile $fname1  ${Geometry[$cue]} > "${logDIR}/${Geometry[$cue]}.log"

	echo "Exit code :-> " $?

	if [ $? -ne 0 ]; then
		errmesg="${Geometry[$cue]}  ----->  Failed"
		echo ${errmesg} >> ${logDIR}/${fname3}
		#echo "test failed"
	else
		#echo "test passed"
		errmesg="${Geometry[$cue]}  Passed."
		echo ${errmesg} >> ${logDIR}/${fname3}
	fi
done


