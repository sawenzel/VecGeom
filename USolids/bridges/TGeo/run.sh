#!/bin/bash

export FULLPATH=$(cd `dirname "$0"`/../..; pwd)

export LD_LIBRARY_PATH="$FULLPATH:$LD_LIBRARY_PATH"
root run.C
#valgrind --tool=callgrind root.exe run.C
#gdb root.exe
####   After call yourself at gdb prompt: 
#### (gdb) run run.C
