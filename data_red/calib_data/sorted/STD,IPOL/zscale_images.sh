#!/bin/bash 

# create an array with all the filer/dir inside ~/myDir
currdir = $(pwd)
arr=(~/myDir/*)

# 
for f in "${arr[@]}"; do
   echo "$f"
done
