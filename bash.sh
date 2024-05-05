#!/bin/bash
# script to run the foo program on asax
source /apps/profiles/modules_asax.sh.dyn
module load cuda lapack
./tiled_mm
