#!/bin/bash
#SBATCH -p pascal
#SBATCH --nodes 1
#SBATCH -J elket

. /etc/profile.d/modules.sh
module load intel
module load cuda/cuda-10.0

make all TYPE=FLOAT
make p4

