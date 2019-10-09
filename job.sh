#!/bin/bash
#SBATCH -p teslav
#SBATCH --nodes 1
#SBATCH -J elket

. /etc/profile.d/modules.sh
module load intel
module load cuda/cuda-10.0

make all TYPE=DOUBLE
make p4

