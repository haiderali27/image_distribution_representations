#!/bin/bash
#
# At first let's give job some descriptive name to distinct
# from other jobs running on cluster
#SBATCH -J JP_JOB
#
# Let's redirect job's out some other file than default slurm-%jobid-out
#SBATCH --output=res/res_qqhaal_%a.txt
#SBATCH --error=res/err_qqhaal_%a.txt
#
#!/bin/bash
#SBATCH --job-name=qqhaal_zs
#SBATCH --partition=gpu
#SBATCH --time=6-23:59:00
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:0
#SBATCH  --nodes=1
#SBATCH --mail-type=END
#SBATCH --mail-user=haider.ali@tuni.fi
#
# These commands will be executed on the compute node:

# Load all modules you need below. Edit these to your needs.

# module lodad GCC/5.3.0-2.26
# module load cuda/10.1.243
# conda activate pynoptorch
# conda activate py3.9
# module load anaconda3

module load fgci-common
module load gcc/9.2.0
module load CUDA/11.2

module load miniconda

source activate w_env




# conda env list

# echo 'Line No x'

# if some error happens in the initialation of parallel process then you can
# get the debug info. This can easily increase the size of out.txt.
# export NCCL_DEBUG=INFO  # comment it if you are not debugging distributed parallel setup

# export NCCL_DEBUG_SUBSYS=ALL # comment it if you are not debugging distributed parallel setup

# find the ip-address of one of the node. Treat it as master
ip1=`hostname -I | awk '{print $1}'`
echo $ip1

# Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_ADDR=$(hostname)

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

#train params
EXP_NAME=$1

# Finally run your job. Here's an example of a python script.

python CalculateDistributions_mse.py


