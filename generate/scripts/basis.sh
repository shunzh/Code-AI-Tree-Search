# set submit commands
if [ -x "$(command -v jbsub)" ]
then
  export flag="ccc"
  echo "Using LSF submission commands"
  export submit="jbsub -queue x86_6h -t 3:0 -cores 1+1 -require 'a100' -mem 256g"
elif [ -x "$(command -v srun)" ]
then
  export flag="aimos"
  echo "Using Slurm submission commands"
  export submit="srun --gres=gpu:1 --cpus-per-task=4 -N 1 --mem=256G --time 3:00:00 --error=%j.err --output=j.out"
else
  export flag=""
  echo "Don't what server this is. Running scripts locally."
  export submit=""
fi
