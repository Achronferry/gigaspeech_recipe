# MAIN_ROOT=~/espnet
# KALDI_ROOT=/GPFS/data/chenyuyang/kaldi

# export PATH=$KALDI_ROOT/tools/openfst/bin:$PATH
export PATH=$PWD/utils/:$PATH
# [ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
# . $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

# . "${MAIN_ROOT}"/tools/activate_python.sh && . "${MAIN_ROOT}"/tools/extra_path.sh

# export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"

# NOTE(kamo): Source at the last to overwrite the setting
. local/path.sh
export PYTHONPATH=`pwd`:$PYTHONPATH