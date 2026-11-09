#!/bin/bash
# submit_cnn.sh
set -u
umask 002

# 1. Check Kerberos ticket so the job can read /projects
if ! klist -s; then
    echo "ERROR: No valid Kerberos ticket! Your job won't be able to read /projects."
    echo "Run these commands first:"
    echo "  kinit -r 7d"
    echo "  krenew -b -K 30"
    exit 1
fi

WORK_DIR=$(cd "$(dirname "$0")" && pwd)
RUN_NAME=${1:-"run_$(date +%Y%m%d_%H%M%S)"}
OUT_DIR="$WORK_DIR/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"

# 2. Configure your training/testing parameters here
POS_DIR="/path/to/your/pos_data"
NEG_DIR="/projects/cos-lab-aotte6/cos-4a17118/Trinity/TrinityDemonstratorDataAnalysis/event_cleaning/Output/"
TRAIN_START=1
TRAIN_END=4000
TEST_START=4001
TEST_END=5000
EPOCHS=15

# 3. Write the HTCondor submit file
cat > "$OUT_DIR/cnn.submit" <<EOF
executable = $WORK_DIR/run_cnn_condor.sh
transfer_executable = false
initialdir = $WORK_DIR

arguments = --pos_dir $POS_DIR --neg_dir $NEG_DIR --train_start $TRAIN_START --train_end $TRAIN_END --test_start $TEST_START --test_end $TEST_END --epochs $EPOCHS --out_csv $OUT_DIR/metrics.csv

output = $OUT_DIR/cnn.out
error  = $OUT_DIR/cnn.err
log    = $OUT_DIR/cnn.log

getenv = True
should_transfer_files = NO

# Resource requests (increase memory if loading large event ranges)
request_cpus = 4
request_memory = 8GB
# Uncomment the line below ONLY if your Condor pool has GPUs (check with: condor_status -compact -constraint 'TotalGpus > 0')
# request_gpus = 1

queue
EOF

# 4. Submit the job
condor_submit "$OUT_DIR/cnn.submit"
echo "--------------------------------------------------------"
echo "Submitted CNN run: $RUN_NAME"
echo "Check queue status:  condor_q"
echo "Watch live output:   tail -f $OUT_DIR/cnn.out"
echo "Check errors:        cat $OUT_DIR/cnn.err"
echo "Output CSV will be:  $OUT_DIR/metrics.csv"