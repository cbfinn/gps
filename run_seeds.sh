export DISPLAY=":0"
trap "exit" SIGINT SIGTERM
for SEED in 0 1 2; do
    for EXPT in lqr badmm mdgps_{lqr,nn}; do
        DIR="$1/${EXPT}/${SEED}"
        echo "Running expt $DIR"
        python python/gps/gps_main.py "$DIR"
    done
done
