export DISPLAY=":0"
trap "exit" SIGINT SIGTERM
for SEED in 0 1 2; do
    DIR="$1/${SEED}"
    echo "Running expt $DIR"
    python python/gps/gps_main.py "$DIR"
done
