# decompose original model with all ranks
for rank in 1 2 3 4 5 6 7 8
do
    python decompose.py --job_dir result/lr0.1_wd5e-4 -r $rank
done
