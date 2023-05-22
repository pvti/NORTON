for cpr in [0.]+[0.08]*6+[0.09]*6+[0.08]*26 [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12
do
    python prune.py densenet_40 --ckpt checkpoint/cifar10/decomposed/densenet_40/densenet_40_[0.]*100_3.pt --job_dir results/prune -r 3 -cpr $cpr
done
