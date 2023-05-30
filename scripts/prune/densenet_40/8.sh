for cpr in [0.]+[0.1]*12+[0.]+[0.1]*12+[0.]+[0.1]*12 [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 [0.]+[0.3]*12+[0.]+[0.3]*12+[0.]+[0.3]*12
do
    python prune.py --arch densenet_40 --ckpt checkpoint/cifar10/decomposed/densenet_40/densenet_40_[0.]*100_8.pt --job_dir results/prune -r 8 -cpr $cpr
done
