for cpr in [0.]+[0.18]*29 [0.]+[0.12]*2+[0.4]*27 [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9
do
    python prune.py --arch resnet_56 --ckpt checkpoint/cifar10/decomposed/resnet_56/resnet_56_[0.]*100_1.pt --job_dir results/prune -r 1 -cpr $cpr
done
