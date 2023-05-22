for cpr in [0.05]*7+[0.2]*6 [0.2]*7+[0.5]*6 [0.25]*7+[0.75]*6
do
    python prune.py --ckpt checkpoint/cifar10/decomposed/vgg/vgg_16_bn_[0.]*100_8.pt --job_dir results/prune -r 8 -cpr $cpr
done
