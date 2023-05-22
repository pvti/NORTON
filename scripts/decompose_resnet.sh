for rank in 1 2 3 4 5 6 7 8
do
    python decompose.py --arch resnet_56 --ckpt checkpoint/cifar10/resnet_56.pt -r $rank
done
