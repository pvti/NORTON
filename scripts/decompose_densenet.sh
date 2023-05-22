for rank in 1 2 3 4 5 6 7 8
do
    python decompose.py --arch densenet_40 --ckpt checkpoint/cifar10/densenet_40.pt -r $rank
done
