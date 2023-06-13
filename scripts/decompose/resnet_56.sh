for batch_size in 128 256 512
do
    for lr in 0.1 0.05 0.01 0.005 0.001
    do
        for weight_decay in 0.0005 0.005
        do
            python decompose.py --arch resnet_56 --ckpt checkpoint/cifar10/coring/resnet56_soft.pt -r 7 -cpr [0.]+[0.18]*29 --n_iter_max 1000 --name FromCORING --batch_size $batch_size --lr $lr --weight_decay $weight_decay
            python decompose.py --arch resnet_56 --ckpt checkpoint/cifar10/coring/resnet56_moderate.pt -r 7 -cpr [0.]+[0.12]*2+[0.4]*27 --n_iter_max 1000 --name FromCORING --batch_size $batch_size --lr $lr --weight_decay $weight_decay
            python decompose.py --arch resnet_56 --ckpt checkpoint/cifar10/coring/resnet56_hard.pt -r 7 -cpr [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 --n_iter_max 1000 --name FromCORING --batch_size $batch_size --lr $lr --weight_decay $weight_decay
        done
    done
done