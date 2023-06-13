for batch_size in 128 256 512
do
    for lr in 0.1 0.05 0.01 0.005 0.001
    do
        for weight_decay in 0.0005 0.005
        do
            python decompose.py --arch densenet_40 --ckpt checkpoint/cifar10/coring/densenet_40_soft.pt -r 8 -cpr [0.]+[0.08]*6+[0.09]*6+[0.08]*26 --n_iter_max 1000 --name FromCORING --batch_size $batch_size --lr $lr --weight_decay $weight_decay
            python decompose.py --arch densenet_40 --ckpt checkpoint/cifar10/coring/densenet_40_moderate.pt -r 8 -cpr [0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 --n_iter_max 1000 --name FromCORING --batch_size $batch_size --lr $lr --weight_decay $weight_decay
            python decompose.py --arch densenet_40 --ckpt checkpoint/cifar10/coring/densenet_40_hard.pt -r 8 -cpr [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*12 --n_iter_max 1000 --name FromCORING --batch_size $batch_size --lr $lr --weight_decay $weight_decay
        done
    done
done
