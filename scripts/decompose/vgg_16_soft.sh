for batch_size in 128 256 512
do
    for lr in 0.1 0.05 0.01 0.005 0.001
    do
        for weight_decay in 0.0005 0.005
        do
            python decompose.py --arch vgg_16_bn --ckpt checkpoint/cifar10/coring/vgg16_soft.pt -r 8 -cpr [0.21]*7+[0.75]*5 --n_iter_max 1000 --name FromCORING --batch_size $batch_size --lr $lr --weight_decay $weight_decay
        done
    done
done
