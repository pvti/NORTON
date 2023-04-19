for rank in 5 6 7 8
do
    python decompose_coring.py --pruned checkpoint/cifar10/pruned/vgg16_[0.05]*7+[0.2]*5.pt --job_dir prune_decompose -r $rank -cpr [0.05]*7+[0.2]*5 --gpu 0 &
    python decompose_coring.py --pruned checkpoint/cifar10/pruned/vgg16_[0.21]*7+[0.75]*5.pt --job_dir prune_decompose -r $rank -cpr [0.21]*7+[0.75]*5 --gpu 1 &
    python decompose_coring.py --pruned checkpoint/cifar10/pruned/vgg16_[0.3]*7+[0.75]*5.pt --job_dir prune_decompose -r $rank -cpr [0.3]*7+[0.75]*5 --gpu 2
done