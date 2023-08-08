DATASET=$1

encoders=(resnet18 resnet34 resnet50 mobilenet_v2 timm-mobilenetv3_large_100 vgg11 vgg19_bn)
archs=(FPN Unet DeepLabV3)


for encoder in ${encoders[@]}; do
    for arch in ${archs[@]}; do
        python3.10 segment_train.py --dataset $DATASET --accelerator gpu --epochs 20 --n_cpu 6 --batch_size 128 --encoder $encoder --arch $arch
    done
done
