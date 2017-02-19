./tools/train_net.py --gpu 1 \
  --solver models/pascal_voc/ResNet101_BN_SCALE_Merged/faster_rcnn_end2end/solver.prototxt \
  --weights ResNet-101-BN-SCALE-Merged/ResNet101_BN_SCALE_Merged.caffemodel \
  --imdb voc_0712_trainval \
  --iters 70000 \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml