# NORTON's Demo
+ FasterRCNN for object detection
<table style="width: 100%; border: none; border-collapse: collapse;">
  <tr>
    <td style="width: 50%; padding: 10px; border: none;">
      <img src="assets/baseline_faster.gif" alt="Baseline" style="width: 100%;">
    </td>
    <td style="width: 50%; padding: 10px; border: none;">
      <img src="assets/compressed_faster.gif" alt="Pruned" style="width: 100%;">
    </td>
  </tr>
</table>

+ MaskRCNN for instance segmentation
<table style="width: 100%; border: none; border-collapse: collapse;">
  <tr>
    <td style="width: 50%; padding: 10px; border: none;">
      <img src="assets/baseline_mask.gif" alt="Baseline" style="width: 100%;">
    </td>
    <td style="width: 50%; padding: 10px; border: none;">
      <img src="assets/compressed_mask.gif" alt="Pruned" style="width: 100%;">
    </td>
  </tr>
</table>

+ KeypointRCNN for human keypoint detection
<table style="width: 100%; border: none; border-collapse: collapse;">
  <tr>
    <td style="width: 50%; padding: 10px; border: none;">
      <img src="assets/baseline_keypoint.gif" alt="Baseline" style="width: 100%;">
    </td>
    <td style="width: 50%; padding: 10px; border: none;">
      <img src="assets/compressed_keypoint.gif" alt="Pruned" style="width: 100%;">
    </td>
  </tr>
</table>

<div align="center">
    Baseline (<em>left</em>) vs Compressed (<em>right</em>) model inference.
</div>

To underscore the practical advantages of NORTON, an experiment was meticulously conducted, involving a direct comparison between a baseline model and a compressed model, both tailored for object detection tasks. Leveraging the FasterRCNN_ResNet50_FPN architecture on a RTX 3060 GPU, the experiment robustly highlights the substantial performance enhancement achieved by NORTON. The accompanying GIFs offer a vivid visual depiction: the baseline model showcases an inference speed of approximately 9 FPS, while the NORTON-compressed model boasts a remarkable twofold acceleration in throughput. This notable disparity effectively showcases NORTON's efficacy and scalability, firmly establishing its relevance and applicability across diverse deployment scenarios.

*Note*: For replication of this experiment, please refer to [detection/README.md](detection/README.md).
