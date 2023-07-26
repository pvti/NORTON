<p align="center" width="100%">
<img src="assets\NORTON-Logo.png"  width="80%" height="80%">
</p>

<div>
<div align="center">
    <a href='https://github.com/pvtien96' target='_blank'>Van Tien PHAM<sup>1,&#x2709</sup></a>&emsp;
    <a href='https://yzniyed.blogspot.com/p/about-me.html' target='_blank'>Yassine ZNIYED<sup>1</sup></a>&emsp;
    <a href='http://tpnguyen.univ-tln.fr/' target='_blank'>Thanh Phuong NGUYEN<sup>1</sup></a>&emsp;
</div>
<div>

<div align="center">
    <sup>1</sup>Université de Toulon, Aix Marseille Université, CNRS, LIS, UMR 7020, France&emsp;
    <sup>&#x2709</sup> Corresponding Author
</div>

-----------------
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fpvtien96%2FNORTON&countColor=%23263759)

### 🌟 Enhanced network compression through tensor decompositions and pruning
In this work, we propose NORTON (enhanced Network cOmpRession through TensOr decompositions and pruNing), a novel method for network compression. NORTON introduces the concept of filter decomposition, enabling a more detailed decomposition of the network while preserving the weight's multidimensional properties. Our method incorporates a novel structured pruning approach, effectively integrating the decomposed model. Through extensive experiments on various architectures, benchmark datasets, and representative vision tasks, we demonstrate the usefulness of our method. NORTON achieves superior results compared to state-of-the-art techniques in terms of complexity and accuracy.

<div>
  <img class="image" src="assets\OverallFramework.png" width="51%" height="100%">
  <img class="image" src="assets\OneLayer.png" width="48%" height="100%">
</div>
<div align="center ">
    Left: Graphic illustration of the NORTON approach. Right: The decomposition and then pruning process for one layer.
</div>



<p align="left">
<img src="assets\CPDBlock.png" width=100%>
</p>
<div align="center ">
    Illustration of the CPDBlock structure for classical deep learning frameworks
</div>


## 🚩🚩🚩 Main results 


<div align="center ">
  <img class="image" src="assets\Vgg16-CIFAR10.png" width="40%" height="100%">
</div>
<div align="center ">
    The accuracy-MACs reduction Pareto curves of compressed VGG-16 models are compared on CIFAR-10.
</div>


In order to demonstrate the adaptability of NORTON, we assess three representative architectures: VGG-16-BN, ResNet-56/110 with residual blocks, and DenseNet-40 with dense blocks. These models are tested on the CIFAR-10 dataset. Additionally, to validate the scalability of NORTON, experiments are conducted on the challenging ImageNet dataset using the ResNet-50 architecture. Furthermore, the compressed ResNet-50 model is employed as the backbone network for FasterRCNN-FPN, MaskRCNN, and KeypointRCNN on the COCO-2017 dataset.

NORTON is compared with the SOTA in the fields of low-rank decompositions, structured pruning, and hybrid methods. 

1. VGG-16-BN/CIFAR-10
<div align="center">

| Model                       | Top-1 (%)| MACs (↓%)     | Params. (↓%)  |
|-----------------------------|----------|---------------|---------------|
| VGG-16-BN                   | 93.96    | 313.73M (00)  | 14.98M (00)   |
| HRank-1                     | 93.43    | 145.61M (54)  | 2.51M (83)    |
| CHIP                        | 93.86    | 131.17M (58)  | 2.76M (82)    |
| EZCrop                      | 93.01    | 131.17M (58)  | 2.76M (82)    |
| DECORE-500                  | 94.02    | 203.08M (35)  | 5.54M (63)    |
| AutoBot                     | 94.19    | 145.61M (54)  | 7.53M (50)    |
| **NORTON (Ours)**           | **94.45**    | **126.49M (60)**  | **2.58M (83)**    |
| HRank-2                     | 92.34    | 108.61M (65)  | 2.64M (82)    |
| DECORE-200                  | 93.56    | 110.51M (65)  | 1.66M (89)    |
| EZCrop                      | 93.70    | 104.78M (67)  | 2.50M (83)    |
| CHIP                        | 93.72    | 104.78M (67)  | 2.50M (83)    |
| AutoBot                     | 94.01    | 108.71M (65)  | 6.44M (57)    |
| **NORTON (Ours)**           | **94.16**    | **101.91M (68)**  | **2.34M (84)**    |
| WhiteBox                    | 93.47    | 75.30M (76)   | N/A           |
| AutoBot                     | 93.62    | 72.60M (77)   | 5.51M (63)    |
| **NORTON (Ours)**           | **94.11**    | **74.14M (77)**   | **3.60M (76)**    |
| QSFM                        | 92.17    | 79.00M (75)   | 3.68M (75)    |
| DECORE-100                  | 92.44    | 51.20M (82)   | 0.51M (96)    |
| FSM                         | 92.86    | 59.61M (81)   | 1.50M (90)    |
| ALDS                        | 92.67    | 66.95M (86)   | 1.90M (96)    |
| Lebedev et al.              | 93.07    | 68.53M (78)   | 3.22M (78)    |
| EPruner-0.73                | 93.08    | 74.42M (76)   | 1.65M (89)    |
| HALOC                       | 93.16    | 43.92M (86)   | 0.30M (98)    |
| CHIP                        | 93.18    | 66.95M (79)   | 1.90M (87)    |
| ASTER                       | 93.45    | 60.00M (81)   | N/A           |
| FSM                         | 93.73    | 106.67M (66)  | 2.10M (86)    |
| **NORTON (Ours)**           | **93.84**    | **37.68M (88)**   | **1.94M (87)**    |
| HRank-3                     | 91.23    | 73.70M (77)   | 1.78M (92)    |
| DECORE-50                   | 91.68    | 36.85M (88)   | 0.26M (98)    |
| **NORTON (Ours)**           | **92.54**    | **13.54M (96)**   | **0.24M (98)**    |
| **NORTON (Ours)**           | **90.32**    | **4.58M (99)**    | **0.14M (99)**    |
</div>
2. ResNet-56/110/CIFAR-10
<div align="center">

| Model                   | Top-1(%) | MACs (↓%)    | Params. (↓%) |
|-------------------------|---------|--------------|--------------|
| ResNet-56               | 93.26   | 125.49M (00) | 0.85M (00)   |
| HRank-1                 | 93.52   | 88.72M (29)  | 0.71M (17)   |
| DECORE-450              | 93.34   | 92.48M (26)  | 0.64M (24)   |
| FilterSketch            | 93.65   | 88.05M (30)  | 0.68M (21)   |
| TPP                     | 93.81   | 86.59M (31)  | N/A          |
| WHC                     | 93.91   | 90.35M (28)  | N/A          |
| **NORTON (Ours)**       | **94.46** | 93.34M (27)  | **0.58M (31)** |
| HRank-2                 | 93.17   | 62.72M (50)  | 0.49M (42)   |
| FilterSketch            | 93.19   | 73.36M (41)  | 0.50M (41)   |
| DECORE-200              | 93.26   | 62.93M (50)  | 0.43M (49)   |
| TPP                     | 93.46   | 62.75M (50)  | N/A          |
| MFP                     | 93.56   | 59.40M (53)  | N/A          |
| FSM                     | 93.63   | 61.49M (51)  | 0.48M (44)   |
| CC-0.5                  | 93.64   | 60.00M (52)  | 0.44M (48)   |
| **NORTON (Ours)**       | **94.00** | 73.22M (42)  | **0.44M (48)** |
| QSFM                    | 91.88   | 50.62M (60)  | 0.25M (71)   |
| CHIP                    | 92.05   | 34.79M (72)  | 0.24M (72)   |
| TPP                     | 92.35   | 36.39M (71)  | N/A          |
| **NORTON (Ours)**       | **92.27** | 40.91M (68)  | **0.23M (72)** |
| HRank-3                 | 90.72   | 32.52M (74)  | 0.27M (68)   |
| DECORE-55               | 90.85   | 23.22M (81)  | 0.13M (85)   |
| FilterSketch            | 91.20   | 32.47M (74)  | 0.24M (72)   |
| **NORTON (Ours)**       | **91.62** | **14.47M (89)** | **0.08M (91)** |
| ResNet-110              | 93.50   | 256.04M (00) | 1.73M (00)   |
| DECORE-500              | 93.88   | 163.30M (35) | 1.11M (36)   |
| **NORTON (Ours)**       | **94.85** | **163.00M (35)** | **1.08M (38)** |
| DECORE-300              | 93.50   | 96.66M (62)  | 0.61M (65)   |
| **NORTON (Ours)**       | **94.11** | **92.99M (64)** | **0.59M (65)** |
| DECORE-175              | 92.71   | 58.37M (77)  | 0.35M (80)   |
| **NORTON (Ours)**       | **92.77** | **47.34M (82)** | **0.30M (83)** |
</div>
3. DenseNet-40/CIFAR-10
<div align="center">

| Model               | Top-1(%) | MACs (↓%)    | Params. (↓%) |
|---------------------|---------|--------------|--------------|
| DenseNet-40         | 94.81   | 282.92M (00) | 1.04M (00)   |
| DECORE-175          | 94.85   | 228.96M (19) | 0.83M (21)   |
| **NORTON (Ours)**   | **94.86** | **213.58M (26)** | **0.74M (30)** |
| HRank-1             | 94.24   | 167.41M (41) | 0.66M (37)   |
| DECORE-115          | 94.59   | 171.36M (39) | 0.56M (46)   |
| AutoBot            | 94.67   | 167.64M (42) | 0.76M (28)   |
| **NORTON (Ours)**   | **94.67** | **168.23M (42)** | 0.58M (45)   |
| HRank-2             | 93.68   | 110.15M (61) | 0.48M (54)   |
| EZCrop              | 93.76   | 113.08M (60) | 0.39M (62)   |
| DECORE-70           | 94.04   | 128.13M (55) | 0.37M (65)   |
| **NORTON (Ours)**   | **94.14** | 123.14M (58)** | **0.40M (62)** |
</div>
4. ResNet-50/Imagenet
<div align="center">

| Model                                 | Top-1 | Top-5 | MACs(↓%)    | Params(↓%)  |
|---------------------------------------|-------|-------|-------------|-------------|
| ResNet-50                             | 76.15 | 92.87 | 4.09G(00)   | 25.50M(00)  |
| ABCPruner-100%                        | 72.84 | 92.97 | 2.56G(37)   | 18.02M(29)  |
| CLR-RNF-0.2                           | 74.85 | 92.31 | 2.45G(40)   | 16.92M(34)  |
| EPruner-0.81                          | 74.95 | 92.36 | 2.37G(42)   | N/A         |
| FilterSketch-0.7                     | 75.22 | 92.41 | 2.64G(36)   | 16.95M(33)  |
| Kim et al.                            | 75.34 | 92.68 | N/A         | 17.60M(31)  |
| PFP                                   | 75.91 | 92.81 | 3.65G(11)   | 20.88M(18)  |
| LeGR                                  | 76.20 | 93.00 | 2.99G(27)   | N/A         |
| DECORE-8                              | 76.31 | 93.02 | 3.54G(13)   | 22.69M(11)  |
| CHIP                                  | 76.30 | 93.02 | 2.26G(44)   | 15.10M(41)  |
| TPP                                   | 76.44 | N/A   | 2.74G(33)   | N/A         |
| **NORTON (Ours)**                     | **76.91** | **93.57** | **2.32G(43)** | **14.51M(43)** |
| FilterSketch-0.6                     | 74.68 | 92.17 | 2.23G(46)   | 14.53M(43)  |
| Hinge                                 | 74.70 | N/A   | 2.17G(47)   | N/A         |
| HRank-1                               | 74.98 | 92.33 | 2.30G(44)   | 16.15M(37)  |
| DECORE-6                              | 74.58 | 92.18 | 2.36G(42)   | 14.10M(45)  |
| PFP                                   | 75.21 | 92.43 | 2.29G(44)   | 17.82M(30)  |
| WhiteBox                              | 75.32 | 92.43 | 2.22G(46)   | N/A         |
| EZCrop                                | 75.68 | 92.70 | 2.26G(45)   | 15.09M(41)  |
| LeGR                                  | 75.70 | 92.70 | 2.37G(42)   | N/A         |
| DepGraph                              | 75.83 | N/A   | 2.09G(49)   | N/A         |
| SCOP                                  | 75.95 | 92.79 | 2.24G(45)   | 14.59M(43)  |
| CATRO                                 | 75.98 | 92.79 | 2.21G(46)   | N/A         |
| WHC                                   | 76.06 | 92.86 | 2.37G(42)   | N/A         |
| CHIP                                  | 76.15 | 92.91 | 2.10G(49)   | 14.23M(44)  |
| **NORTON (Ours)**                     | **76.58** | **93.43** | **2.08G(50)** | **13.51M(47)** |
| HRank-2                               | 71.98 | 91.01 | 1.55G(62)   | 13.77M(46)  |
| FilterSketch-0.4                     | 73.04 | 91.18 | 1.51G(63)   | 10.40M(59)  |
| WhiteBox                              | 74.21 | 92.01 | 1.50G(63)   | N/A         |
| EZCrop                                | 74.33 | 92.00 | 1.52G(63)   | 11.05M(57)  |
| DAIS                                  | 74.45 | 92.21 | 1.83G(55)   | N/A         |
| CC-0.6                                | 74.54 | 92.25 | 1.53G(63)   | 10.58M(59)  |
| Phan et al.                           | 74.68 | 92.16 | 1.56G(62)   | N/A         |
| MFP                                   | 74.86 | 92.43 | 1.88G(54)   | N/A         |
| TPP                                   | 75.12 | N/A   | 1.60G(61)   | N/A         |
| SCOP                                  | 75.26 | 92.53 | 1.86G(55)   | 12.29M(52)  |
| CHIP                                  | 75.26 | 92.53 | 1.52G(63)   | 11.04M(57)  |
| **NORTON (Ours)**                     | **75.95** | **92.91** | **1.49G(64)** | **10.52M(59)** |
| HRank-3                               | 69.10 | 89.58 | 0.98G(76)   | 8.27M(68)   |
| DECORE-5                              | 72.06 | 90.82 | 1.60G(61)   | 8.87M(65)   |
| ABCPruner-50%                         | 72.58 | 90.91 | 1.30G(68)   | 9.10M(64)   |
| CHIP                                  | 72.30 | 90.74 | 0.95G(77)   | 8.01M(69)   |
| CLR-RNF-0.44                          | 72.67 | 91.09 | 1.23G(70)   | 9.00M(65)   |
| EPruner-0.81                          | 72.73 | 91.01 | 1.29G(68)   | N/A         |
| **NORTON (Ours)**                     | **74.00** | **92.00** | **0.96G(77)** | **7.96M(69)** |
| FilterSketch-0.2                     | 69.43 | 89.23 | 0.93G(77)   | 7.18M(72)   |
| DECORE-4                              | 69.71 | 89.37 | 1.19G(71)   | 6.12M(76)   |
| CURL                                  | 73.39 | 91.46 | 1.11G(73)   | 6.67M(74)   |
| **NORTON (Ours)**                     | **73.65** | **91.64** | **0.92G(78)** | **5.88M(77)** |
</div>
5. Faster/Mask/Keypoint-RCNN/COCO-2017
<div align="center">

| Model                               | AP<sup>0.5:0.95</sup> | AP<sup>0.5</sup> | AP<sup>0.75</sup> | AR<sup>1</sup> | AR<sup>10</sup> | AR<sup>100</sup> | MACs(↓%)     | Params(↓%)    |
|-------------------------------------|-----------------|------------|-------------|-------|-----------|------------|--------------|--------------|
| *FasterRCNN*                        | 0.37            | 0.58       | 0.39        | 0.31  | 0.48      | 0.51       | 134.85G(00)  | 41.81M(00)   |
| **NORTON (Ours)**                   | **0.37**       | **0.58**  | **0.39**     | **0.31** | **0.49**  | **0.51**     | 111.47G(17)  | 30.72M(27)   |
| **NORTON (Ours)**                   | 0.32            | 0.52       | 0.34        | 0.29  | 0.46      | 0.48       | **93.39G(31)** | **22.01M(47)** |
| *MaskRCNN*                          | 0.34            | 0.55       | 0.36        | 0.29  | 0.45      | 0.47       | 134.85G(00)  | 44.46M(00)   |
| **NORTON (Ours)**                   | **0.35**       | **0.57**  | **0.37**     | **0.30** | **0.46**  | **0.48**     | 111.47G(17)  | 33.36M(25)   |
| **NORTON (Ours)**                   | 0.32            | 0.52       | 0.33        | 0.28  | 0.44      | 0.46       | **93.39G(31)** | **24.65M(45)** |
|                                     |                 |            |             | AR<sup>0.5:0.95</sup> | AR<sup>0.5</sup> | AR<sup>0.75</sup> |              |              |
| *KeypointRCNN*                      | 0.65            | 0.86       | 0.71        | 0.71  | 0.90      | 0.77       | 137.42G(00)  | 59.19M(00)   |
| **NORTON (Ours)**                   | **0.65**       | **0.86**  | **0.71**     | **0.71** | **0.91**  | **0.77**     | 114.04G(17)  | 48.10M(19)   |
| **NORTON (Ours)**                   | 0.63            | 0.85       | 0.69        | 0.69  | 0.90      | 0.75       | **95.97G(30)** | **39.39M(34)** |
</div>


# 🌟 News
* **2023.7.26:** Paper submitted to IEEE TNNLS. The code will be gradually released. Stay tuned for more exciting updates!⌛

# 🔓 Verification and Reproducibility

Baseline and compressed models will be uploaded soon.

# 🕙 ToDo
- [ ] Write detailed documentation.
- [ ] Upload compressed models.
- [ ] Clean code.

# Contact
 In this work, we have proposed NORTON, a novel network compression method. This hybrid approach enjoys the advantages of both tensor decomposition and structured pruning by combining them orthogonally. Its superiority is demonstrated through various experiments and analyses. We introduce the concept of filter decomposition for the first time in the literature, along with the discrimination of approach in tensor decomposition which was ambiguous before. Besides, the novel distance-based structured pruning algorithm is developed and proved to incorporate well with the decomposition phase. We hope that the new perspective of NORTON and its template may inspire more developments 🚀🚀🚀 on network compression via filter decomposition and hybrid approach.

Your contributions can play a significant role in this endeavor, and we warmly welcome your participation in our project!

To contact us, never hesitate to send an email to [pvtien96@gmail.com](mailto:pvtien96@gmail.com) (for technical problems), [zniyed@univ-tln.fr](mailto:zniyed@univ-tln.fr) (for tensor decompositions), or [tpnguyen@univ-tln.fr](mailto:tpnguyen@univ-tln.fr) (for general development)!
<br></br>

# Citation
If the code and paper help your research, please kindly cite:
```
@misc{pham2023norton,
    title={Enhanced network compression through tensor decompositions and pruning}, 
    author={Van Tien, Pham and Yassine, Zniyed and Thanh Phuong, Nguyen},
    year={2023},
    howpublished={\url{https://github.com/pvtien96/NORTON}},
  }
```
# License
This project is released under the [Apache 2.0 license](LICENSE).
# Acknowledgement
This code is developed based on excellent open-sourced projects including [Torchvision](https://github.com/pytorch/vision), [HRankPlus](https://github.com/lmbxmu/HRankPlus/tree/master), and [Tensor Decompositions](https://github.com/mostafaelhoushi/tensor-decompositions).
