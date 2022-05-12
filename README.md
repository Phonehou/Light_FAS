# Two-stream_FAS
A two-stream framework for Face Anti-Spoofing 

The directories are shown below.
* **CAM**: visualization tool CAM(Class Activation Mapping)'s code
* **data_label**: save samples'path and corresponding labels in a json file
* **DSDG**: implement of DSDG(from paper *Dual Spoof Disentanglement Generation for Face Anti-spoofing with Depth Uncertainty Learning*
* **experiment**: train's code about RGB branch and PLGF branch
* **loss**: loss function about experiments
* **models**: neural network models about experiments 
* **TSNE**: visualization tool TSNE's code
* **utils**: utility about experiments, such as dataloader, performance metrics 
***
## CAM
* *main_cam.py*: generate CAM for a image or a series of images

|parameters|description|example|
|:---|:---|:---|
|network|trained model's path|/DISK3/houjyuf/workspace/Adv_FAS_v2/experiment/Vlad_plgf/CASIA2Replay_checkpoint/C2R_v2/best_model/model_best_0.03564_03960.pth.tar|
|image-path|image's directory|/home/users/houjyuf/workspace/Adv_FAS/CAM-Grad/examples/CASIA_plgf|
|weight-path|network, weight's path,default None, specially for preset model(e.g.,ResNet, VGG16)|None|
|output-dir|output directory to save results|VLAD_C2R_intra|
|dst_dataset|the name of dataset for testing|CASIA|
|postfix|dataset's postfix|mask_mini|

* *grad_cam.py*: definition of class GradCAM and class GradCamPlusPlus
* *guided_back_progagation.py*: definition of GuidedBackPropagation
***
## data_label
* *generate_grg_label.py*: save grg(广电运通) samples'path and corresponding labels in a json file

* *generate_mask_label.py*: save public FASD(Face Anti-Spoofing Database) samples'path and corresponding labels in a json file

This script can generate these json files under each given output directory:

1. all_label.json 
2. train_label.json
3. train_real.json
4. train_fake.json
5. devel_label.json
6. devel_real.json
7. devel_fake.json
8. test_label.json
9. test_real.json
10. test_fake.json
11. real_label.json
12. fake_label.json

Each item in a json file include four terms: **photo_path, photo_label, photo_belong_to_video_ID, mask_path**

|parameters|description|example|
|:---|:---|:---|
|data_dir|data directory|/DISK3/rosine/cross_dataset/frames_crop/msu_crop|
|mask_dir|images' corresponding PLGF's directory|/DISK3/houjyuf/dataset/ros_plgf/msu|
|dataset|name of dataset|msu|
|save_dir|output directory to save json file|msu_aug200|
|fake_internal|fake samples' sampled interval,default 1| 1|
|train_internal|training set samples' sampled interval,default 1| 1|
|devel_internal|validation set samples' sampled interval,default 1| 1|
|test_internal|testing set samples' sampled interval,default 1| 1|
***
## DSDG
* *train_generator.py*: train the generator, can run by *train_generator.sh*, parameters' description can be seen in *train_generator.sh*

* *generated.py*: generate additional samples by trained generator
***
## loss
* *AdLoss.py*: Adversarial loss function in paper *Single-Side Domain Generalization for Face Anti-Spoofing*

* *hard_triplet_loss.py*: hard triplet loss function used in PLGF branch's supervised learning

* *Self Contrastive Loss.py*: Self contrastive loss function
***
## models 
* *CAM.py*: attention mechanism's implement about paper *CBAM: Convolutional Block Attention Module*
* *encoder.py*: JPEG encoder's suppress implement 
* *generator.py*: DSDG's network implement 
* *light_cnn.py*: DSDG's additional discrimination network implement
* *SGTD.py*: SGTD's network implement about paper *Deep Spatial Gradient and Temporal Depth Learning for Face Anti-spoofing*
* *SGTD_single.py*: SGTD's network for single frame implement about paper *Deep Spatial Gradient and Temporal Depth Learning for Face Anti-spoofing*
* *mpl_resnet.py*: network implement about RGB branch
* *spoofNet.py*: network implement about PLGF branch
* *spoofNet_rsgb.py*: network implement about PLGF branch with rsgb(Resnet Spatial Gradient Block)
* *test_model.py*: network's structure and parameters amount evaluate module
***
## TSNE
* *generate_npy.py*: generate npy file for data

* *tsne_main.py*: generate tsne for npy file
***
## utils
* *augmentation.py*: data augmentation implement for UDA
* *dataset.py*: data handler class implement for experiment
* *statistic.py*: performance metrics for experiments

* *data_mpl.py*: data loader for RGB branch
* *utils_mpl.py*: utility for training RGB branch, such as loss function

* *plgf_generate.py*: generate PLGF for images
* *get_loader.py*: data loader for PLGF branch
* *utils.py*: utility for training PLGF branch, such as loss function

* *DSDG_dataset.py*: data handler class implement for DSDG
* *DSDG_utils.py*: utility for training DSDG, such as loss function

* *metrics.py*: performance metrics by paper *Searching Central Difference Convolutional Networks for Face Anti-Spoofing*
* *visual.py*: visualization for experiment

* *Huffman.py*: Huffman-encoder implement for JPEG
* *utils_jpeg.py*: utility for JPEG compress
***
## experiment
### Meta_Pseudo_Label
* *main.py*: RGB branch experiment with WideResNet under MPL+UDA

* *mpl_wtUDA.py*: RGB branch experiment with WideResNet under MPL

* **main.sh** can run the main.py and mpl_wtUDA.py*
### Pseudo_Label
* *main.py*: RGB branch experiment with WideResNet under PL

### WideResNet
* *main.py*: RGB branch experiment with WideResNet

### DG
* *main.py*: RGB branch DG(Domain Generalization) experiment with WideResNet under MPL+UDA
* *train_map.py*: RGB branch DG experiment with WideResNet_map under MPL+UDA

* **main.sh** can run the main.py and mpl_wtUDA.py*
### my_rsgb 
* *mpl_map.py*: RGB branch experiment with WideResNet_map under MPL+UDA

### Fusion
* *main.py*: Given DSDG's generated data, RGB branch experiment with WideResNet under MPL+UDA
* *main_map.py*: Given DSDG's generated data, RGB branch experiment with WideResNet_map under MPL+UDA

### Vlad_plgf
* *main.py*: PLGF branch experiment with SpoofNet/SpoofNet_MPL
* *main_DG.py*: PLGF branch DG experiment with SpoofNet/SpoofNet_MPL

### SGTD_single
* *main_rsgb.py*: PLGF branch experiment with SpoofNet_rsgb/SpoofNet_rsgb+att 
* *main_CIM.py*: PLGF branch CIM experiment with SpoofNet_rsgb/SpoofNet_rsgb+att 

### SSDG_spoofNet
* *main_ssdg.py*: PLGF branch experiment with SpoofNet, under SSDG(paper:Single-Side Domain Generalization for Face Anti-Spoofing) framework

### Two-Stream
* *main.py*: train RGB branch and PLGF branch at the same time(Need to be improved)
* *main_two.py*: train RGB branch and PLGF branch at the same time, with class balance(Need to be improved)
***
# How to run this project
RGB branch and PLGF branch can train independently.
## 1. dataset preparation
If wanna ready-made croped frames of public FAS dataset(CASIA, Replay, msu, Oulu), can access diretory show in *Database Introduction.pdf*.
Their corresponding PLGF frames all in diretory '/DISK3/houjyuf/dataset/ros_plgf/[dataset]'   

## 2. save samples'path and corresponding labels as json file
Generate the diretory of json file under data_label.If necessary, the ready-made json files can be access in '/DISK3/houjyuf/workspace/Adv_FAS_v2/data_label/' 
```shell script
cd data_label 
python generate_mask_label.py --data_dir your data diretory --mask_dir your PLGF diretory --dataset name of dataset --save_dir output diretory(relative path)
```

## 3. conduct the experiment
Your can follow above description to conduct certain experiment.
For example, for RGB branch experiment with WideResNet under MPL+UDA.
If occur the error as *'bash: ./main.sh: /bin/bash^M: 解释器错误: 没有那个文件或目录
'*, run the command *sed -i 's/\r$//' xxxxxxx.sh* can solve.
```shell script
cd experiment/Meta_Pseudo_Label
./main.sh  # or python main.py --parameters xxx
## if their is no main.sh, run the python script and add corresponding parameters 
```

