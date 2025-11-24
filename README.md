The official implementation of "Data-Efficient Segmentation of Scanning Probe Microscopy Images via an Enhanced Large Vision Model for Materials Characterization"

Before training, please download the SAM checkpoints from [**here**](https://github.com/facebookresearch/segment-anything#model-checkpoints), and replace the **sam_checkpoint** in **yaml** file.

If you try to run this project, refer to the file train_MAML_new_bs4.py, whose name means "the latest version of trainning code, and the batch size(bs) is 4".

Using the following structure to organize the data: 

<pre>
Dataset

--LiFePO4  
--LiCoO2  
  --support_<strong>train</strong>  
    --img  
      --1.jpg  
      --2.jpg   
      --3.jpg  
      --4.jpg  
      ...
    --mask  
      --1.png  
      --2.png  
      --3.png  
      --4.png  
      ...
  --query_<strong>train</strong>  
    --img  
      --11.jpg  
      --12.jpg
      --13.jpg  
      --14.jpg  
      ...
    --mask  
      --11.png  
      --12.png
      --13.png  
      --14.png 
      ...
  --support_eval  
    --(the same as training data organization)
  --query_eval  
    --(the same as training data organization)
--LiMn2O4  
--BiOCl
--...
</pre>  


The dataset **SPM-Seg** can be downloaded [**here**](https://drive.google.com/drive/folders/1er8rNT8MF8AoJxIf5fWHfkL3vejJtx66?usp=drive_link)


This repo is mainly built based on [**PFENet**](https://github.com/dvlab-research/PFENet) and [**SSM-SAM**](https://github.com/DragonDescentZerotsu/SSM-SAM/tree/main). Thanks for their great work!
