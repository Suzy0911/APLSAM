The official implementation of "Data-Efficient Segmentation of Scanning Probe Microscopy Images via an Enhanced Large Vision Model for Materials Characterization"

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
