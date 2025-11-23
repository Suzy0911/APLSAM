# APLSAM
The official implementation of "Data-Efficient Segmentation of Scanning Probe Microscopy Images via an Enhanced Large Vision Model for Materials Characterization"

If you try to run this project, refer to the file **train_MAML_new_bs4.py**, whose name means "the latest version of trainning code, and the batch size(bs) is 4".  

 
Using the following structure to organize the data:
e.g. ------
<pre>
Dataset

--LiFePO4  
--LiCoO2  
  --support_<strong>eval</strong>  
    --img  
      --3.jpg  
      --3.jpg   
      --3.jpg  
      --3.jpg  
      ------------  (virtual divider, do not really exist in the folder)
      --8.jpg
      ...
    --mask  
      --3.png  
      --3.png  
      --3.png  
      --3.png  
      ------------
      --8.png
      ...
  --query_<strong>eval</strong>  
    --img  
      --1.jpg  
      --2.jpg (<strong>notice</strong> : slice no.3 and no.8 are used as the support image in the chunk)  
      --4.jpg  
      --5.jpg  
      ------------
      --6.jpg
      --7.jpg
      --9.jpg
      --10.jpg
      ------------
      ...
    --mask  
      --1.png  
      --2.png (<strong>notice</strong> : slice no.3 and no.8 are used as the support image in the chunk)  
      --4.png  
      --5.png 
      ------------
      --6.jpg
      --7.jpg
      --9.jpg
      --10.jpg
      ------------
      ...
  --support_train  
    --(the same as evaluation(eval) data organization)
  --query_train  
    --(the same as evaluation(eval) data organization)
--LiMn2O4  
--BiOCl
--...
</pre>  
