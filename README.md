# Probabilistic U-Net    
now, in production....

# discription  
 Implementation of probabilistic U-Net model using tensorflow.   If any bug, please send me e-mail.  
 
# literature  
 [probabilistic U-Net](https://arxiv.org/abs/1806.05034)  

# dependency  
I confirmed operation only with..   
1)python 3.6.3  
2)tensorflow 1.7.0  
3)numpy 1.14.2    
4)Pillow 4.3.0  

# TODO  
metrics and etc....  

# email  
t.ohmasa@w-farmer.com  

# result image  
after 100 epochs, I have predicted CityScape test datasets.   
![resultimage_181019mod04_p20_val__150](https://user-images.githubusercontent.com/15444879/47248181-95726680-d443-11e8-96cf-d2b4fb5a0d5a.png)  
1st row: original image(cityScape datasets test images).  
2nd row: labels.  
3rd row: prediction.  

and, predict 1 test image changing latent z.  

![resultimage_181019mod04_p20_diff_z__150](https://user-images.githubusercontent.com/15444879/47248191-9c997480-d443-11e8-97a3-e03b211cb1a2.png)  

