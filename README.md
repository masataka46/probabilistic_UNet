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
![resultimage_18101801_val__0](https://user-images.githubusercontent.com/15444879/47130272-dcd9e500-d2d3-11e8-8d5e-19c487a301f7.png)  
1st row: original image(cityScape datasets test images).  
2nd row: labels.  
3rd row: prediction.  

and, predict 1 test image changing latent z.  

![resultimage_18101801_diff_z__0](https://user-images.githubusercontent.com/15444879/47130282-e400f300-d2d3-11e8-981d-7f10fa52673c.png)  
but...result is not desirable....  
