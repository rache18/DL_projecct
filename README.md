# DL_projecct

Sapir & Racheli


<u> things we want to change : </u>
1. add augmentation
	we now implement diefferent augmentations combinations whit marking each augmentation as note/command on the 	train.py itself. consider add "augmentation_options" such as "shape_options" to pick combination when running the 	model

2. change number of cutouts in an image ( instead of 1 , we will make 2/3 .. ), change shape, change inensity -> done with holes 
3. contact encoder 
4. dropout for cnn (spatialDropout) 

* we need to check any of those 4 options seperattly and then check combinations of them. 

* finally, we can show that the best model is good for other models ( wide resnet, others.. ) 


<u> about augmentation taken: </u>
RandomHorizontalFlip: Flips the image horizontally with a probability of 0.5. This augmentation helps the model become invariant to horizontal flips, which is a common transformation in images.

RandomCrop: Randomly crops a portion of the image. This augmentation helps the model learn translation invariance and increases robustness.

RandomRotation: Rotates the image by a random angle within a specified range. This augmentation helps the model become invariant to rotations.

Normalize: Normalizes the image by subtracting the mean and dividing by the standard deviation. This normalization helps in reducing the scale differences between features and stabilizes the training process.

ColorJitter: Adjusts the brightness, contrast, saturation, and hue of the image by random amounts. This augmentation adds variation to the color space, making the model more robust to different lighting conditions.

