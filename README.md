# DL_projecct , 2023 

Sapir & Racheli

this ripository contains several different models that compare between different shapes of cutouts, transparncy in changes levels of cutouts and different implmentations of data augmentation in aim to find the best model that will benefit architecture of ResNet18. 

models were run over CIFAR-10 dataset for object detection (classification) task 


<u> to run the models follow those steps: </u>
use GPU 

1. run the commands:
	 !git clone https://github.com/rache18/DL_projecct.git <br>
	 %cd /content/DL_projecct <br>
	 !mkdir checkpoints

2. Run the command train.py after setting the args of the model you want implement: 

	cutout_options = ['None', 'Cutout', 'Cutout_intesity', 'Cutout_Shape', 'Cutout_intensity_shapes']
	shape_options = ['square', 'circle', 'triangle']

	to use data augmentation add the arg --data_augmentation to the command.
	
	choose length of shape ( 16 is optimal for square )

	choose number of epochs to run

	best model: !python train.py --dataset cifar10 --data_augmentation --model resnet18 --cutout Cutout_Shape --shape square --length 16 --epochs 150


*  data augmentation was defined to be the combination who got best results : 
   RandomCrop(32, padding=4)
   RandomHorizontalFlip()

results: 
will be printed on terminal and cvs.file will be created inside "logs" folder 

*all different models is inside "utils" folder*

* "csv models 150 epochs" folder contains csv files with results of models that have been run

* "referens" folder contains the paper of the project 

* "plots" folder contains plot.py to ploting each csv file, and plots for example
