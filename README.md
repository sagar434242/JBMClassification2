
# Classifying Healthy Defected parts

## Guide to use Project

#### Unzip the contents of JBMData (Drive Link) attached with mail inside `./JBMClassificatrion2/`

###  Training Classifier
The folder  './JBMClassification2' contains subfolder YE358311_Healthy & YE358311_defects which has dataset
1.	Open playingwithmodel.py
2.	Cd to JBMClassification2 
3.	Make sure to modify the path to your local dir in Line No. 43
4.	Now the Classifier can be trained by executing the below command.
	4.1.	python playingwithmodels.py
5.	Saved network at the end or training is generated with file name JBM_Classification

###  Testing the model
Already trained model present in `./JBMClassificatrion2/`  (with name JBM_Classification.h5 )directory can be used for testing on new images. Execute the below command.
1. Open test_model.py
2. Cd to JBMClassification2
3. Modify the path for trained_model_path (`./JBMClassificatrion2/`JBM_Classification.h5), incept_model_path(`./JBMClassificatrion2/incept_model.h5`  ) Line No. 31 & 32
4. Make a Test folder inside the JBMClassification2 and keep test images(Healthy/Defective) inside this folder with image name being test.img Line No. 33 in test_model.py

OR 
```
python ./test_model.py \
--images_path={Path of the Image file or Folder of images} \
--trained_model_path = ./ JBM_Classification.h5\
--incept_model_path = ./incept_model.h5
```

The trained model achieved **~93.55** % accuracy with following parameters :- 
- Epochs = 150
- Batch size = 32
- Learning Rate = 1e-5 (RMSprop)

<div align="center">
<a href="https://imgflip.com/i/30cbgd"><img width="864" height="288" src="https://i.imgflip.com/30cbgd.jpg" title="made at imgflip.com"/></a>
</div>

### Deploying App to Smarthone and Cloud
In `./JBMClassificatrion2/` we have following video demo.
1. `QualityTesterByJBM` A Demo of AI model deployed on the android App
2. `GuidetoTestNetwork` Guide on how to use test_model.py to predict from saved network.
3. `TraningRecord`  Screen Capture while traning, Accuracy reaches to 93.55% towards epoch 150
