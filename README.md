

# Weakly-Supervised Semantic Segmentation of Airborne LiDAR Point Clouds

This repository applied **Semantic Query Network (SQN)** to classify the Airborne LiDAR Point Clouds. The SQN only requires annotating less than 0.1% of raw points and allows great error tolerance to decide the annotation in boundary areas for model training, which largely saves manpower and time to prepare training data for large-scale ALS datasets.
The figure shows an example of point cloud classification result.

![sqn_result](https://github.com/user-attachments/assets/0dd99fa3-f7df-4f72-8deb-4a3f426e24ac)

## Usage notes
This code for point cloud classification has been implemented with Python 3.6, TensorFlow 1.11.0, CUDA 9.0, and cuDNN 7.4.1 on Ubuntu 18.04.6. Following instruction of [SQN](https://github.com/QingyongHu/SQN?tab=readme-ov-file) to set up. The pretrain model using Glasgow city point cloud can be found in the [model](https://github.com/QiaosiLi/SQN_ALS_Classification/tree/main/models) folder.

## Dataset
### Glasgow Annotated Airborne LiDAR Point Clouds 
We prepared a set of training and validation data to classify the whole LiDAR dataset. Four tiles of 1×1 km2 ALS point clouds were labelled for SQN model training. Training data cover diverse landscape, which include the historical, modern buildings, common residential, stylish building complex, planted trees, and semi-natural woodlands. Four tiles of 0.5×0.5 km2 covering commercial, residential, industrial area, and modern building complex were full point annotated. Our annotated point clouds and the training and validation input data that are ready to feed into the SQN model are published in [Urban Big Data Centre data catalogue](https://data.ubdc.ac.uk/datasets/glasgow-3d-city-models-derived-from-airborne-lidar-point-clouds-licensed-data).

The annotated point cloud data can be used to train a deep learning model for point cloud classification or help advance the manipulation within airborne LiDAR. 

### Hong Kong Annotated Airborne LiDAR Point Clouds 
The annotated point clouds were generated to train the weakly supervised semantic segmentation algorithm Semantic Query Network (SQN) to classify point clouds. The dataset covers 16 tiles of airborne LiDAR data in an area of 7.2 km2  in Shatin, Hong Kong, China. 11 tiles were used for training, while 5 tiles were used for validation. There are multiple types of construction in the dataset including high-rise residential buildings, low-rise village houses, and large public buildings. Green spaces are mainly composed of wood areas in open spaces (e.g., in parks and hills) and planted trees in residential gardens and nearby roads. Point clouds are classified in ground, buildings, and trees. This dataset is published in [Zenodo](???).

The LiDAR data is owned by the Hong Kong government. Please visit the [Spatial Data Portal, Survey Division, CEDD](https://sdportal.cedd.gov.hk/#/en/) for more details.

## Citation

If you find our work useful in your research, please consider citing:

	@misc{https://doi.org/10.20394/vwyl2on6,
		doi = {10.20394/VWYL2ON6},
		url = {https://data.ubdc.ac.uk/dataset/8bccf530-0f07-4ff3-a8d5-443328fcd415},
		author = {{Urban Big Data Centre}},
		keywords = {Urban Planning},
		language = {en},
		title = {Glasgow 3D city models derived from airborne LiDAR point clouds licensed data},
		publisher = {University of Glasgow},
		year = {2024}
	}
	
	@INPROCEEDINGS{10144215,
		author={Li, Qiaosi and Zhao, Qunshan},
		booktitle={2023 Joint Urban Remote Sensing Event (JURSE)}, 
		title={Weakly-Supervised Semantic Segmentation of Airborne LiDAR Point Clouds in Hong Kong Urban Areas}, 
		year={2023},
		volume={},
		number={},
		pages={1-4},
		keywords={Point cloud compression;Solid modeling;Laser radar;Three-dimensional displays;Annotations;Semantic segmentation;Atmospheric modeling;Airborne LiDAR;Point cloud classification;Urban buildings and trees;Deep learning},
		doi={10.1109/JURSE57346.2023.10144215}
	}


## Related Repos

1. [SQN: Weakly-Supervised Semantic Segmentation of Large-Scale 3D Point Clouds](https://github.com/QingyongHu/SQN)
2. [construct_building_tree_3d_models_by_lidar](https://github.com/QiaosiLi/construct_building_tree_3d_models_by_lidar)




