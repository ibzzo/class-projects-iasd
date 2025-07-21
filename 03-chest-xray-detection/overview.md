## Overview
Chest X-ray Detection Challenge
Your task is to develop a baseline object detector for pathological findings on chest X-rays.
You will use a subset of the NIH ChestXray dataset annotated with bounding boxes.

This is a real-world medical dataset — focus on building a solid pipeline and a good baseline detector!

Description
📂 Data Description
The dataset consists of:

Images: Chest X-ray images in PNG format
Annotations: Bounding boxes around pathological findings
Provided in a CSV file (train.csv) with:
-- Image Index: Filename
-- Finding Label: Type of finding
-- Bbox [x, y, w, h]: Bounding box coordinates
A stratified train split has already been created for you.

🎯 Task
Build an object detection model that can predict the bounding boxes of the findings.

Multiclass detection: one class per pathology

🛠️ Requirements
Deliverables:

A training pipeline

A basic model that performs detection

Evaluation metrics (e.g., mAP, IoU, recall)

Visualizations of predictions

A short notebook explaining your approach

🧠 Recommendations
Start simple (e.g., Faster R-CNN, YOLOv5)

If results are poor, merge all findings into a single class

Focus on clarity and reproducibility rather than achieving SOTA

Feel free to augment with additional data sources

Bonus: Propose possible improvements

Submissions ⚠️
You must submit your predictions in CSV format.
This file contains the predictions of your model with the bounding boxes as well as the confidence score.

⚠️ Each prediction must be associated with a unique id, not the image filename. A mapping between image_id and id is provided — make sure to match your predictions using this table. Submissions with incorrect or missing id values will not be evaluated.

Required format
Column	Type	Description
id	int	Unique identifier matching each image (see provided mapping)
image_id	string	Name of the image file
x_min	float	X coordinate of the top-left corner of the bounding box
y_min	float	Y coordinate of the top-left corner of the bounding box
x_max	float	X coordinate of the bottom-right corner of the bounding box
y_max	float	Y coordinate of the bottom-right corner of the bounding box
confidence	float	Confidence score of the prediction (ranging from 0 to 1)
label	string	Predicted class label for this bounding box
\

📅 Timeline
Start Date: TBD

End Date: TBD

📄 License and Attribution
This competition uses data from the NIH ChestX-ray14 dataset, made publicly available by the NIH Clinical Center.

The original dataset is hosted here: https://nihcc.app.box.com/v/ChestXray-NIHCC
ChestXray Dataset Folder: https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345

If you use this data, please cite the following paper:

Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017).
ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3462–3471.

The NIH Clinical Center is the original provider of the data.

📈 Evaluation Metric
Submissions will be evaluated based on Mean Average Precision (mAP) at an Intersection over Union (IoU) threshold of 0.5.



## Dataset Description

📁 Dataset Description
The dataset is based on the NIH ChestXray14 dataset, a large collection of chest X-ray images annotated with common thoracic pathologies.

📦 Folder Structure
The dataset is organized as follows:

train_images/       # Training images
test_images/        # Test images
train.csv            # Ground truth annotations for training images
📄 train.csv
This CSV file contains bounding box annotations for pathological findings in the training images. Each row corresponds to one finding (some images may have multiple findings).

Column Name	Description
Image Index	The filename of the image
Finding Label	The name of the pathology
Bbox [x	X-coordinate of the top-left corner
y	Y-coordinate of the top-left corner
w	Width of the bounding box
h]	Height of the bounding box
Note: The test images are unlabeled. Your model must generate predictions on these images to be submitted for evaluation.

Predictions should be submitted using the provided id values, not the raw image filenames.
A separate ID-to-image mapping is provided to match image_id to its corresponding id.
You must use this mapping to create your submission.csv.

🔸 Expected submission.csv format:
Column	Type	Description
id	int	Unique identifier for the image (see provided mapping)
x_min	float	X coordinate of the top-left corner of the bounding box
y_min	float	Y coordinate of the top-left corner of the bounding box
x_max	float	X coordinate of the bottom-right corner of the bounding box
y_max	float	Y coordinate of the bottom-right corner of the bounding box
confidence	float	Confidence score of the prediction (ranging from 0 to 1)
label	string	Predicted class label for this bounding box
Files
907 files

Size
366.81 MB

Type
png, csv

License
CC0: Public Domain

ID_to_Image_Mapping.csv(3.01 kB)

2 of 2 columns


id

image_id
Label	Count
1.00 - 15.70	15
15.70 - 30.40	15
30.40 - 45.10	15
45.10 - 59.80	14
59.80 - 74.50	15
74.50 - 89.20	15
89.20 - 103.90	14
103.90 - 118.60	15
118.60 - 133.30	15
133.30 - 148.00	15
Label	Count
1.00 - 15.70	15
15.70 - 30.40	15
30.40 - 45.10	15
45.10 - 59.80	14
59.80 - 74.50	15
74.50 - 89.20	15
89.20 - 103.90	14
103.90 - 118.60	15
118.60 - 133.30	15
133.30 - 148.00	15
1
148
00005066_030.png
1%
00013659_019.png
1%
Other (144)
97%
1
00000865_006.png
2
00028383_002.png
3
00027577_003.png
4
00000468_033.png
5
00013922_021.png
6
00023078_003.png
7
00026451_068.png
8
00025962_000.png
9
00019187_000.png
10
00023089_004.png
11
00005089_040.png
12
00020673_005.png
13
00010575_002.png
14
00006948_002.png
15
00011514_015.png
16
00021896_003.png
17
00015649_000.png
18
00021181_002.png
19
00030162_029.png
20
00013285_026.png
21
00005066_030.png
22
00026983_001.png
23
00006851_034.png
24
00008814_010.png
25
00001373_039.png
26
00030394_001.png
27
00001836_082.png
28
00028265_007.png
29
00020184_013.png
30
00019177_000.png
31
00013272_005.png
32
00026538_034.png
33
00028454_016.png
34
00026194_010.png
35
00019863_010.png
36
00022727_000.png
37
00029464_003.png
38
00013807_009.png
39
00016786_001.png
40
00021489_013.png
41
00026769_010.png
42
00021420_020.png
43
00005869_001.png
44
00020438_011.png
45
00019767_016.png
46
00018721_010.png
47
00003973_008.png
48
00013659_019.png
49
00019124_090.png
50
00020405_041.png
51
00005089_014.png
52
00005066_030.png
53
00013670_151.png
54
00016587_069.png
55
00004808_090.png
56
00013471_002.png
57
00025228_005.png
58
00003333_002.png
59
00015719_005.png
60
00020332_000.png
61
00013031_005.png
62
00012686_003.png
63
00010815_006.png
64
00009619_000.png
65
00028518_012.png
66
00014716_007.png
67
00018496_006.png
68
00009745_000.png
69
00000845_000.png
70
00030260_005.png
71
00004461_000.png
72
00012592_005.png
73
00012376_010.png
74
00019499_000.png
75
00027479_013.png
76
00013187_002.png
77
00009368_006.png
78
00001836_041.png
79
00014223_009.png
80
00005089_014.png
81
00017188_002.png
82
00029861_013.png
83
00021818_026.png
84
00014839_017.png
85
00028518_021.png
86
00029647_002.png
87
00017514_008.png
88
00008841_025.png
89
00013721_005.png
90
00014870_004.png
91
00019625_002.png
92
00025270_000.png
93
00023162_025.png
94
00013659_019.png
95
00020259_002.png
96
00022961_008.png
97
00029843_001.png
98
00017500_002.png
99
00018762_001.png
100
00025787_050.png
101
00021024_022.png
102
00026920_000.png
103
00025969_000.png
104
00022155_008.png
105
00010770_000.png
106
00001673_016.png
107
00026555_001.png
108
00028018_000.png
109
00008399_007.png
110
00012829_004.png
111
00021670_004.png
112
00009608_024.png
113
00019634_004.png
114
00013977_005.png
115
00026695_000.png
116
00012376_011.png
117
00025747_000.png
118
00029075_013.png
119
00028924_009.png
120
00021748_000.png
121
00012021_081.png
122
00030412_001.png
123
00012094_047.png
124
00023093_007.png
125
00014663_013.png
126
00009229_003.png
127
00018366_000.png
128
00025529_018.png
129
00010936_011.png
130
00027278_007.png
131
00019154_002.png
132
00020065_008.png
133
00011402_007.png
134
00013922_022.png
135
00013508_001.png
136
00000830_000.png
137
00004342_002.png
138
00026196_001.png
139
00019399_010.png
140
00027927_009.png
141
00020277_001.png
142
00010277_000.png
143
00028607_000.png
144
00022416_048.png
145
00011502_001.png
146
00014976_003.png
147
00012261_001.png
148
00015078_013.png
No more data to show
Data Explorer
366.81 MB

test

train

ID_to_Image_Mapping.csv

train.csv

Summary
907 files

8 columns


Download All


## Submissions ⚠️
You must submit your predictions in CSV format.
This file contains the predictions of your model with the bounding boxes as well as the confidence score.

⚠️ Each prediction must be associated with a unique id, not the image filename. A mapping between image_id and id is provided — make sure to match your predictions using this table. Submissions with incorrect or missing id values will not be evaluated.

Required format
Column	Type	Description
id	int	Unique identifier matching each image (see provided mapping)
image_id	string	Name of the image file
x_min	float	X coordinate of the top-left corner of the bounding box
y_min	float	Y coordinate of the top-left corner of the bounding box
x_max	float	X coordinate of the bottom-right corner of the bounding box
y_max	float	Y coordinate of the bottom-right corner of the bounding box
confidence	float	Confidence score of the prediction (ranging from 0 to 1)
label	string	Predicted class label for this bounding box