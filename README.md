# Automated RCC subtype classification
this package provide the model for automated diagnosis of renal cell carcinoma subtype and findings.

# Docker
```
cd BuildDocker
docker build . -t multicnn
docker run --gpus all --shm-size=32g --name multicnn_container -v /home/shono/:/srv -v /mnt/shono/:/home/shono -v /data1/:/data1_true -v /data2/:/data1 -v /share:/share -d -p 8208:8888 -p 8206:6006  multicnn
```

# Requirements:
pandas
numpy
SimpleITK
matplotlib
pytorch
torchvision
cv2
coral_pytorch
sklearn
pickle
glob
efficientnet_pytorch
tqdm
jupyter

# Required files
* /data2/RCC/shono_dicom2
* /data2/RCC/RCC_detail_done.csv 
* /data2/RCC/accession_DB_merged.csv
