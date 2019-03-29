# Squeezed Very Deep Convolutional Networks for Text Classification (SVDCNN)

A. D. Duque, L. L. Santos, D. Macêdo, C. Zanchettin, "Squeezed Very Deep Convolutional Networks for Text Classification".


## Datasets:
| Dataset                | Classes | Train samples | Test samples | source |
|------------------------|:---------:|:---------------:|:--------------:|:--------:|
| AG’s News              |    4    |    120 000    |     7 600    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Yelp Review Polarity   |    2    |    560 000    |    38 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Yelp Review Full       |    5    |    650 000    |    50 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|


Thanks to [@ArdalanM](https://github.com/ArdalanM) for the **VDCNN** implentation used as baseline ([here](https://github.com/ArdalanM/nlp-benchmarks/))

## Execution 
Execute the run.sh file through the command: 
- bash ./run.sh

## Results:

### Model size in MB for a generic text classification problem with 4 target classes

| | SVDCNN | VDCNN | Char-CNN | 
|:---------------:| :-------------:| :-------------:| :-------------:|
| 6 layers | --- | --- | 43.25 |
| 9 layers | 2.80 | 54.75 | --- |
| 17 layers | 5.52 | 62.74 | --- |
| 29 layers | 6.03 | 64.16 | --- |

### Ag news accuracy
| | SVDCNN | VDCNN | Char-CNN | 
|:---------------:| :-------------:| :-------------:| :-------------:|
| 6 layers | --- | --- | 92.36 |
| 9 layers | 90.13 | 90.83 | --- |
| 17 layers | 90.43 | 91.12 | --- |
| 29 layers | 90.55 | 91.27 | --- |

### Yelp polarity accuracy
| | SVDCNN | VDCNN | Char-CNN | 
|:---------------:| :-------------:| :-------------:| :-------------:|
| 6 layers | --- | --- | 95.64 |
| 9 layers | 94.99 | 95.12 | --- |
| 17 layers | 95.04 | 95.50 | --- |
| 29 layers | 95.26 | 95.72 | --- |

### Yelp review accuracy
| | SVDCNN | VDCNN | Char-CNN | 
|:---------------:| :-------------:| :-------------:| :-------------:|
| 6 layers | --- | --- | 62.05 |
| 9 layers | 61.97 | 63.27 | --- |
| 17 layers | 63.00 | 63.93 | --- |
| 29 layers | 63.20 | 64.26 | --- |
