# Squeezed Very Deep Convolutional Networks (SVDCNN)

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

### Ag news 
| Model           | accuracy 
|:---------------:| :-------------:|
| SVDCNN 9 layers  |  90.13        |
| SVDCNN 17 layers |  90.43        |
| SVDCNN 29 layers |  90.55        |

### Yelp polarity
| Model           | accuracy 
|:---------------:| :-------------:|
| SVDCNN 9 layers  |  94.99        |
| SVDCNN 17 layers |  95.04        |
| SVDCNN 29 layers |  95.26        |

### Yelp review
| Model           | accuracy 
|:---------------:| :-------------:|
| SVDCNN 9 layers  |  61.97        |
| SVDCNN 17 layers |  63.00        |
| SVDCNN 29 layers |  63.20        |
