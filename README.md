## Artifacts: 
### Probe-Me-NOT: Protecting Pre-trained Encoders from Malicious Probing

### Abstract
Adapting pre-trained deep learning models to customized tasks has become a popular strategy for developers facing limited computational resources and data volume. In this work, we propose **EncoderLock**, a novel applicability authorization method designed to prevent malicious probing, i.e., fine-tuning downstream heads of pre-trained encoders on prohibited domains. We introduce three levels of EncoderLocks tailored to different dataset availabilities: supervised, unsupervised, and zero-shot EncoderLocks. Specifically, supervised EncoderLocks leverage labeled prohibited domains, while unsupervised and zero-shot EncoderLocks focus on prohibited or synthetic datasets without labels. Accordingly, in this package, we provide the source code for both scenarios—training with and without labeled datasets. After preparing the datasets and the victim pre-trained models, we recommend running the scripts in the `test` directory, where we offer examples of training and evaluating EncoderLock for both types of datasets.

### Environments:
We run the EncoderLock on a server:

- CPU: AMD Ryzen 9 3900X 12-Core Processor
- GPU: NVIDIA TITAN RTX 
    -  Driver Version: 470.182.03   
    -  CUDA Version: 11.4  
- Python Version: 3.9.13
- Pytorch Version: 1.12.1+cu102

To automatically install the required packages, run the following code in the conda environment:
    ```conda env create -f environment.yml```

### Quick Evaluation:
We provide 12 protected encoders with supervised and unsupervised EncoderLock. Note that you should download [unprotected models](https://drive.google.com/drive/folders/1GOwsVl8K6qLoFWJ57geFv5oWrfcNgMOs?usp=sharing) and put it in a folder `pretrained_models\`. The saved checkpoints can be found in `modified_models`.
These modified models are encoders with EncoderLock, which shows significant accuracy degradation in the target domain (prohibited domain) but preserves high accuracy in the source domain (authorized domain).

- **Step 1** Prepare for the environment, datasets, and pre-trained models.
- **Step 2** Check the available hardened encoders in `modified_models`; Change function `load_feature_extractor` in `src/evaluate/evaluate-encoder.py` line 75 to switch between `example-supervised` or `example-unsupervised`, for supervised or unsupervised EncoderLock.
- **Step 3** Run script `bash evaluate-encoder.sh`

### Directory structure:
```
EncoderLock
    |
    |--data: datasets used in the experiment
    |
    |--pretrained_models: pretrained models (encoders)
    |
    |--modified_models: protected encoders using EncoderLock
    |
    |--src: source code main directory
        |
        |--utils:
            |
            |--argparser.py
            |
            |--data.py
            |
            |--utils.py
        |
        |--evaluate:
            |
            |--evaluate-encoder.py
        |
        |--supervised-encoderlock.py
        |
        |--unspervised-encoderlock.py
    |
    |--tests: test scripts
        |
        |--evaluate-encoder.sh
        |
        |--train-supervised-encoderlock.sh
        |
        |--train-unsupervised-encoderlock.sh
```

### Datasets
Most of the dataset we use in the experiment can be downloaded automatically in the torchvision package, 
- including: 
    - Digits:
        - MNIST
        - USPS
        - SVHN
        - MNISTM
        - SYN
    - Simple Images:
        - CIFAR10
        - EMNIST
        - STL10

    - For the evaluation on our real-world example, you can need to download the ImageNette and ImageWoof dataset [here](https://github.com/fastai/imagenette), and download the prohibited dataset--military vehicle dataset [here](https://www.kaggle.com/datasets/amanrajbose/millitary-vechiles). 
**NOTE**: for the image datasets, you should change the model input size to $224$.


### Pre-trained Models 
We provide the unprotected pretrained models [here](https://drive.google.com/drive/folders/1GOwsVl8K6qLoFWJ57geFv5oWrfcNgMOs?usp=sharing)

### Train and Test EncoderLock from scratch:
We provide several scripts to run the experiments:

----
1. Train the Supervised EncoderLock:
    - Before running the EncoderLock, make sure you have the pre-trained victim model to protect and save that in `pretrained_models`. For instance, we provide a pre-trained model checkpoint in `pretrained_models/mnist_vgg11_30`, which is fine-tuned from the pytorch pre-trained VGG11 with mnist dataset for 30 epochs.

    - To run the training process of Supervised EncoderLock, simply run:

            bash tests/train-supervised-encoderlock.sh

        the script will automatically run supervised EncoderLock and save the modified models in `modified models/supervised/vgg11--mnist-usps`. And it will also save the training logs in `logs/supervised/`. The log will provide information about the accuracy for both source and target domains for each epoch and number of modified weights for each round.

    - You can also change the source (authorized) and target (prohibited) domains in this script by modify `dataset` (for source, and make sure you have the pretrained model!) or `std_dataset` for the target domain. You can also chagne the model architectures.
    Furthermore, you are able to adjust the hyperparameters including E, R and data volume, that discussed in ablation study of our manuscript.
----
2. Train the Unsupervised EncoderLock/Zero-shot EncoderLock
    Similar to the supervised EncoderLock, to run training process of unsupervised EncoderLock, using 

            bash tests/train-unsupervised-encoderlock.sh

    For zero-shot EncoderLock, you need generate a synthetic dataset first.

----
3. Evaluate modified encoders
    - Once you have the modified model, we offer scripts to evaluate it, simply run 
          ```bash evaluate-encoder.sh```

    - Change function `load_feature_extractor` in `src/evaluate/evaluate-encoder.py` line 75 to switch between 
        - `example-supervised`
        - `example-unsupervised`  
        - `supervised`
        - `unsupervised`
        to choose the modified encoder to evaluate



