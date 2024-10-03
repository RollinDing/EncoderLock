## Artifacts: 

### Probe-Me-NOT: Protecting Pre-trained Encoders from Malicious Probing

### Environments:
We run the EncoderLock on a server:

- CPU: AMD Ryzen 9 3900X 12-Core Processor
- GPU: NVIDIA TITAN RTX 
    -  Driver Version: 470.182.03   
    -  CUDA Version: 11.4  
- Python Version: 3.9.13
- Pytorch Version: 1.12.1+cu102

To automatically install the required packages, run following code in the conda environment:
    `conda `

### Direcotry structure:
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
        |
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

    - For the evaluation on our real-world example, you can need to download the ImageNette and ImageWoof dataset [here](https://github.com/fastai/imagenette), and download the prohibited dataset--military vehicle dataset [here](https://www.kaggle.com/datasets/amanrajbose/millitary-vechiles). 
**NOTE**: for the image datasets, you should change the model input size to $224$.

### Run the tests:
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
            bash evaluate-encoder.sh

    - Change function `load_feature_extractor` in `src/evaluate/evaluate-encoder.py` line 75 to switch between 
        - `example-supervised`
        - `example-unsupervised`  
        - `supervised`
        - `unsupervised`
        to choose the modified encoder to evaluate

    - We provide $6$ pre-trained encoders with supervised and unsupervised EncoderLock that is used in our experiments results, based on the `mnist` dataset as the source. Note that you should train the unprotected model and put it in `pretrained_models' first to do the further evaluation as well. 

