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

In this quick evaluation, we offer pre-trained encoder's with modified weights on digits datasets. 
For each dataset, it will take  within 5 minutes to do the evaluation steps, as we only run **20 epochs** for each domain.

To train your own encoderlock, you can also follow the steps in "Quick Evaluation on the Training Procedure of EncoderLock", it may takes longer time (about 30 minutes when source domain is the MNIST and target domain is the USPS. It may take longer time for larger datasets.)
The final accuracy can also be found in the end of the logging file in `logs/`

#### Evaluation Steps:

- **Step 1** Prepare for the environment, datasets, and pre-trained models. We provide 6 protected encoders with supervised and unsupervised EncoderLock. The saved checkpoints can be found in `modified_models`.
- **Step 2** Check the availability of protected encoders in `modified_models`; Note that you should download [unprotected models](https://drive.google.com/drive/folders/1GOwsVl8K6qLoFWJ57geFv5oWrfcNgMOs?usp=sharing) and put them in a folder `./pretrained_models`. 
- **Step 3** To evaluate our pretrained supervised EncoderLock, Run script `tests/evaluate-encoder`, 
          bash tests/evaluate-encoder.sh 
          using level=example-supervised
- **Step 3** To evaluate our pretrained unsupervised EncoderLock, Run script `tests/evaluate-encoder`, 
          bash tests/evaluate-encoder.sh 
          using level=example-unsupervised

- **Expected Outputs**:
    - the script will first probe for the source domain and then run the probing process for the target domain,
    - **STDOUT**: 
            Training the encoder on the target domain
        
            Training the downstream classifier from scratch on the source domain!
        
            ==> Epoch: 0 | Loss: 2.138676404953003 | Train Accuracy: 18.73% | Val Loss: 2.0617 | Val Accuracy: 26.04%
          
            ...
          
            ==> Epoch: 64 | Loss: 0.4867952764034271 | Train Accuracy: 98.18% | Val Loss: 0.4543 | Val Accuracy: 98.03%
        
        
            ==> Epoch: 65 | Loss: 0.4273190498352051 | Train Accuracy: 98.30% | Val Loss: 0.4484 | Val Accuracy: 98.15% 
        
            Early stopping triggered.
        
            Finish training the downstream classifier on the source domain, the best accuracy on the source domain is **98.03%**
        
            Training the downstream classifier from scratch on the target domain!
        
            ==> Epoch: 0 | Loss: 4.532858371734619 | Train Accuracy: 7.40% | Val Loss: 2.4023 | Val Accuracy: 6.96% 
        
            ==> Epoch: 1 | Loss: 2.350107192993164 | Train Accuracy: 7.28% | Val Loss: 2.3876 | Val Accuracy: 7.15% 
        
            ....
        
            Finish training the downstream classifier on the target domain, the best accuracy on the target domain is **15.7460049170252%**
            
    - **Logging File**: 
    We record the log for every evaluation in 
    `logs/example-superised/evaluation-{model}-{source domain}-{target domain}/`
    
    In the log file, you can find:
    1. Total number weights of the encoder;
    2. Total number of changed weights, and the percentage of modified weights;
    3. Loss/Accuracy on the source domain
    4. Loss/Accuracy on the target domain.

    - **For reproduciability**
    Running `bash tests/evaluate-encoder.sh`, we are going to reproduce the results in Table II, III in the manuscript. 
    1. The (train from scratch) source accuracy should be $\pm 2 %$ (depends on the evaluation epochs), compared with the average accuracy report in the diagonal of the table, it indicates that the source accuracy does not drop too much. 
    The reported accuracy can be evaluated during the training phase (not train from scratch.)
    2. The target accuracy should be **less than** the accuracy after $\Rightarrow$ in the table, indicating that the accuracy on the target domain is low.
    3. The original accuracy can also be found in the table or you may reproduce it by running the training steps. The original model accuracy will be presented at the beginning of the logging file.


These modified models are encoders with EncoderLock, which shows significant accuracy degradation in the target domain (prohibited domain) but preserves high accuracy in the source domain (authorized domain).

#### Quick Evaluation on the Training Procedure of EncoderLock

We provide two scripts to evaluate the training procedure of supervised EncoderLock and unsuperized EncoderLock.

- supervised EncoderLock
Run 

```
bash train-supervised-encoderlock.sh
```
This script will output a log file contains testing accuracy for each 5 steps in the training process.
The final output model will be in modified models.
The final accuracy present in the log file can also be refer to the source/target accuracy in Table II and III.



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

    **Output**: In the produced `.log` file, you will be able to find the running time for each steps and encoder testing accuracy for source and target domain. The final results will be listed too.
----
3. Evaluate modified encoders: to evaluate the modified model only, we also provide a script.
    - Once you have the modified model, we offer scripts to evaluate it, simply run 
          ```bash tests/evaluate-encoder.sh```

    -  In tests/evaluate-encoder.sh switch between levels:
        - `supervised`
        - `unsupervised`
        to choose the type of modified encoder to evaluate

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


### Reference
```
@article{protecting2024,
  author = {Ruyi Ding, Tong Zhou, Lili Su, Aidong Adam Ding, Xiaolin Xu, Yunsi Fei}, 
  title = {Probe-Me-Not: Protecting Pre-trained Encoders from Malicious Probing},
  journal = {arXiv preprint arXiv:2411.12508},
  year = {2024},
  url = {https://arxiv.org/abs/2411.12508}
}
```

### License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
