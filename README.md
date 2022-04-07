# **Sign language Detection using Mediapipe and Convolutional Neural Networks**



Datasets used:
1. Akash, N. (2018) ‘ASL Alphabet: Image dataset for alphabets in the American Sign Language’ [Online] Available at: https://www.kaggle.com/grassknoted/asl-alphabet/metadata (Accessed: 22 December, 2021)
2. Lee, D. (2020) ‘American Sign Language Letters Dataset’  [Online] Available at: https://public.roboflow.com/object-detection/american-sign-language-letters (Accessed: 22 December, 2021)
3. Mondal, P. (2022) 'ASL alphabets' [Online] Available at: https://www.kaggle.com/pradiptomondal/aslalphabets


Datasets were collected from various sources and converted hand landmarks were extracted from them using mediapipe.

The datasets were arranged according to the following directory structure:

```
dataset
│    
└──-train  
│   └───A
│   │   │─── image1.jpg
│   │   │─── image2.jpg
│   │   │─── ...
│   │
│   └───B
│   │   │─── image1.jpg
│   │   │─── image2.jpg
│   │   │─── ...
│   │
│   └─── ...
│
└───valid
    └───A
    │   │─── image1.jpg
    │   │─── image2.jpg
    │   │─── ...
    │
    └───B
    │   │─── image1.jpg
    │   │─── image2.jpg
    │   │─── ...
    │
    └─── ...
```

## Steps to train the model:
1. Download the project repository, and cd into it.
```
$ git clone https://github.com/pradipto111/mediapipe_SignLanguage.git
```

2. Create a python virual environment and activate it.


For Unix/MacOS:
```
$ python3 -m venv sign_lang
$ source sign_lang/bin/activate
```
For windows:
```
$ py -m venv sign_lang
$ .\sign_lang\Scripts\activate
```

3. Install the required libraries.
```
$ pip3 install -r requirements.txt
```

4. Training the Mediapipe model
    1. The landmarks are derived from the combination of these two datasets using mediapipe - [ASL-grassknotted](https://www.kaggle.com/grassknoted/asl-alphabet) and [Roboflow](https://public.roboflow.com/object-detection/american-sign-language-letters/1)
    2. Store the dataset inside the 'mediapipe' folder, and navigate to this directory through the command line.
    3. Train the model:
    ```
    $ python3 train.py dataset_path \
    --epochs 20 \
    --learning_rate 0.0001 \
    --weight_decay 0.0001 \
    --grad_clip 0.1 \
    --batch_size 32
    ```
    Default values:
    -   Epochs 10
    -   Learning Rate 1e-5
    -   Weight Decay 1e-4
    -   Gradient Clipping 0.1
    -   Batch Size 32

5. Training the CNN model
    1. Download the image dataset from [ASL-grassknotted](https://www.kaggle.com/grassknoted/asl-alphabet) and [Roboflow](https://public.roboflow.com/object-detection/american-sign-language-letters/1)
    These images are taken from the original dataset, organised into sub-directories and preprocessed.
    2. Store the dataset inside the 'CNN' folder, and navigate to this directory through the command line.
    3. Train the model:
    ```
    $ python3 train.py dataset_path model_name \
    --epochs 20 \
    --learning_rate 0.0001 \
    --weight_decay 0.0001 \
    --grad_clip 0.1 \
    --batch_size 32
    ```
    Default values:
    -   Epochs 10
    -   Learning Rate 1e-5
    -   Weight Decay 1e-4
    -   Gradient Clipping 0.1
    -   Batch Size 32

    As model_name insert resnet34 or mobilenet_v2 as desired. Change the other arguments as necessary.

6. Save the model if needed with appropriate response to the prompt, insert ```MODEL_NAME``` and ```VERSION``` for the same.




## Evaluation:
The best results were achieved using CNN architectures of Resnet & Mobilenet.
To evaluate these models on a test_dataset, run the script ```eval.py``` inside the ```CNN``` folder, with the following arguments:
```
python3 eval.py <dataset_path> <model_name> <weight_file_path>
```
Note that, the argument ```model_name``` can accept only two arguments ```resnet34``` and ```mobilenet_v2```.
**All the three arguments are mandatory**



## Inferencing:

1. ### Mediapipe model:
    1. Run the script ```test.py``` inside the ```mediapipe``` folder, with the following arguments:
    ```
    python3 test.py --model_path <path to .pt file of model>
    ```

    2. Show the hand gesture to on the webcam window which will be opened. 
    3. Press ``SPACE BAR`` key when ready, and the prediction will be displayed in the terminal / command prompt window.


2. ### CNN model:
    1. Run the script ```test.py``` inside the ```CNN``` folder, with the following arguments:
    ```
    python3 test.py --model <Model name resnet34/mobilenet_v2> --model_path <path to .pt file of model>
    ```

    2. Show the hand gesture to on the webcam window which will be opened. 
    3. Press ``SPACE BAR`` key when ready, and the prediction will be displayed in the terminal / command prompt window.



