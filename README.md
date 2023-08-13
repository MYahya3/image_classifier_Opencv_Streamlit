# Image classifier using Opencv Streamlit

#### To try the Image Classifer Model App online use the below link 
```bash
https://imageclassifieropencvapp-6xmbrrrvtgkfneaw7afy59.streamlit.app/
```
## Local System Configuration

1. Clone the repository:

```bash
git clone https://github.com/MrYahya18/image_classifier_Opencv_Streamlit.git
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To access the motion detector, run the following command:

```bash
streamlit run ./Image_classifier_opencv.py
```
![image](https://github.com/MrYahya18/image_classifier_Opencv_Streamlit/assets/88489038/674e468e-022e-4d42-8db5-56d64c42e12f)

## Working Process

Here's an overview of the working process of the code:

1. Upload Images from the Uploading Option.
2. Choose an image to classify it.
3. Used Opencv Caffe Model to predict image class and visualize on top of the image
7. Class and Score in red color means less then 50% and green means > 50% 
