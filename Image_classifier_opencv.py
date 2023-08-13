import cv2
import numpy as np
import streamlit as st

def load_model():
    """Loads the DNN model."""
    # Read the ImageNet class names.
    with open('models/classification_classes_ILSVRC2012.txt', 'r') as f:
        image_net_names = f.read().split('\n')

    # Final class names, picking just the first name if multiple in the class.
    class_names = [name.split(',')[0] for name in image_net_names]

    # Load the neural network model.
    model = cv2.dnn.readNet(
        model='models/DenseNet_121.caffemodel',
        config='models/DenseNet_121.prototxt',
        framework='Caffe')
    return model, class_names

def main():
    net, class_names = load_model()
    st.header("Image Classifier Opencv CaffeModel")

    # Upload image.
    uploaded_file = st.sidebar.file_uploader('Upload Images:', type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)

    if len(uploaded_file) > 0:
        images_files = [file.name for file in uploaded_file]
        img_list = list(dict.fromkeys(images_files))
        img_choose = st.selectbox("Choose the image ", img_list)
        image = images_files.index(img_choose)
        # Convert the file to an opencv image.
        raw_bytes = np.asarray(bytearray(uploaded_file[image].read()), dtype=np.uint8)
        img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

        # Set Blob for input image conversion
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.017, size=(224,224), mean=(104, 117, 123))
        # Set blob into model structure
        net.setInput(blob)
        outputs = net.forward()
        final_outputs = outputs[0]
        # Make all the outputs 1D.
        final_outputs = final_outputs.reshape(1000, 1)

        label_id = np.argmax(final_outputs)
        # Convert the output scores to softmax probabilities.
        probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
        # Get the final highest probability.
        final_prob = np.max(probs) * 100.
        # Map the max confidence to the class label names.
        out_name = class_names[label_id]
        if final_prob > 50:
            color = "green"
        else:
            color = "red"
        out_text = f"Class: **:{color}[{out_name}]** \n\n Probability: **:{color}[{final_prob:.1f}%]**"
        st.markdown(out_text )
        st.image(img, channels="BGR")
    else:
        st.warning("Upload Images to Classify")

if __name__ == "__main__":
    main()