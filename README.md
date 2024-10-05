# Gender-Prediction-using-CNN-with-pytorch

# 1. Data Upload and Preprocessing
The user is required to upload a ZIP file containing images. The app extracts the zip file and reads the images and corresponding labels (age, gender, ethnicity) from the filenames. Each image is processed and resized to a fixed size (128x128 pixels) and converted to grayscale. These images serve as the features for the CNN model.
The app expects the filenames to follow a specific format, where the image file names contain the labels for age, gender, and ethnicity. For example, a filename like 24_0_1.jpg might indicate an image of a 24-year-old male of a certain ethnicity. This structure allows the app to extract gender labels (0 for male, 1 for female) directly from the filenames.

# 2. Image Display and Gender Distribution
The app includes the option for the user to visualize the data:
Image Grid: A grid of sample images is displayed with their predicted genders.
Gender Distribution: A bar plot shows the distribution of male and female labels in the dataset, which helps understand the dataset balance.

# 3. CNN Model Architecture
The CNN model is designed to classify gender based on the image features. It is a common deep learning model used for image classification tasks. The key layers in this CNN are:
Convolutional Layers: These layers detect patterns such as edges, textures, and more complex features as the image passes through deeper layers.
MaxPooling Layers: These reduce the spatial dimensions of the feature maps, helping the model generalize better.
Dropout: A regularization technique used to prevent overfitting by randomly turning off some neurons during training.
Flatten Layer: Converts the 2D feature maps into a 1D vector for fully connected layers.
Dense Layers: Fully connected layers at the end of the network used for classification.
The final layer has a sigmoid activation function, which outputs a value between 0 and 1, representing the predicted probability of the image being male or female.

# 4. Training and Validation
The model is trained using a dataset split into training and validation sets. The training process optimizes the model’s weights using the binary cross-entropy loss function, which is suitable for binary classification tasks like gender prediction. The training process is visualized with accuracy and loss graphs, showing how well the model performs on both the training and validation data over time.

# 5. Model Prediction
Once trained, the app allows the user to select an image from the dataset and predict the gender based on the trained model. The model outputs the predicted gender, and the app compares it to the true gender label. This prediction is visualized by showing the selected image alongside the predicted gender.

# 6. User Interface
The app’s interface, built using Streamlit, makes it interactive and easy for users to:
Upload datasets.
Visualize data.
Train the CNN model.
Check predictions.
Streamlit also helps in displaying progress bars and real-time feedback during various operations, such as image loading and model training.

# 7. Overall Workflow
Upload the Dataset: User uploads a zip file containing images.
Extract and Preprocess: Images are extracted, resized, and their labels are extracted from filenames.
Data Visualization: Option to view sample images and the distribution of genders.
Train the CNN Model: User can train the model on the dataset, and the accuracy and loss during training are displayed.
Make Predictions: Once the model is trained, the user can select images and predict the gender.
This approach combines deep learning for image classification with a user-friendly interface, making it accessible for users to train and use a gender prediction model.
