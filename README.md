# Pokémon Card Classifier

This project uses a fine-tuned **EfficientNetB1** model to recognize Pokémon cards from images. The model has been trained on a dataset of Pokémon card images and can accurately classify new cards captured via webcam. The project also includes a graphical interface to display the predicted card alongside the screenshot captured by the camera.

## Features

- **Card Classification:** Uses a fine-tuned EfficientNetB1 model to recognize Pokémon cards.
- **Image Capture:** Captures images of the cards using a webcam.
- **Graphical Interface:** Displays the predicted card and the captured screenshot in a Tkinter window.
- **Data Management:** Supports saving screenshots and handling JSON files for card information.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/PatricReineri/pokemon-card-classifier.git
    cd pokemon-card-classifier
    ```

2. **Install required libraries:**

    Make sure you have Python 3.x installed. You can install all required libraries using the `requirements.txt` file provided in the repository:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download images and data:**

    Before training the model, you need to download Pokémon card images and data. Use the `download_set.py` script to download all necessary images and data. This script will save the images and JSON files       in the appropriate folders:

    ```bash
    python download_set.py
    ```
    The `download_set.py` script will create a `base1_images` folder for card images and a `base1` folder for JSON files. Make sure these directories are populated with the downloaded data.

4. **Train the model:**

    Once the images and data have been downloaded, you can train the EfficientNetB1 model using the `card-classifier.py` script. This will generate the required model file       (`pokemon_classifier_crop_cards.pth`) and the label encoder file (`classes.npy`):

    ```bash
    python card-classifier.py
    ```

## Usage

1. **Run the classifier script:**

    To start capturing images and classifying Pokémon cards, run the following command:

    ```bash
    python cam-tcgclassifier.py
    ```

2. **Capture and Classify:**

    The script will open your webcam, show a rectangle to help you position the card, and capture an image after 5 seconds. It will then classify the card and display the predicted card image and the captured screenshot in a Tkinter window.

    ![Webcam Capture](w1.png)

4. **Test the trained model:**

    You can test the trained model using the `card_classifier_usage.py` script. This script allows you to provide an image for classification and will display the predicted result. Run the script with:

    ```bash
    python card_classifier_usage.py
    ```

5. **Interact with the Interface:**

    - **"Next" Button:** Click the "Next" button to close the Tkinter window and start a new capture.

## Code Explanation

- **`download_set.py`**: Used to download Pokémon card images and data. It saves images and JSON files required for training and classification.
- **`card-classifier.py`**: Used to train the EfficientNetB1 model and generate the necessary model and label encoder files.
- **`cam-tcgclassifier.py`**: Contains the code to capture images using the webcam, classify the captured images with the Pokémon card classifier, and display the results using Tkinter.
- **`card_classifier_usage.py`**: Allows testing the trained model with provided images to verify classification results.
- **`display_predicted_card(image_path)`**: A function that creates a Tkinter window to display the predicted card and screenshot.
- **`capture_image()`**: Captures an image from the webcam and saves it as `screenshot.png`.
- **`classify_and_show_image()`**: Classifies the captured image and displays the result.

## Contributing

If you would like to contribute to this project, please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **[EfficientNet](https://arxiv.org/abs/1905.11946)**: For the model architecture.
- **[OpenCV](https://opencv.org/)**: For the computer vision tools used for image capture.
