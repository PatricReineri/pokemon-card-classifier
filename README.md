# Pokémon Card Classifier

This project involves a classifier that uses a fine-tuned ResNet18 model to recognize Pokémon cards from images. The model is trained on a dataset of Pokémon card images and can accurately classify new cards captured via a webcam. The project also includes a graphical interface for displaying the predicted card alongside a screenshot of the card captured by the camera.

## Features

- **Card Classification:** Uses a fine-tuned ResNet18 model for recognizing Pokémon cards.
- **Image Capture:** Captures images of cards using a webcam.
- **Display Interface:** Shows the predicted card and the captured screenshot in a Tkinter window.
- **Data Handling:** Supports saving screenshots and handling JSON files for card information.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/PatricReineri/pokemon-card-classifier.git
    cd pokemon-card-classifier
    ```

2. **Install required libraries:**

    Ensure you have Python 3.x installed. You can install all required libraries using the `requirements.txt` file provided in the repository:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download images and data:**

    Before training the model, you need to download Pokémon card images and data. Use the `download_set.py` script to download all necessary images and data. This script will save the images and JSON files in the appropriate folders:

    ```bash
    python download_set.py
    ```

4. **Train the model:**

    Once you have downloaded the images and data, you can train the ResNet18 model using the `card-classifier.py` script. This will generate the required model file (`pokemon_classifier_crop_cards.pth`) and label encoder file (`classes.npy`):

    ```bash
    python card-classifier.py
    ```

5. **Prepare the directories:**

    - The `download_set.py` script will create a `base1_images` folder for card images and a `base1` folder for JSON files. Ensure these directories are populated with the downloaded data.

## Usage

1. **Run the classifier script:**

    To start capturing images and classifying Pokémon cards, run the following command:

    ```bash
    python cam-tcgclassifier.py
    ```

2. **Capture and Classify:**

    The script will open your webcam, display a rectangle to help you position the card, and capture an image after 5 seconds. It will then classify the card and display the predicted card image and the captured screenshot in a Tkinter window.

3. **Test the trained model:**

    You can test the trained model using the `card_classifier_usage.py` script. This script allows you to provide an image for classification and will display the predicted result. Run the script with:

    ```bash
    python card_classifier_usage.py
    ```

4. **Interact with the Interface:**

    - **Next Button:** Click the "Next" button to close the Tkinter window and start a new capture.

## Code Explanation

- **`download_set.py`**: Used for downloading Pokémon card images and data. It saves images and JSON files required for training and classification.

- **`card-classifier.py`**: Used for training the ResNet18 model and generating the necessary model and label encoder files.

- **`cam-tcgclassifier.py`**: Contains code for capturing images using the webcam, classifying the captured images with the Pokémon card classifier, and displaying the results using Tkinter.

- **`card_classifier_usage.py`**: Allows testing the trained model with provided images to check classification results.

- **`display_predicted_card(image_path)`**: Function that creates a Tkinter window to display the predicted card and screenshot.

- **`capture_image()`**: Captures an image from the webcam and saves it as `screenshot.png`.

- **`classify_and_show_image()`**: Classifies the captured image and displays the result.

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **[ResNet](https://arxiv.org/abs/1512.03385)**: For the original architecture of the ResNet model.
- **[OpenCV](https://opencv.org/)**: For the computer vision tools used in image capture.



