# Hand Gesture Classifier

This project aims to build a hand gesture classifier using computer vision techniques. The classifier will be trained on hand gesture data collected from multiple team members and will be evaluated on unseen data.

## Project Structure

```
hand-gesture-classifier
├── notebooks
│   └── hand_gesture_classifier_colab.ipynb  # Jupyter notebook for Google Colab
├── data
│   ├── train
│   │   ├── member_1                           # Training data for team member 1
│   │   ├── member_2                           # Training data for team member 2
│   └── test
│       └── member_3                           # Testing data for team member 3
├── src
│   ├── data_collection.py                      # Script for capturing hand gesture data
│   ├── feature_extraction.py                   # Script for extracting features from images
│   └── train_classifier.py                      # Script for training the gesture classifier
├── requirements.txt                            # Required packages for the project
└── README.md                                   # Project documentation
```

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine.
   
   ```bash
   git clone <repository-url>
   ```

2. **Install Requirements**: Install the necessary packages listed in `requirements.txt`. You can do this by running:

   ```bash
   pip install -r requirements.txt
   ```

3. **Open the Notebook**: Navigate to the `notebooks` directory and open `hand_gesture_classifier_colab.ipynb` in Google Colab.

4. **Run the Notebook**: Follow the instructions in the notebook to set up the environment, collect hand gesture data from the camera, and train the classifier using the provided data.

## Usage

- The `data_collection.py` script is used to capture hand gesture images from the camera and save them in the appropriate directories.
- The `feature_extraction.py` script processes the captured images to extract relevant features for gesture recognition.
- The `train_classifier.py` script trains the gesture classifier using the extracted features and the training data.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Workshop Assets

Some hand gesture images used in the workshop materials were generated
using OpenAI's image generation tools for demonstration purposes.

## Dataset

The gesture classification dataset was collected by the workshop development team using webcam images and MediaPipe Hand Landmarker.

It contains four gestures (`OPEN_PALM`, `FIST`, `PEACE`, `POINTING`) collected from three contributors using both hands. Two contributors are used for training, while the third is held out to evaluate generalisation to an unseen participant.

See [`data/README.md`](data/README.md) for the full data collection methodology.