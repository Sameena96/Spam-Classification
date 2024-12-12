# Spam-Classification
This project demonstrates a spam classification system using a Random Forest classifier. The system processes a dataset of text messages, classifies them as spam or not spam (ham), and allows users to predict classifications for individual messages interactively.

**Features**

1. Large Dataset Generation
A synthetic dataset of spam and ham messages is generated using predefined templates.
Alternating labels are used to create a balanced dataset of over 10,000 entries.

2. Data Preprocessing
Includes text cleaning steps such as removing URLs, non-alphanumeric characters, and stop words.
Uses stemming for text normalization to improve classification accuracy.

3. Text Vectorization
Employs TF-IDF vectorization to convert text messages into numerical feature vectors suitable for machine learning models.

4. Spam Classification
Implements a Random Forest classifier for high-accuracy predictions.
Provides performance evaluation metrics such as accuracy and a detailed classification report.

5. Interactive Prediction
Allows users to input an index from the dataset to view the message, its actual label, and the model's prediction.

**Prerequisites**

**Required Libraries**

1. pandas
2. scikit-learn
3. nltk
4. numpy

**Install the required libraries using:**

      __pip install pandas scikit-learn nltk numpy

**Setup**

Ensure the dataset spam_text_large_dataset.csv is in the project directory.

**How It Works**

**Dataset Generation**
Predefined spam and ham messages are randomly selected to create a CSV file (spam_text_large_dataset.csv) with Message and Prediction columns.

**Preprocessing**
- Text preprocessing includes removing unnecessary characters, stop word removal, and stemming.
- The processed data is vectorized using the TF-IDF method, converting text to numerical form.

**Model Training and Evaluation**
- The dataset is split into training and testing subsets.
- A Random Forest model is trained on the processed data.
- The model's performance is evaluated using accuracy and detailed classification reports.

**Interactive Prediction**

**.** Users can input the index of any message in the dataset to:

    - View the message text.
    - See the actual label (Spam or Not Spam).
    - Get the model's prediction.

**Usage**

1. **Generate the dataset:**

    python generate_dataset.py
This will create spam_text_large_dataset.csv in the project directory.
2. **Run the spam classification script:**

    python classify_spam.py
3. **Follow the on-screen instructions to input message indices and view predictions.**

**File Structure**

spam-classification/
├── generate_dataset.py    # Script to generate the synthetic dataset
├── classify_spam.py       # Main script for spam classification
├── spam_text_large_dataset.csv # Generated dataset file
├── README.md              # Project documentation

**Future Enhancements**
- Data Augmentation: Add real-world spam datasets for enhanced training.
- Model Optimization: Experiment with other machine learning models such as SVM or neural networks.
- GUI Interface: Create a user-friendly graphical interface for easier interaction.
- Real-Time Processing: Integrate with email systems for live spam detection.
