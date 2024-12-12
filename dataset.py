import pandas as pd
import random

# Define messages
spam_messages = [
    "Win a free vacation to the Bahamas! Reply now to claim.",
    "Act now! This deal expires in 24 hours.",
    "Congratulations! You've won a $1000 gift card. Click here to claim your prize!",
    "URGENT! Your account has been compromised. Verify your details immediately.",
    "You have been selected for a special offer. Don't miss out!",
]

ham_messages = [
    "Hi, are we still on for the meeting tomorrow?",
    "Can you send me the files for the project?",
    "Thanks for your help on the recent project.",
    "Let's catch up this weekend. Let me know when you're free.",
    "Please review the attached document for our meeting agenda.",
]

# Generate dataset
num_entries = 10001
data = {
    "Message": [random.choice(spam_messages if i % 2 == 0 else ham_messages) for i in range(num_entries)],
    "Prediction": [1 if i % 2 == 0 else 0 for i in range(num_entries)],  # Alternating spam (1) and ham (0)
}

# Create a DataFrame and save it as CSV
dataset = pd.DataFrame(data)
dataset.to_csv("spam_text_large_dataset.csv", index=False)
print("Large dataset generated and saved as 'spam_text_large_dataset.csv'.")

