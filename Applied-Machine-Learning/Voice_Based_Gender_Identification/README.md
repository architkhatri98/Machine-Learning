This project will identify whether the given voice data belongs to a male or a female. I tried model performance on Logistic Regression, Random Forest and SVM. I chose to select SVM as the best classifier based on similarity between test set and train set. Random Forest was overfitting on training set with score of 1.0 and therefore rejected. I then did some fine tuning along grid search to find the best parameters and best SVM kernel.
The best SVM kernel was found to be 'linear' and C value was found to be 0.67 and accuracy achieved is 0.97.

LINK:- https://www.kaggle.com/primaryobjects/voicegender
