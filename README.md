Skin Condition Classifier

This project aims to create both linear regression and neural network models to be able to classify skin conditionsbased on image data.

skin.py is the main python script file that takes in the dataset, creates the logistic regression and neural network models with the ability to classify these skin conditions, then present the results of each of the models for comparison. 

class_distribution.png is a bar chart that represents the amount of times each condition appears in the dataset.

confusion_matrix_lr.png is a cofusion matrix that shows the true/false positives/negatives produced by the linear regression model performing predictions on the dataset.

confusion_matrix_nn.png is a confusion matrix that shows the true/false positives/negatives produced by the neural network model when it performed predictions on the dataset.

model_comparison.csv shows a tabular representation of the comparison metrics for each of the models used on the dataset. This includes accuracy, precision, recall, F1, and training time.

nn_training_curves.png demonstrates the loss and accuracy over time as the neural network is being trained in the form of two side-by-side line graphs.

sample_images.png include one image of each of the skin conditions we want the models to be able to classify.
