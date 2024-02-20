This folder contains codes and data for a sentiment classification of IMBD reviews with the SVM model.
Data is a set of 25,000 highly polar movie reviews for training and 25,000 for testing. Obtained from https://ai.stanford.edu/~amaas/data/sentiment/

This code perform the process of training a SVM with hyperparameter tuning using grid search. It reads txt 25000 txt files from the training dataset, extract their content and label them as 'positive' or 'negative' as target variable. Then, it prerprocess the text by converting text to lowercase, removing punctuation, removes stop words using NLTK's English stop words, and applies stemming using the Porter stemming algorithm. Due to the use of a large dataset, the model training process involves fearture selection ('SelectKBest') and a linear SVM classifier ('LinearSVC) with grid search for hyperparameter tuning.

The results of grid search are as follows.
                                              params  mean_test_score (f1 score)
rank_test_score                                                      
1                 {'k_best__k': 1000, 'svm__C': 0.1}         0.875461
2                  {'k_best__k': 1000, 'svm__C': 10}         0.875460
2                   {'k_best__k': 1000, 'svm__C': 1}         0.875460
4                  {'k_best__k': 500, 'svm__C': 0.1}         0.870159
5                   {'k_best__k': 500, 'svm__C': 10}         0.870159
5                    {'k_best__k': 500, 'svm__C': 1}         0.870159
7                  {'k_best__k': 100, 'svm__C': 0.1}         0.835369
8                   {'k_best__k': 100, 'svm__C': 10}         0.835337
9                    {'k_best__k': 100, 'svm__C': 1}         0.835292
10               {'k_best__k': 10000, 'svm__C': 0.1}         0.814981
11                 {'k_best__k': 10000, 'svm__C': 1}         0.812318
12                {'k_best__k': 10000, 'svm__C': 10}         0.812146
13                {'k_best__k': 5000, 'svm__C': 0.1}         0.809390
14                  {'k_best__k': 5000, 'svm__C': 1}         0.807641
15                 {'k_best__k': 5000, 'svm__C': 10}         0.807413
16                    {'k_best__k': 10, 'svm__C': 1}         0.764883
16                   {'k_best__k': 10, 'svm__C': 10}         0.764883
18                  {'k_best__k': 10, 'svm__C': 0.1}         0.764856

Best parameters: {'k_best__k': 1000, 'svm__C': 0.1}
Evaluation Metrics:
Accuracy: 0.87412
Precision: 0.8671586715867159
Recall: 0.8836
F1 Score: 0.8753021357530612
