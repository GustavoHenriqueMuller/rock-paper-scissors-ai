import sys
import numpy
import time

import pickle

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from rpscv import imgproc as imp
from rpscv import utils

def dt(t0):
    return round(time.time() - t0, 2)

def main():
    train()

def train():
    # Defining constants
    randomSeed = 42
    classifierFileName = 'classifier.pkl'
    nSplits = 5
    pcaNumberComponents = [40]
    classifierGamma = [.0001, .001, .01]
    classifierC = [1, 10, 100]
    scoringMethod = 'f1_micro'
    numberJobs = 4
    
    t0 = time.time()

    # Generate image data from stored images
    print('+{}s: Generating image data'.format(dt(t0)))
    features, labels = imp.generateGrayFeatures(verbose = False, randomSeed = randomSeed)

    unique, count = numpy.unique(labels, return_counts = True)

    # Print the number of traning images for each label
    for i, label in enumerate(unique):
        print('  {}: {} images'.format(utils.gestureTexts[label], count[i]))

    # Generate test set
    print('+{}s: Generating test set'.format(dt(t0)))
    stratifiedKFold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = randomSeed)

    for train_index, test_index in stratifiedKFold.split(features, labels):
        features_train = features[train_index]
        features_test = features[test_index]
        labels_train = labels[train_index]
        labels_test = labels[test_index]

    # Define pipeline parameters
    print('+{}s: Defining pipeline'.format(dt(t0)))
    steps = [('pca', PCA()), ('clf', SVC(kernel = 'rbf'))]
    pipeline = Pipeline(steps)
    
    # Define cross-validation parameters
    print('+{}s: Defining cross-validation'.format(dt(t0)))
    crossValidator = StratifiedKFold(n_splits = nSplits, shuffle = True, random_state = randomSeed)

    # Define grid-search parameters
    print('+{}s: Defining grid search'.format(dt(t0)))
    gridParams = dict(pca__n_components = pcaNumberComponents,
                       clf__gamma = classifierGamma,
                       clf__C = classifierC)
					   
    grid = GridSearchCV(pipeline, gridParams, scoring = scoringMethod,
                        n_jobs = numberJobs, refit = True, cv = crossValidator, verbose = 1)

    print('Grid search parameters:')
    print(grid)

    # Fit the classifier
    trainT0 = time.time()
    print('+{}s: Fitting classifier'.format(dt(t0)))
    grid.fit(features_train, labels_train)
    trainDt = time.time() - trainT0

    # Print the results of the grid search cross-validation
    cvres = grid.cv_results_
    print('Cross-validation results:')
    for score, std, params in zip(cvres['mean_test_score'],
            cvres['std_test_score'], cvres['params']):

        print('  {}, {}, {}'.format(round(score, 4), round(std, 5), params))

    # Print the best score and best parameters from the grid-search
    print('Grid search best score: {}'.format(grid.best_score_))
    print('Grid search best parameters:')
    for key, value in grid.best_params_.items():
        print('  {}: {}'.format(key, value))

    # Validate classifier on test set
    print('+{}s: Validating classifier on test set'.format(dt(t0)))
    pred = grid.predict(features_test)
    score = f1_score(labels_test, pred, average = 'micro')

    print('Classifier f1-score on test set: {}'.format(score))

    print('Confusion matrix:')
    print(confusion_matrix(labels_test, pred))

    print('Classification report:')

    tn = [utils.gestureTexts[i] for i in range(3)]
    print(classification_report(labels_test, pred, target_names=tn))

    # Write classifier to a .pkl file
    print('+{}s: Writing classifier to {}'.format(dt(t0), classifierFileName))
    with open(classifierFileName, 'wb') as f:
        f.flush()
        pickle.dump(grid, f)

    print('+{}s: Done!'.format(dt(t0)))

    return grid.best_score_, score, trainDt

if __name__ == '__main__':
    train()
