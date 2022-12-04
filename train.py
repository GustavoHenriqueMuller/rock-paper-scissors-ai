import sys
import numpy as np

# Random State
rs = 42

# Classifier output .pkl filename
pklFilename = 'clf.pkl'

# Number of folds of Stratified KFold cross-validation
n_splits = 5

# Grid Search parameters
pca__n_components = [40] # Number of components of Principal Component Analysis
clf__gamma = np.logspace(-4, -2, 3) # [.0001, .001, .01]
clf__C = np.logspace(0, 2, 3) # [1, 10, 100]
scoring = 'f1_micro'

# The n_jobs parameter controls the number of CPU cores to use in parallel for
# training the machine learning model.
n_jobs = 4

def train(showCvScore = True):
    import time
    t0 = time.time()

    def dt():
        return round(time.time() - t0, 2)

    print('+{}s: Importing libraries'.format(dt()))

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

    # Generate image data from stored images
    print('+{}s: Generating image data'.format(dt()))
    features, labels = imp.generateGrayFeatures(verbose = False, randomSeed = rs)

    unique, count = np.unique(labels, return_counts = True)

    # Print the number of traning images for each label
    for i, label in enumerate(unique):
        print('  {}: {} images'.format(utils.gestureTxt[label], count[i]))

    # Generate test set
    print('+{}s: Generating test set'.format(dt()))
    sssplit = StratifiedShuffleSplit(n_splits = 1, test_size = .15, random_state = rs)

    for train_index, test_index in sssplit.split(features, labels):
        features_train = features[train_index]
        features_test = features[test_index]
        labels_train = labels[train_index]
        labels_test = labels[test_index]

    # Define pipeline parameters
    print('+{}s: Defining pipeline'.format(dt()))
    steps = [('pca', PCA()), ('clf', SVC(kernel = 'rbf'))]
    pipe = Pipeline(steps)

    # Define cross-validation parameters
    print('+{}s: Defining cross-validation'.format(dt()))
    cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = rs)

    # Define grid-search parameters
    print('+{}s: Defining grid search'.format(dt()))
    grid_params = dict(pca__n_components = pca__n_components,
                       clf__gamma = clf__gamma,
                       clf__C = clf__C)
					   
    grid = GridSearchCV(pipe, grid_params, scoring = scoring, n_jobs=n_jobs, refit = True, cv = cv, verbose = 1)

    print('Grid search parameters:')
    print(grid)

    # Fit the classifier
    t0_train = time.time()
    print('+{}s: Fitting classifier'.format(dt()))
    grid.fit(features_train, labels_train)
    dt_train = time.time() - t0_train

    if showCvScore:
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
    print('+{}s: Validating classifier on test set'.format(dt()))
    pred = grid.predict(features_test)
    score = f1_score(labels_test, pred, average='micro')
    print('Classifier f1-score on test set: {}'.format(score))
    print('Confusion matrix:')
    print(confusion_matrix(labels_test, pred))
    print('Classification report:')
    tn = [utils.gestureTxt[i] for i in range(3)]
    print(classification_report(labels_test, pred, target_names=tn))

    # Write classifier to a .pkl file
    print('+{}s: Writing classifier to {}'.format(dt(), pklFilename))
    with open(pklFilename, 'wb') as f:
        f.flush()
        pickle.dump(grid, f)

    print('+{}s: Done!'.format(dt()))

    return grid.best_score_, score, dt_train

if __name__ == '__main__':
    # Read command line arguments
    argv = sys.argv
    showCvScore = True

    if len(sys.argv) > 1:
        for arg in argv[1:]:
            if arg == '--no-cv-score':
                showCvScore = False

    train(showCvScore = showCvScore)
