from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets	
from sklearn import svm    	
import numpy as np
import matplotlib.pyplot as plt 

def cross_validation(clf, data, target):
    
    scores = cross_val_score(clf, data, target, cv=2)

    print "Cross Validation Scores"
    
    print scores


def test_model(clf, data, target):

    scores = clf.score(data, target)

    print "Test Scores"

    print scores

def fit_classifier(clf, data, target):

    fitted_clf = clf.fit(data, target)

    return fitted_clf

def get_misclassified(clf, x_test, y_test):

    misclassified = np.where(y_test != clf.predict(x_test))

    print "Misclassified intance indices"
    
    print misclassified

    np.savetxt('regrotp.txt', zip(clf.predict(x_test)),fmt="%i")

def read_file(h):
    import csv
    import numpy as np
    arr1 = []
    arr2 = []
    arr3 = []
    no_of_Fslash = []
    no_of_para = []
    l_of_add = []
    with open(h, 'rb') as f:
         reader = csv.reader(f, delimiter=',')
         for row in reader:
             arr1.append(row[0:3])
             no_of_Fslash.append(row[0].count('/'))
             no_of_para.append(row[0].count('.'))
             l_of_add.append(len(row[0]))
             if row[3]=='benign':
     		arr2.append(1)
	     elif row[3]=='malicious':
		  arr2.append(0)
    b= np.loadtxt(arr2,ndmin=2)
    q= np.loadtxt(no_of_Fslash, ndmin=2)
    r= np.loadtxt(no_of_para, ndmin=2)
    s= np.loadtxt(l_of_add, ndmin=2)
    p= np.hstack((q,r,s))
    p = np.delete(p, (0), axis=0)
    return [b,p]
    

def plotter(p,b):
    
    # import some data to play with
    X = p[:, :2]  # we only take the first two features.
    Y = b

    h = .6  # step size in the mesh

    logreg = LogisticRegression(C=1e5)

    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(X, Y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('No. of front slash')
    plt.ylabel('No. of dots')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()
def main():

    # iris = datasets.load_iris()

    # Separate 40% of data for testing

    # Cross validation
    
    y,x=read_file('input.csv')

    b,a=read_file('testing_data.csv')

    x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=0.4, random_state=0)

    clf = LogisticRegression(C=10)

        
    cross_validation(clf, x_train, y_train)

    fitted_clf = fit_classifier(clf, x_train, y_train)

    test_model(fitted_clf, x_test, y_test)

    get_misclassified(fitted_clf, a, b)
    
    plotter(x_train,y_train)
    
if __name__ == "__main__":
   main()


