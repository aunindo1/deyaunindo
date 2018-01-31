from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
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

    np.savetxt('svm_opt.txt', zip(clf.predict(x_test)),fmt="%i")

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

    X = p[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
    y = b

    h = .6  # step size in the mesh

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    clf = svm.SVC(kernel='linear', C=C).fit(X, y)


    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel']



    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=0.6, hspace=0.6)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('No. of front slash')
    plt.ylabel('No. of dots')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles)

    plt.show()
   
def main():

    # iris = datasets.load_iris()

    # Separate 40% of data for testing

    # Cross validation
    
    y,x=read_file('input.csv')

    b,a=read_file('testing_data.csv')

    x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=0.4, random_state=0)

    clf = svm.SVC(kernel='linear', C=1)
        
    cross_validation(clf, x_train, y_train)

    fitted_clf = fit_classifier(clf, x_train, y_train)

    test_model(fitted_clf, x_test, y_test)

    get_misclassified(fitted_clf, a, b)
    
    plotter(x_train,y_train)
    
if __name__ == "__main__":
   main()


