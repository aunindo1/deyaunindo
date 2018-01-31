from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
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
    
    np.savetxt('treeotp.txt', zip(clf.predict(x_test)),fmt="%i")
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
    import pydotplus
    clf = tree.DecisionTreeClassifier().fit(p, b) 
    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_pdf("Dtree.pdf")
def main():

    # iris = datasets.load_iris()

    # Separate 40% of data for testing

    # Cross validation
    
    y,x=read_file('input.csv')

    b,a=read_file('testing_data.csv')

    x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=0.4, random_state=0)

    clf = tree.DecisionTreeClassifier()
        
    cross_validation(clf, x_train, y_train)

    fitted_clf = fit_classifier(clf, x_train, y_train)

    test_model(fitted_clf, x_test, y_test)

    get_misclassified(fitted_clf, a, b)
    
    plotter(x_train,y_train)
    
if __name__ == "__main__":
   main()


