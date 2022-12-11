Project to add weight to sklearn.ensemble

A personal project using numpy's api and scikit-learn's code.

Reference Site:   
https://numpy.org   
https://github.com/numpy/numpy   
https://scikit-learn.org   
https://github.com/scikit-learn/scikit-learn

# Dependencies
Installing scikit-learn   
Follow the dependencies of scikit-learn

# How to use it
WeightForestClassifier inherits RandomForestClassifier   
Use in a similar way to RandomForestClassifier   
Override the predict_proba function   
Use the weight_fit function to adjust the weight

If the correct DecisionTree exists within the forest and some labels are rare, this class will be useful (I hope so)   
The underlying idea is to add weight to each decision tree and adjust the weight through the weight_fit function   
A kind of learning rate, reward and punishment, exists as instance attributes, but you can ignore them and pass them to weight_fit as parameters.

See the code for more information
