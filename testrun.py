# Running with the data: Example of how to call the functions
# A Data class is created so that the user doesn't need to keep track of the all the data structures involved
# To run, just call on it's functions.

#1. create an instance of the Data class
a = Data()

#2a. Prepare the training data. This function takes the following parameters: 
#   - filepath = the path & name to your training dataset
#   - y_name = the name of the column that you want to classify the data into
#   - typecont = a list of boolean values that specify whether each column of your dataset is discrete or continuous
a.prepare_train('data1/q3.csv', ' is spam', [False, False, False, False, False, False, True, True, False])

#2b. If you want to drop certain columns from calculation, it is easier to print out the x_names field in your instance
# After that, you can call drop_columns()
#   - column_names = a list of string values that specify the column names to drop
print(a.x_names)
a.drop_columns([' has my name', ' has sig'])

#3. Train your data. The data instance will input all the necessary parameters into the actual NaiveBayes algorithm.
a.run_train()

#4. Print the parameters of your training.
print()
a.print_parameters()

#5. Prepare the testing data. This function takes just one parameter as it is assumed the columns are the same as the training set: 
#   - filepath = the path & name to your testing dataset
print()
a.prepare_test('data1/q3b.csv')

#6. Make predictions with the test data you prepared before.
a.run_test()

#7. Get the number of entries that you predicted correctly vs incorrectly. print_accuracy will give you confusion matrix. 
a.print_accuracy()

