# Data encapsulation class

class Data:
    # Class variables: 
    # x_names = 1D list of strings containing the features' column names to loop over
    # y_name = string containing the out classification column name
    # x_typecont = 1D list of of booleans that tell whether each features column to loop over is continuous(true) or discrete(false)
    # y_labels = 1D list of unique classification categories
    # y_values = 1D list of probabilities of each classification category, aka P(y), in y_labels
    # x_labels = 2D dataframe with x_names number of columns and the unique classification categories of each feature
    # x_values = 2D dataframe with x_names number of columns and the probabilities of that feature given the classification column probability, 
    #            probability, aka P(xn|y)
    
    def prepare_train(self, filepath, y_name, typecont):
        self.df_train = pd.read_csv(filepath)
        self.y_name = y_name
        self.x_names = self.df_train.columns
        self.x_typecont = typecont
        dropindex = self.x_names.get_loc(y_name)
        self.x_names = self.x_names.drop(self.y_name)
        self.x_typecont.pop(dropindex)
        
    def prepare_test(self, filepath):
        self.df_test = pd.read_csv(filepath)
        
    def drop_columns(self, column_names):
        for name in column_names:
            dropindex = self.x_names.get_loc(name)
            self.x_names = self.x_names.drop(name)
            self.x_typecont.pop(dropindex)
    
    def run_train(self):
        self.y_labels, self.y_values, self.x_labels, self.x_values = naiveBayes(self.df_train, self.x_names, self.y_name, self.x_typecont)
    
    def run_test(self):
        self.results = classify(self.df_test, self.x_names, self.y_name, self.y_labels, self.y_values, self.x_labels, self.x_values, self.x_typecont)
    
    def print_parameters(self):
        print('P(Y):')
        print(a.y_labels)
        print(a.y_values)
        print('\nP(X|Y):')
        print('maximum likelihood parameters sorting by y, then x: aka, (x=f|y=f), (x=t|y=f), (x=f|y=t), (x=t|y=t)')
        print(a.x_labels)
        print(a.x_values)
        
    def print_accuracy(self):
        y_actual = pd.Series(self.df_test[self.y_name], name='Actual')
        y_predicted = pd.Series(self.results[self.y_name], name='Predicted')
        print(pd.crosstab(y_actual, y_predicted))