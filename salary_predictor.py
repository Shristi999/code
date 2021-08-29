#to import dataset
import pandas
#to other data types into np.array
import numpy
#to save data as a file
import joblib  


#to load the dataset
db = pandas.read_csv('SalaryData2.csv')   
print("The type of the dataset = " , type(db))


# assign the dependent variable
#assign the independent variable/ feature.
y = db["Salary"]   
x = db["YearsExperience"]   
print("The dataset is = ", db)

print("The type of dependent variable before = " ,type(x))

#values to convert pandas series into numpy array bcoz fit fn. made for arrays
X = x.values  
print("The type of dependent variable after = ",type(X))
X = X.reshape(30,1)

from sklearn.linear_model import LinearRegression

# this fn. is like a mind which process everything
mind = LinearRegression()  
#model trained
mind.fit(X,y)   
# to predict the value

print("---------------------Your Model is Trained , ready for testing---------------------------")
t1 = float(input("Enter the value_1(YOE)")) 
r1 = mind.predict([[t1]])
print("Estimated salary for the ",t1," year of exp. is ",r1)
t2 = float(input("Enter the value_2(YOE)")) 
r2 = mind.predict([[t2]]) 
print("Estimated salary for the ",t2," year of exp. is ",r2)


#62647.05325234/63218.0 * 100  # accuracy

#acc = (((r1/y[7]) * 100) + ((r2/y[24]) * 100)) / 2
#print("The accuracy of your model is = ",acc,"%")

#Linear fn. y= b+ wx
# through this fn. bias and weight is calculated

print("The coefficient of the equation is = " ,mind.coef_)

joblib.dump(mind,'sal_pred.pk1')
type('sal_pred.pk1')

