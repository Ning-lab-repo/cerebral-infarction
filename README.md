# cerebral-infarction
# Note  
Cerebral infarction is a common type of stroke that poses a significant health burden globally. If not addressed promptly, transient ischemic attacks can progress to cerebral infarction. Therefore, we employed machine learning models to predict these two conditions early in the population.  

# The description of cerebral-infarction source codes  
  Match and transpose data.py The code is used for matching the test results and health status of the same participants by ID in two files; removing duplicate entries for the same individual, retaining only one row; and converting the file format into a Python-readable format.  
  Delete missing values.py The code is used for removing samples with excessive missing values and features with too many missing values, and then using KNN to impute the remaining missing values.  
  Search parameters.py This code is used to search for the optimal parameters of a machine learning model for subsequent analysis.  
  Multiple model ROC.py This code is used to plot the binary classification ROC curves of multiple machine learning models on a single graph.  
  Shap.py This code is used to explain the intermediate processes of a machine learning model and display the top ten features that contribute the most to the model.
  Peak chart.py This code is used to plot a peak chart for the four features with the highest contribution.  
  Binary classification ROC.py This code is used to re-plot the binary classification ROC curves using the top ten features with the highest contributions.  
  Stacked bar chart.py This code is used to create a bar chart that displays gender statistics grouped by health status.  
  Confusion matrix.py This code is used to plot the confusion matrix of the best-performing model to illustrate its effectiveness in distinguishing between the groups.  
  
# The description of cerebral-infarction  
Overall, the results indicate that the machine learning model can effectively predict cerebral infarction and transient ischemic attacks, showing a high AUC. The features with the highest contributions are mainly related to glucose metabolism, lipid metabolism, and liver metabolism.  
