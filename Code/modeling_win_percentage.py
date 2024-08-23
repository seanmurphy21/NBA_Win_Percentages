#!/usr/bin/env python
# coding: utf-8

# This file will be used to develop a model predicting win percentage from team statistics

# In[2]:


# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
sns.set_theme(style = 'white', palette = 'colorblind')


# In[3]:


# load in the data
nba_df = pd.read_csv('cleaned_nba_team_stats.csv')


# In[4]:


nba_df.head()


# In[5]:


# check out the shape
nba_df.shape


# In[6]:


# examine the correlations between the numeric variables; many high correlations, we should expect a penalized regression
# model to perform well
plt.figure(figsize = (15,12), dpi = 300)
heat_map = sns.heatmap(data = nba_df.corr(numeric_only = True), cmap = 'vlag')
plt.title('Correlation Map of the Numeric Variables', fontsize = 22);


# In[7]:


# split the data into predictors and labels
X = nba_df.drop(['Team', 'Made_Playoffs', 'Win_Percentage', 'Won_Title', 'Winning_Record', 'Highest_Net_Rating'], axis = 1)
y = nba_df['Win_Percentage']


# In[8]:


# split the data into a training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)


# In[9]:


# scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test) # only transform to prevent data leakage from training set


# We will try mulitple linear regression, support vector regressor, elastic net, and adaboost regressor with hyperparameter tuning

# In[11]:


# import models
from sklearn.linear_model import LinearRegression
lr_mod = LinearRegression()
from sklearn.svm import SVR
svr_mod = SVR()
from sklearn.linear_model import ElasticNet
enet_mod = ElasticNet(random_state = 1)
from sklearn.ensemble import AdaBoostRegressor
ada_mod = AdaBoostRegressor(random_state = 1)


# In[12]:


# set up the parameter grids for a grid search for each model type
lr_param_grid = {'fit_intercept': [True, False]}
    
svr_param_grid = {'kernel': ['linear', 'rbf'],
                  'gamma': ['scale', 'auto'],
                  'C': list(np.linspace(.0001,.01,10)),
                  'epsilon': list(np.linspace(.001,.1,10)),
                  'max_iter': [1000]}

enet_param_grid = {'alpha': list(np.linspace(.0001,.01,20)),
                   'l1_ratio': [0,0.2,0.4,0.6,0.8,1],
                   'max_iter': [10000]}

ada_param_grid = {'n_estimators': list(np.arange(150,251,25)),
                  'learning_rate': list(np.linspace(.01,10,8))}


# In[13]:


# import the grid search
from sklearn.model_selection import GridSearchCV


# In[14]:


# set up the grid search for each model
lr_grid_model = GridSearchCV(estimator = lr_mod,
                             param_grid = lr_param_grid,
                             scoring = 'neg_mean_squared_error',
                             cv = 10,
                             verbose = 1)


# In[15]:


# now for the SVR model
svr_grid_model = GridSearchCV(estimator = svr_mod,
                              param_grid = svr_param_grid,
                              scoring = 'neg_mean_squared_error',
                              cv = 10,
                              verbose = 1,return_train_score = True)


# In[16]:


# now for the elastic net model
enet_grid_model = GridSearchCV(estimator = enet_mod,
                               param_grid = enet_param_grid,
                               scoring = 'neg_mean_squared_error',
                               cv = 10,
                               verbose = 1,
                               return_train_score = True)


# In[17]:


# now for the AdaBoost model
ada_grid_model = GridSearchCV(estimator = ada_mod,
                              param_grid = ada_param_grid,
                              scoring = 'neg_mean_squared_error',
                              cv = 10,
                              verbose = 1,
                              return_train_score = True)


# In[18]:


# fit the linear regression model
lr_grid_model.fit(scaled_X_train, y_train)


# In[19]:


# fit the svr model
svr_grid_model.fit(scaled_X_train, y_train)


# In[20]:


# fit the enet model
enet_grid_model.fit(scaled_X_train, y_train)


# In[21]:


# fit the adaboost regressor model
ada_grid_model.fit(X_train, y_train) # no need to scale for tree-based method


# In[22]:


# find the index of the best model
cv_scores = [lr_grid_model.best_score_,
             svr_grid_model.best_score_,
             enet_grid_model.best_score_,
             ada_grid_model.best_score_]
cv_scores.index(np.max(cv_scores))


# In[23]:


# this means our best cross-validated RMSE is as follows
np.sqrt(abs(enet_grid_model.best_score_))


# In[162]:


# let's look at a heat map of the tuning parameter grid along with the cross-validated negative mean squared errors
enet_cv_results = pd.DataFrame(enet_grid_model.cv_results_)[['param_alpha', 'param_l1_ratio', 'mean_test_score']]
plt.figure(figsize = (4,6), dpi = 300)
sns.heatmap(data = enet_cv_results.pivot_table(index = 'param_alpha',
                                               columns = 'param_l1_ratio',
                                               values = 'mean_test_score'),
            cmap = 'viridis')
plt.title('Grid Search Results for Elastic Net Model')
plt.xlabel('L1 Ratio')
plt.ylabel('Alpha')
plt.savefig('grid_search_heatmap_enet.png', dpi = 300, bbox_inches = 'tight');


# In[24]:


# let's see what the honest accuracy looks like on the validation set
y_pred = enet_grid_model.predict(scaled_X_test) # using best model for predictions


# In[25]:


# import the metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[26]:


# honest mean absolute error
mean_absolute_error(y_true = y_test, y_pred = y_pred)


# In[27]:


# honest root mean squared error
np.sqrt(mean_squared_error(y_true = y_test, y_pred = y_pred))


# Our best model using all predictors can honestly predict a team's win percentage to within an average of 2.775% (MAE) and 3.447% (RMSE)

# But this model uses advanced statistics.  Let's see if we can model win percentage without using advanced stats

# In[30]:


# select all the predictors that do not include advanced statistics from the data frame
trad_stats_df = nba_df[['Team', 'Field_Goals_Made', 'Field_Goals_Attempted',
                        'Field_Goal_Percentage', '3-Pointers_Made', '3-Pointers_Attempted',
                        '3-Point_Field_Goal_Percentage', '2-Pointers_Made',
                        '2-Pointers_Attempted', '2-Point_Field_Goal_Percentage',
                        'Free_Throws_Made', 'Free_Throws_Attempted', 'Free_Throw_Percentage',
                        'Offensive_Rebounds', 'Defensive_Rebounds', 'Total_Rebounds', 'Assists',
                        'Steals', 'Blocks', 'Turnovers', 'Personal_Fouls', 'Points','Win_Percentage']]


# In[130]:


# examine the correlation between the numeric predictors; still many highly correlated
plt.figure(dpi = 300)
sns.heatmap(data = trad_stats_df.corr(numeric_only = True), cmap = 'vlag')
plt.title('Correlation Map of Traditional Statistics Predictors')
plt.savefig('trad_stat_corr_map.png', dpi = 300, bbox_inches = 'tight');


# In[32]:


# using the same process as above, we will split data into predictors and labels
trad_X = trad_stats_df.drop(['Team', 'Win_Percentage'], axis = 1)
trad_y = trad_stats_df['Win_Percentage']


# In[33]:


# perform the train test split
trad_X_train, trad_X_test, trad_y_train, trad_y_test = train_test_split(trad_X, trad_y, test_size=0.15, random_state=1)


# In[34]:


# scale the predictors
from sklearn.preprocessing import StandardScaler
trad_scaler = StandardScaler()
scaled_trad_X_train = trad_scaler.fit_transform(trad_X_train)
scaled_trad_X_test = trad_scaler.transform(trad_X_test) # only transform to prevent data leakage


# In[35]:


# import same models as before
from sklearn.linear_model import LinearRegression
lr_mod = LinearRegression()
from sklearn.svm import SVR
svr_mod = SVR()
from sklearn.linear_model import ElasticNet
enet_mod = ElasticNet(random_state = 1)
from sklearn.ensemble import AdaBoostRegressor
ada_mod = AdaBoostRegressor(random_state = 1)


# In[36]:


# set up the parameter grids for a grid search for each model type as before
lr_param_grid = {'fit_intercept': [True, False]}
    
svr_param_grid = {'kernel': ['linear', 'rbf'],
                  'gamma': ['scale', 'auto'],
                  'C': list(np.linspace(.01,1,10)),
                  'epsilon': list(np.linspace(.001,.1,10)),
                  'max_iter': [1000]}

enet_param_grid = {'alpha': list(np.linspace(.000001,.0001,20)),
                   'l1_ratio': [0,0.2,0.4,0.6,0.8,1],
                   'max_iter': [10000]}

ada_param_grid = {'n_estimators': list(np.arange(200,301,25)),
                  'learning_rate': list(np.linspace(.01,10,8))}


# In[37]:


# set up the grid search for each traditional statistics model
trad_lr_grid_model = GridSearchCV(estimator = lr_mod,
                                  param_grid = lr_param_grid,
                                  scoring = 'neg_mean_squared_error',
                                  cv = 10,
                                  verbose = 1)
trad_svr_grid_model = GridSearchCV(estimator = svr_mod,
                                   param_grid = svr_param_grid,
                                   scoring = 'neg_mean_squared_error',
                                   cv = 10,
                                   verbose = 1,
                                   return_train_score = True)
trad_enet_grid_model = GridSearchCV(estimator = enet_mod,
                                    param_grid = enet_param_grid,
                                    scoring = 'neg_mean_squared_error',
                                    cv = 10,
                                    verbose = 1,
                                    return_train_score = True)
trad_ada_grid_model = GridSearchCV(estimator = ada_mod,
                                   param_grid = ada_param_grid,
                                   scoring = 'neg_mean_squared_error',
                                   cv = 10,
                                   verbose = 1,
                                   return_train_score = True)


# In[38]:


# train the linear regression model
trad_lr_grid_model.fit(scaled_trad_X_train, trad_y_train)


# In[39]:


# train the svr model
trad_svr_grid_model.fit(scaled_trad_X_train, trad_y_train)


# In[40]:


# train the elastic net model
trad_enet_grid_model.fit(scaled_trad_X_train, trad_y_train)


# In[41]:


# train the adaboost model
trad_ada_grid_model.fit(trad_X_train, trad_y_train) # no need for scaling with AdaBoost


# In[42]:


# find the index of the best model
trad_cv_scores = [trad_lr_grid_model.best_score_,
                  trad_svr_grid_model.best_score_,
                  trad_enet_grid_model.best_score_,
                  trad_ada_grid_model.best_score_]
trad_cv_scores.index(np.max(trad_cv_scores))


# In[43]:


# this means our best cross-validated RMSE is as follows, for the SVR model this time
np.sqrt(abs(trad_svr_grid_model.best_score_))


# In[194]:


# let's look at a heat map of the tuning parameter grid for the SVR model
# along with the cross-validated negative mean squared errors
trad_svr_cv_results = pd.DataFrame(trad_svr_grid_model.cv_results_)[['param_C', 'param_epsilon', 'mean_test_score']]
plt.figure(figsize = (4,6), dpi = 300)
sns.heatmap(data = trad_svr_cv_results.pivot_table(index = 'param_epsilon',
                                                   columns = 'param_C',
                                                   values = 'mean_test_score',
                                                   # 'max' displays best score with these tuning parameter vals
                                                   # because there are other tuning parameters 
                                                   aggfunc = 'max'),
            cmap = 'viridis')
plt.title('Grid Search Results for SVR Model Using Traditional Statistics')
plt.xlabel('Epsilon')
plt.ylabel('C')
plt.savefig('grid_search_heatmap_svr.png', dpi = 300, bbox_inches = 'tight');


# In[44]:


# let's go ahead and get our honest performance metrics for the best model
trad_y_pred = trad_svr_grid_model.predict(scaled_trad_X_test)


# In[45]:


# first, mean absolute error
mean_absolute_error(y_true = trad_y_test, y_pred = trad_y_pred)


# In[46]:


# next, root mean squared error
np.sqrt(mean_squared_error(y_true = trad_y_test, y_pred = trad_y_pred))


# Thus, our best model using only traditional statistics still performs quite well, and can predict a team's win percentage to within an average of 5.164% (MAE) and 6.596% (RMSE)

# We want to get some idea of what the most important predictors are in each model

# In[49]:


# fit the elastic net model selected by cross-validation with advanced stats to the full dataset
# recall parameters selected for that model
enet_grid_model.best_params_


# In[50]:


# grab the X matrix of predictors for the full data set
advanced_X = nba_df.drop(['Team',
                          'Made_Playoffs',
                          'Win_Percentage',
                          'Won_Title',
                          'Winning_Record',
                          'Highest_Net_Rating'], axis = 1)
advanced_y = nba_df['Win_Percentage']


# In[51]:


# scale the predictors from the full model (including all rows)
advanced_scaler = StandardScaler()
scaled_advanced_X = advanced_scaler.fit_transform(X)


# In[52]:


# set up and fit the model
full_enet_model = ElasticNet(alpha = 0.0016631578947368423,
                             l1_ratio = 0.8, max_iter = 10000)
full_enet_model.fit(scaled_advanced_X, y)


# In[53]:


# first, let's see how many coefficients have shrunk to zero
full_enet_model.coef_


# In[54]:


# use boolean indexing to select the predictors with non-zero coefficients
nonzero_ind = full_enet_model.coef_ != 0
nonzero_predictors = X.columns[nonzero_ind]
# grab the coefficient values for these nonzero predictors
nonzero_coefs = full_enet_model.coef_[nonzero_ind]


# In[55]:


# grab the sorted indices of the absolute value of the coefficients
abs_sorted_ind = abs(nonzero_coefs).argsort()[::-1]


# In[56]:


# plot the predictors and the absolute values of their corresponding coefficients in the full model
plt.figure(figsize = (6,4), dpi = 300)
bar_plot = sns.barplot(x = nonzero_predictors[abs_sorted_ind],
                       y = abs(nonzero_coefs)[abs_sorted_ind], color = 'skyblue')
bar_plot.set_xticks(labels = ['Margin_of_Victory', 'Net_Rating', 'Average_Age', 'Opp FT per FGA',
                              'Simple_Rating_System', 'Field_Goal_Percentage', 'Blocks', 'Opp Effective FG%'],
                    ticks = bar_plot.get_xticks(), rotation = 90)
plt.title('Abs Value of Coefficients for Important Predictors in Advanced Stats Model')
plt.ylabel('Absolute Value of Coefficient')
plt.xlabel('Predictor')
plt.xticks(rotation = 90)
plt.savefig('enet_important_predictors.png', dpi = 300, bbox_inches = 'tight');


# In[57]:


# for partial dependence plots, it will be easier if we refit the model with a pipeline that scales the predictors
# set up a pipeline that scales the predictors and fits the desired elastic net model
from sklearn.pipeline import make_pipeline
full_enet_model_pipe = make_pipeline(StandardScaler(),
                                     ElasticNet(alpha = 0.0016631578947368423,
                                                l1_ratio = 0.8, max_iter = 10000))


# In[58]:


# fit the full elastic net model with the pipeline
full_enet_model_pipe.fit(X, y)


# In[59]:


# let's see the partial dependence plots for the three most important predictors here
from sklearn.inspection import PartialDependenceDisplay
plt.figure(figsize = (12,6), dpi = 300)
PartialDependenceDisplay.from_estimator(full_enet_model_pipe,
                                        X,
                                        features = ['Margin_of_Victory', 'Net_Rating', 'Average_Age'],
                                        kind = 'both',
                                        random_state = 1)
plt.suptitle('Partial Dependence Plots of Most Important Advanced Statistic Predictors')
plt.savefig('enet_partial_dependence.png', dpi = 300, bbox_inches = 'tight');


# Now, on to the SVR model for the traditional statistics subset

# In[61]:


# determine the most important predictors in the SVR model using permutation importance
from sklearn.inspection import permutation_importance
perm_importance_svr = permutation_importance(trad_svr_grid_model,
                                             scaled_trad_X_test,
                                             trad_y_test,scoring = 'neg_mean_squared_error',
                                             n_repeats = 20,random_state = 1)


# In[62]:


# grab the feature names
features = np.array(list(trad_X.columns))


# In[63]:


# sort the feature importances and grab their indices
sorted_importance_indices = perm_importance_svr.importances_mean.argsort()[::-1]


# In[64]:


# plot the feature importances
plt.figure(figsize = (6,4), dpi = 300)
sns.barplot(x = features[sorted_importance_indices],
            y = perm_importance_svr.importances_mean[sorted_importance_indices],
            color = 'skyblue')
plt.xticks(rotation = 90)
plt.title('Feature Importances for SVR Model Using Traditional Statistics')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.savefig('svr_feature_importances.png', dpi = 300, bbox_inches = 'tight');


# In[65]:


# fit the SVR model to the full traditional statistics data set
# recall the best parameters
trad_svr_grid_model.best_params_


# In[66]:


# make a pipeline that scales the predictors and fits the desired SVR model
from sklearn.pipeline import make_pipeline
# set up the full model
full_trad_svr_model = make_pipeline(StandardScaler(),
                                    SVR(C = 0.23, epsilon = .012, gamma = 'scale',
                                        kernel = 'rbf', max_iter = 1000))


# In[67]:


# fit the model
full_trad_svr_model.fit(trad_X, trad_y)


# In[68]:


# let's see how some of the most important features in this model relate to the predicted win percentage using PDPs
# (Partial Dependence Plots)
from sklearn.inspection import PartialDependenceDisplay
plt.figure(figsize = (12,6), dpi = 300)
PartialDependenceDisplay.from_estimator(full_trad_svr_model,
                                        trad_X,
                                        features = ['Turnovers', 'Field_Goals_Attempted', 'Steals'],
                                        kind = 'both',
                                        random_state = 1)
plt.suptitle('Partial Dependence Plots of Most Important Traditional Statistic Predictors')
plt.savefig('svr_partial_dependence.png', dpi = 300, bbox_inches = 'tight');

