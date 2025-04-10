# Making a box plot 
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Using Artificial Neural Networks for Regression in Python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Date</th>\n",
       "      <th>StationName</th>\n",
       "      <th>Tmax</th>\n",
       "      <th>Tmin</th>\n",
       "      <th>Tav</th>\n",
       "      <th>Tdew</th>\n",
       "      <th>Rs</th>\n",
       "      <th>ws</th>\n",
       "      <th>RHav</th>\n",
       "      <th>PM</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>2002-08-06</td>\n",
       "      <td>Jay</td>\n",
       "      <td>23.789</td>\n",
       "      <td>21.699</td>\n",
       "      <td>22.7440</td>\n",
       "      <td>23.093000</td>\n",
       "      <td>10.363118</td>\n",
       "      <td>0.820694</td>\n",
       "      <td>72.5515</td>\n",
       "      <td>2.328837</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2002-08-07</td>\n",
       "      <td>Jay</td>\n",
       "      <td>25.638</td>\n",
       "      <td>22.445</td>\n",
       "      <td>24.0415</td>\n",
       "      <td>23.747250</td>\n",
       "      <td>23.829095</td>\n",
       "      <td>0.966647</td>\n",
       "      <td>71.9450</td>\n",
       "      <td>4.223238</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>2002-08-08</td>\n",
       "      <td>Jay</td>\n",
       "      <td>23.687</td>\n",
       "      <td>17.266</td>\n",
       "      <td>20.4765</td>\n",
       "      <td>21.347792</td>\n",
       "      <td>21.078407</td>\n",
       "      <td>1.281092</td>\n",
       "      <td>69.8700</td>\n",
       "      <td>3.688828</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>2002-08-09</td>\n",
       "      <td>Jay</td>\n",
       "      <td>21.780</td>\n",
       "      <td>13.821</td>\n",
       "      <td>17.8005</td>\n",
       "      <td>17.464292</td>\n",
       "      <td>27.385210</td>\n",
       "      <td>1.215175</td>\n",
       "      <td>66.1700</td>\n",
       "      <td>4.081806</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>2002-08-10</td>\n",
       "      <td>Jay</td>\n",
       "      <td>22.865</td>\n",
       "      <td>14.743</td>\n",
       "      <td>18.8040</td>\n",
       "      <td>18.946958</td>\n",
       "      <td>24.918379</td>\n",
       "      <td>1.435128</td>\n",
       "      <td>66.1685</td>\n",
       "      <td>4.052152</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0        Date StationName    Tmax    Tmin      Tav       Tdew  \\\n",
       "0           1  2002-08-06         Jay  23.789  21.699  22.7440  23.093000   \n",
       "1           2  2002-08-07         Jay  25.638  22.445  24.0415  23.747250   \n",
       "2           3  2002-08-08         Jay  23.687  17.266  20.4765  21.347792   \n",
       "3           4  2002-08-09         Jay  21.780  13.821  17.8005  17.464292   \n",
       "4           5  2002-08-10         Jay  22.865  14.743  18.8040  18.946958   \n",
       "\n",
       "          Rs        ws     RHav        PM  \n",
       "0  10.363118  0.820694  72.5515  2.328837  \n",
       "1  23.829095  0.966647  71.9450  4.223238  \n",
       "2  21.078407  1.281092  69.8700  3.688828  \n",
       "3  27.385210  1.215175  66.1700  4.081806  \n",
       "4  24.918379  1.435128  66.1685  4.052152  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "DailyETdata=pd.read_csv('DailyETdata.csv')\n",
    "DailyETdata.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Unnamed: 0     0\n",
       "Date           0\n",
       "StationName    0\n",
       "Tmax           0\n",
       "Tmin           0\n",
       "Tav            0\n",
       "Tdew           0\n",
       "Rs             0\n",
       "ws             0\n",
       "RHav           0\n",
       "PM             0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking missing values again after the treatment\n",
    "DailyETdata.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(149555, 1)\n",
      "(149555, 1)\n",
      "(64095, 1)\n",
      "(64095, 1)\n"
     ]
    }
   ],
   "source": [
    "# Separate Target Variable and Predictor Variables\n",
    "TargetVariable=['PM']\n",
    "Predictors=['Rs']\n",
    "X=DailyETdata[Predictors].values\n",
    "y=DailyETdata[TargetVariable].values\n",
    " \n",
    "### Sandardization of data ###\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "PredictorScaler=StandardScaler()\n",
    "TargetVarScaler=StandardScaler()\n",
    " \n",
    "# Storing the fit object for later reference\n",
    "PredictorScalerFit=PredictorScaler.fit(X)\n",
    "TargetVarScalerFit=TargetVarScaler.fit(y)\n",
    " \n",
    "# Generating the standardized values of X and y\n",
    "X=PredictorScalerFit.transform(X)\n",
    "y=TargetVarScalerFit.transform(y)\n",
    " \n",
    "# Split the data into training and testing set\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    " \n",
    "# Quick sanity check with the shapes of Training and testing datasets\n",
    "print(X_train.shape)\n",
    "print(y_train.shape)\n",
    "print(X_test.shape)\n",
    "print(y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "XX = DailyETdata[Predictors].values\n",
    "YY = DailyETdata[TargetVariable].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "149555/149555 [==============================] - 28s 187us/step - loss: 0.1702\n",
      "Epoch 2/5\n",
      "149555/149555 [==============================] - 27s 179us/step - loss: 0.1643\n",
      "Epoch 3/5\n",
      "149555/149555 [==============================] - 26s 177us/step - loss: 0.1643\n",
      "Epoch 4/5\n",
      "149555/149555 [==============================] - 27s 179us/step - loss: 0.1642\n",
      "Epoch 5/5\n",
      "149555/149555 [==============================] - 28s 186us/step - loss: 0.1642\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.callbacks.History at 0x24b801d07f0>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# importing the libraries\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense\n",
    " \n",
    "# create ANN model\n",
    "model = Sequential()\n",
    " \n",
    "# Defining the Input layer and FIRST hidden layer, both are same!\n",
    "model.add(Dense(units=5, input_dim=1, kernel_initializer='normal', activation='relu'))\n",
    " \n",
    "# Defining the Second layer of the model\n",
    "# after the first layer we don't have to specify input_dim as keras configure it automatically\n",
    "model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))\n",
    " \n",
    "# The output neuron is a single fully connected node \n",
    "# Since we will be predicting a single number\n",
    "model.add(Dense(1, kernel_initializer='normal'))\n",
    " \n",
    "# Compiling the model\n",
    "model.compile(loss='mean_squared_error', optimizer='adam')\n",
    " \n",
    "# Fitting the ANN to the Training set\n",
    "model.fit(X_train, y_train ,batch_size = 5, epochs = 5, verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1 Parameters: batch_size: 5 - epochs: 5 Accuracy: 37.724826851052846\n"
     ]
    }
   ],
   "source": [
    "# Defining a function to find the best parameters for ANN\n",
    "def FunctionFindBestParams(X_train, y_train, X_test, y_test):\n",
    "    \n",
    "    # Defining the list of hyper parameters to try\n",
    "    batch_size_list=[5]\n",
    "    epoch_list  =   [5]\n",
    "    \n",
    "    import pandas as pd\n",
    "    SearchResultsData=pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])\n",
    "    \n",
    "    # initializing the trials\n",
    "    TrialNumber=0\n",
    "    for batch_size_trial in batch_size_list:\n",
    "        for epochs_trial in epoch_list:\n",
    "            TrialNumber+=1\n",
    "            # create ANN model\n",
    "            model = Sequential()\n",
    "            # Defining the first layer of the model\n",
    "            model.add(Dense(units=5, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))\n",
    " \n",
    "            # Defining the Second layer of the model\n",
    "            model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))\n",
    " \n",
    "            # The output neuron is a single fully connected node \n",
    "            # Since we will be predicting a single number\n",
    "            model.add(Dense(1, kernel_initializer='normal'))\n",
    " \n",
    "            # Compiling the model\n",
    "            model.compile(loss='mean_squared_error', optimizer='adam')\n",
    " \n",
    "            # Fitting the ANN to the Training set\n",
    "            model.fit(X_train, y_train ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)\n",
    " \n",
    "            MAPE = np.mean(100 * (np.abs(y_test-model.predict(X_test))/y_test))\n",
    "            \n",
    "            # printing the results of the current iteration\n",
    "            print(TrialNumber, 'Parameters:','batch_size:', batch_size_trial,'-', 'epochs:',epochs_trial, 'Accuracy:', 100-MAPE)\n",
    "            \n",
    "            SearchResultsData=SearchResultsData.append(pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial), 100-MAPE]],\n",
    "                                                                    columns=['TrialNumber', 'Parameters', 'Accuracy'] ))\n",
    "    return(SearchResultsData)\n",
    " \n",
    " \n",
    "######################################################\n",
    "# Calling the function\n",
    "ResultsData=FunctionFindBestParams(X_train, y_train, X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Visualizing the results of parameter trials for ANN\n",
    "#%matplotlib inline\n",
    "#ResultsData.plot(x='Parameters', y='Accuracy', figsize=(15,10), kind='line')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fitting the ANN to the Training set\n",
    "model.fit(X_train, y_train ,batch_size = 5, epochs = 5, verbose=0)\n",
    "\n",
    "# Generating Predictions on testing data\n",
    "Predictions=model.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This graph shows that the best set of parameters are batch_size=15 and epochs=5. \n",
    "Next step is to train the model with these parameters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Rs</th>\n",
       "      <th>PM</th>\n",
       "      <th>PredictedPM</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>13.242572</td>\n",
       "      <td>2.290138</td>\n",
       "      <td>2.429279</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>24.427680</td>\n",
       "      <td>4.217784</td>\n",
       "      <td>3.796140</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>13.513707</td>\n",
       "      <td>2.768181</td>\n",
       "      <td>2.466063</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>15.979637</td>\n",
       "      <td>3.012858</td>\n",
       "      <td>2.799663</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>22.061284</td>\n",
       "      <td>4.137008</td>\n",
       "      <td>3.574643</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Rs        PM  PredictedPM\n",
       "0  13.242572  2.290138     2.429279\n",
       "1  24.427680  4.217784     3.796140\n",
       "2  13.513707  2.768181     2.466063\n",
       "3  15.979637  3.012858     2.799663\n",
       "4  22.061284  4.137008     3.574643"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# Scaling the predicted Price data back to original scale\n",
    "Predictions=TargetVarScalerFit.inverse_transform(Predictions)\n",
    "\n",
    "# Scaling the y_test Price data back to original scale\n",
    "y_test_orig=TargetVarScalerFit.inverse_transform(y_test)\n",
    "\n",
    "# Scaling the test data back to original scale\n",
    "Test_Data=PredictorScalerFit.inverse_transform(X_test)\n",
    "\n",
    "TestingData=pd.DataFrame(data=Test_Data, columns=Predictors)\n",
    "TestingData['PM']=y_test_orig\n",
    "TestingData['PredictedPM']=Predictions\n",
    "TestingData.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Accuracy of ANN model is: 89.10999862366225\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Rs</th>\n",
       "      <th>PM</th>\n",
       "      <th>PredictedPM</th>\n",
       "      <th>APE</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>13.242572</td>\n",
       "      <td>2.290138</td>\n",
       "      <td>2.429279</td>\n",
       "      <td>6.075646</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>24.427680</td>\n",
       "      <td>4.217784</td>\n",
       "      <td>3.796140</td>\n",
       "      <td>9.996812</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>13.513707</td>\n",
       "      <td>2.768181</td>\n",
       "      <td>2.466063</td>\n",
       "      <td>10.913968</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>15.979637</td>\n",
       "      <td>3.012858</td>\n",
       "      <td>2.799663</td>\n",
       "      <td>7.076178</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>22.061284</td>\n",
       "      <td>4.137008</td>\n",
       "      <td>3.574643</td>\n",
       "      <td>13.593530</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Rs        PM  PredictedPM        APE\n",
       "0  13.242572  2.290138     2.429279   6.075646\n",
       "1  24.427680  4.217784     3.796140   9.996812\n",
       "2  13.513707  2.768181     2.466063  10.913968\n",
       "3  15.979637  3.012858     2.799663   7.076178\n",
       "4  22.061284  4.137008     3.574643  13.593530"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Computing the absolute percent error\n",
    "APE=100*(abs(TestingData['PM']-TestingData['PredictedPM'])/TestingData['PM'])\n",
    "TestingData['APE']=APE\n",
    "\n",
    "print('The Accuracy of ANN model is:', 100-np.mean(APE))\n",
    "TestingData.head()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Finding the accuracy of the model\n",
    "Using the final trained model, now we are generating the prediction error for each row in testing data as the Absolute Percentage Error. \n",
    "Taking the average for all the rows is known as Mean Absolute Percentage Error(MAPE).\n",
    "\n",
    "The accuracy is calculated as 100-MAPE."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

