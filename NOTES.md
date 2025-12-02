# Why Normalising

Normalizing data in LSTM models is crucial for several specific reasons related to the training process and model performance: 
## 1. Improved Gradient Flow and Stability: 

* Reason: LSTMs, like other neural networks, rely on gradient descent for learning. When input features have vastly different scales, the gradients associated with larger-scaled features can dominate, leading to unstable updates and slow convergence. Normalization ensures all features contribute more equally to the gradient calculation. 
* Example: Imagine predicting stock prices using historical data. If one feature is "daily volume" (e.g., millions of shares) and another is "daily price change" (e.g., a few dollars), the gradients from volume will be much larger, potentially causing the model to prioritize learning about volume fluctuations over price changes, even if price changes are more informative for the prediction. Normalizing both features to a similar scale (e.g., between 0 and 1) prevents this imbalance. 

## 2. Faster Convergence: 

* Reason: With normalized data, the loss landscape becomes more symmetrical and easier for the optimization algorithm to navigate. This allows the model to find the optimal weights and biases more efficiently, leading to faster training times. 
* Example: If you're training an LSTM to predict weather patterns from temperature, humidity, and wind speed, and these values are in their raw units (Celsius, percentage, m/s), the optimizer might struggle to find the best path. Normalizing them (e.g., using Min-Max scaling) creates a more well-behaved loss function, allowing for quicker convergence to a good solution. 

## 3. Preventing Activation Function Saturation: 

* Reason: LSTMs often use activation functions like tanh or sigmoid within their gates. These functions have a limited output range (e.g., -1 to 1 for tanh, 0 to 1 for sigmoid). If input values are very large or very small, the activations can saturate, meaning the gradients become very close to zero, effectively halting learning in those parts of the network. 
* Example: If an LSTM is processing sensor data with raw values like 1000 or -500, feeding these directly into a tanh function could push the output to its limits (-1 or 1), making it difficult for the network to learn subtle patterns. Normalizing the sensor data to a range like -1 to 1 or 0 to 1 prevents this saturation and allows for more effective learning. 

## 4. Regularization and Reduced Overfitting: 

* Reason: Normalization can act as a form of regularization. By bringing all features to a similar scale, it implicitly encourages the model to learn more robust and generalizable patterns, reducing the risk of overfitting to specific feature scales in the training data. 
* Example: In natural language processing, if you have word embeddings as input to an LSTM, and some word embeddings have much larger magnitudes than others, the model might over-rely on those high-magnitude embeddings. Normalizing the embeddings can help the model learn more balanced representations and generalize better to unseen text. 

## In summary
Normalizing data in LSTMs is a crucial preprocessing step that enhances training stability, accelerates convergence, prevents activation function saturation, and can even contribute to better generalization and reduced overfitting. 

# MIN MAX Scaling

The  in Python's scikit-learn library is a data preprocessing tool that transforms features by scaling each one individually to a specified range of -1 to 1. This technique is commonly known as Min-Max scaling or normalization. [1, 2, 3]  


What it Does ? MinMaxScaler(feature_range=(-1, 1))
## The primary function is to linearly transform the original data such that: 

* The minimum value of a feature in the original dataset becomes -1 in the transformed data. 
* The maximum value of that same feature becomes 1 in the transformed data. 
* All other values are proportionally scaled to fall within this new  range. [5, 6, 7, 8, 9]  

## The formula used for this transformation is: 
$ X_{scaled} = \frac{X - X_{\min}}{X_{\max} - X_{\min}} * (\text{max} - \text{min}) + \text{min}
$ 
where  and  are the values specified in  (i.e., -1 and 1). 

## Why Use It? 

* Equal Contribution: It ensures that all features, regardless of their original magnitudes or units, contribute equally to the machine learning model, preventing features with large numeric ranges from dominating the learning process. 
* Algorithm Performance: Many machine learning algorithms, particularly those using gradient descent (like neural networks) or distance calculations (like K-nearest neighbors and Support Vector Machines), perform better or converge faster when features are scaled to a specific, bounded range. The  range is a common alternative to the default  range for neural network inputs. 
* Preserves Distribution Shape: Unlike standardization (which changes the distribution to have a mean of 0 and standard deviation of 1), Min-Max scaling preserves the original shape of the data's distribution. [6, 7, 11, 12, 13]  

## Key Consideration 
The  is sensitive to outliers. Extreme values in the original data will determine the maximum and minimum of the scaled range, which can compress the bulk of the "normal" data into a very narrow band within the  range. [11, 14, 15]  



[MINMAX scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) can be found here


# Standardisation versus Normalisation when to use which

Standardization transforms data to have a mean of 0 and a standard deviation of 1, making it ideal for algorithms that assume a normal distribution and are sensitive to outliers, such as linear regression and PCA. Normalization scales data to a fixed range (often 0 to 1), which is useful when the data distribution is unclear or when an algorithm is sensitive to feature magnitudes, such as K-Nearest Neighbors and neural networks, and is more sensitive to outliers. [1]  

| Feature [1, 2] | Standardization | Normalization  |
| --- | --- | --- |
| Scaling Method | Uses the mean and standard deviation ($Z = (x - \mu) / \sigma$) | Uses the minimum and maximum values ($X_{norm} = (X - X_{min}) / (X_{max} - X_{min})$)  |
| Output Range | No fixed range; typically between -3 and 3 | A fixed range, most commonly 0 to 1 or -1 to 1  |
| Outlier Sensitivity | Less sensitive to outliers | More sensitive to outliers  |
| Data Distribution | Assumes data is normally distributed or close to it | Does not assume a specific distribution  |
| Use Cases | Linear regression, Support Vector Machines (SVM), and PCA | K-Nearest Neighbors (KNN) and neural networks  |

When to use which 

• Use Standardization when: 

	• Your algorithm assumes a normal distribution, or when you need the data centered at 0. 
	• Your data contains outliers that you don't want to unduly influence your model. 

• Use Normalization when: 

	• Your algorithm is sensitive to the magnitude of your features and you need all features on a similar scale. 
	• Your data does not follow a normal distribution. 
	• You need your features to be within a specific, bounded range. [1, 3, 4]  




