# James's Portfolio
Hi! 👋 I'm James Bentley, a Data Science student at the University of California San Diego with strong interests in Data Science & Machine Learning.
Here are some projects I've completed so far!

## [Shakepeare Sonnet Generation using LSTM](https://github.com/)
### Character Level and Word Level RNN
- Trained recurrent neural network with long short-term memory on Shakespearean sonnets for 15 epochs, using
cross-entropy loss for optimization.
- Implemented preprocessing techniques such as word tokenization, preserving relevant punctuation and syllable
counts.
- Improved model by utilizing word embedding and incorporating syllable count as a feature. Lowered perplexity
score and lowered loss in half with word embedding compared to character embedding.
- Generated sonnets resembling Shakespearean style and structure across different temperature settings.

## [Congress Member Stock Trading Behavior Analysis](https://github.com/JimmyBentley/Predicting-Buy-or-Sell-via-Stock-Trades-of-Congress-Members)
- Analyzed congress member stock trades by scraping public data from congress.gov using Beautiful Soup, cleaned
using Pandas.
- Created a binary classification model for predicting the buy or sell behavior of congress members based on features of their stock trades, utilizing Sklearn pipelines and K-Nearest Neighbors classification.
- Attained an F1-Score of .75.
- Generated visualizations using Matplotlib that highlight differences in stock trading behavior between political parties, including variations in trading volume and frequency, as well as trends in the types of stocks traded.


## [Predicting Ratings of Reviews using Temporal and Sentiment Analysis](https://github.com/JimmyBentley/Prediction-Ratings/blob/main/Ratings_Predictions.pdf)
- Developed a predictive model for rating reviews using Temporal and Sentiment Analysis on Google Local Reviews dataset, with a focus on spam detection
- Extracted time features and conducted sentiment analysis through bag-of-words and TFIDF representations.
- Evaluated various regression models, selecting ridge regression as the top performer with an MSE of 0.915.
- Demonstrated improvement in review quality by identifying discrepancies between text and ratings.
- Highlighted the importance of predicting ratings from review text for spam detection and enhancing user trust in reviews.

## [Quadratic Discriminant Analysis](https://github.com/JimmyBentley/Predictive-ML-with-QDA/blob/main/QDA.ipynb)
- Used probabilistic learning to make predictions on completely unlabeled dataset for class competition.
- Tested on validation set for high sampling distribution accuracy to ensure high accuracy on test set.
- Achieved top 10% of class for best accuracy.

## [Dimensionality Reduction Techniques](https://github.com/JimmyBentley/Dimensionality-Reduction-Techniques)
- Representing high dimensional data with dimensionality techniques, with Principal Component Analysis (PCA) for linear data and Laplacian Eigenmaps for non-linear data.
- PCA performed on faces dataset which takes a top eigenvector as a detector of eyeglasses.
- Laplacian Eigenmaps performed on k-nearest neighbors graph to find similarity between US universities.
