# James's Portfolio
Hi! 👋 I'm James Bentley, a Data Science student at the University of California San Diego with strong interests in Data Science & Machine Learning.
Here are some projects I've completed so far!

## [Large-Scale Data Analysis of Amazon Datasets](https://github.com/jimmybentley/Large-Scale-Amazon-Analysis/tree/main)
### AWS EC2
- Developed a distributed computing infrastructure using Dask and AWS EC2, enabling parallel processing of
Amazon reviews and products datasets across cluster of 3 worker nodes and 1 scheduler node.
- Conducted data cleaning, exploratory analysis, and data integrity checks on massive datasets, leveraging the
distributed computing infrastructure to optimize processing time.
- Optimized data processing algorithms to handle 45 GB of data, achieving significant speedups in computation using
distributed computing techniques to improve driving operational efficiency and cost savings.

### Apache Spark
- Developed scalable data processing pipeline using Apache Spark on Kubernetes cluster with PySQL.
- Trained decision tree regression model using word embedding with Word2Vec to extract meaningful features from
text data and improve recommendation systems.

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

## [Amazon Video Games User Activity Analysis](https://github.com/jimmybentley/Amazon_Game_Reviews)
- Normalized and stored 350 MB of review data into a relational database model, enabling optimized querying and
analysis using indexes.
- Conducted exploratory data analysis on Amazon Video Games user activity data, identifying key business metrics
and trends using Python and SQL.
- Visualized business metric data using Seaborn, providing actionable insights and driving informed business
decisions based on user activity trends

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

## [Natural Probabilistic Language Models](https://github.com/jimmybentley/Language-Models/tree/main)
- Developed Uniform, Unigram, and N-Gram language models from scratch using probabilistic language modeling
techniques and regular expressions.
- Replicated key functions of Python’s NLTK library, utilizing tokenization techniques to preprocess the text for the
language models.
- Trained the model to generate text probabilistically based on a tokenized corpus, leveraging probabilistic models to
generate text data.
