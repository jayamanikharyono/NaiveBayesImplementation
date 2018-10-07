# NaiveBayesImplementation
## Description
This repo for Naive Bayes Implmentation in Python. there are three naive bayes classifier, first is for Handle Discrete Value, Second for Handle Continu Value and the Last for both.

## Instalation
--

## Training
To Train Model Execute Build Model Function
 * For Discrete or Continu Features (NaiveBayesCategoricalValue.py or NaiveBayesContinuValue.py)
 
   ```python
   model = build_model(X_train, Y_train)
   ```
    
 * For Both Discrete and Continu Value (NaiveBayes.py), where param numerical_column use as flag for Continu Features
 
   ```python
   model = build_model(X_train, Y_train, numerical_column = list_numerical_feature)
   ```
  


## Dataset
 * Balance Scale Dataset (Discrete) : https://archive.ics.uci.edu/ml/datasets/Balance+Scale
 * Haberman's Survival Dataset (Continu) : https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival
 * Abalone Dataset (Discrete and Continu) : https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
 
## References 
 * http://msi.binus.ac.id/files/2013/05/0301-02-Sri-Kusumadewi.pdf
 * http://download.portalgaruda.org/article.php?article=137307&val=4190
 
## TO DO
 * Convert to OOP
 * Model Saving
 
## License
 Copyright © 2018 Jaya Haryono Manik. See LICENSE for details.
