# Coursera-Machine Learning by Stanford University
Andrew Ng: Coursera-Machine Learning





## [Contents]
### Introduction
1.1 Introduction

### Supervised Learning
1.2 Linear Regression with One Variable

1.3 Linear Algebra Review

2.1 Environment Setup Instructions

2.2 Linear Regression with Multiple Variables

2.3 Octave/Matlab Tutorial

3.1 Logistic Regression

3.2 Regularization

4.1 Neural Networks: Representation

5.1 Neural Networks: Learning

6.1 Advice for Applying Machine Learning

6.2 Machine Learning System Design

7.1 Support Vector Machines

### Unsupervised Learning
8.1 Unsupervised Learning

8.2 Dimensionality Reduction

9.1 Anomaly Detection

9.2 Recommender Systems

10.1 Large Scale Machine Learning

11.1 Application Example: Photo OCR


## [Lectures]

## 113 videos

## WEEK 1

### 18 videos

## 1.1 Introduction


Welcome to Machine Learning! In this module, we introduce the core idea of teaching a computer to learn concepts using data—without being explicitly programmed. 

The Course Wiki is under construction. Please visit the resources tab for the most complete and up-to-date information.

### 5 videos, 9 readings

### Welcome
- 1 Welcome to Machine Learning!


### Introduction

- 2 Welcome
- 3 What is Machine Learning?
- 4 Supervised Learning
- 5 Unsupervised Learning

### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%201/Lecture1.pdf

## 1.2 Linear Regression with One Variable

Linear regression predicts a real-valued output based on an input value. 

We discuss the application of linear regression to housing price prediction, present the notion of a cost function, and introduce the gradient descent method for learning.

### 7 videos, 8 readings

### Model and Cost Function
- 6 Model Representation
- 7 Cost Function
- 8 Cost Function - Intuition I
- 9 Cost Function - Intuition II

### Parameter Learning
- 10 Gradient Descent
- 11 Gradient Descent Intuition
- 12 Gradient Descent For Linear Regression

### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%201/Lecture2.pdf





## 1.3 Linear Algebra Review

This optional module provides a refresher on linear algebra concepts. 

Basic understanding of linear algebra is necessary for the rest of the course, especially as we begin to cover models with multiple variables.

### 6 videos, 7 readings, 1 practice quiz


- 13 Matrices and Vectors
- 14 Addition and Scalar Multiplication
- 15 Matrix Vector Multiplication
- 16 Matrix Matrix Multiplication
- 17 Matrix Multiplication Properties
- 18 Inverse and Transpose

### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%201/Lecture3.pdf


## WEEK 2

### 14 videos

## 2.1 Environment Setup Instructions


## 2.2 Linear Regression with Multiple Variables


What if your input has more than one value? In this module, we show how linear regression can be extended to accommodate multiple input features. 

We also discuss best practices for implementing linear regression.

### 8 videos, 16 readings

### Multivariate Linear Regression
- 19 Multiple Features
- 20 Gradient Descent for Multiple Variables
- 21 Gradient Descent in Practice I - Feature Scaling
- 22 Gradient Descent in Practice II - Learning Rate
- 23 Features and Polynomial Regression


### Computing Parameters Analitically

- 24 Normal Equation
- 25 Normal Equation Noninvertibility



### Submitting Programming Assignments 
- 26 Working on and Submitting Programming Assignments



### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%202/Lecture4.pdf

## 2.3 Octave/Matlab Tutorial

This course includes programming assignments designed to help you understand how to implement the learning algorithms in practice. 

To complete the programming assignments, you will need to use Octave or MATLAB. 

This module introduces Octave/Matlab and shows you how to submit an assignment.


### 6 videos, 1 reading


- 27 Basic Operations
- 28 Moving Data Around
- 29 Computing on Data
- 30 Plotting Data
- 31 Control Statements: for, while, if statement
- 32 Vectorization


### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%202/Lecture5.pdf


## WEEK 3

### 11 videos

## 3.1 Logistic Regression
Logistic regression is a method for classifying data into discrete outcomes. 

For example, we might use logistic regression to classify an email as spam or not spam. 

In this module, we introduce the notion of classification, the cost function for logistic regression, and the application of logistic regression to multi-class classification.

### 7 videos, 8 readings

### Classification and Representation
- 33 Classification
- 34 Hypothesis Representation
- 35 Decision Boundary

### Logistic Regression Model
- 36 Cost Function
- 37 Simplified Cost Function and Gradient Descent
- 38 Advanced Optimization

### Multiclass Classification
- 39 Multiclass Classification: One-vs-all


### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%203/Lecture6.pdf


## 3.2 Regularization
Machine learning models need to generalize well to new examples that the model has not seen in practice. 

In this module, we introduce regularization, which helps prevent models from overfitting the training data.

### 4 videos, 5 readings


### Solving the Problem of Overfitting
- 40 The Problem of Overfitting
- 41 Cost Function
- 42 Regularized Linear Regression
- 43 Regularized Logistic Regression


### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%203/Lecture7.pdf



## WEEK 4


### 7 videos


## 4.1 Neural Networks: Representation
Neural networks is a model inspired by how the brain works. 

It is widely used today in many applications: when your phone interprets and understand your voice commands, it is likely that a neural network is helping to understand your speech; when you cash a check, the machines that automatically read the digits also use neural networks.

### 7 videos, 6 readings


### Motivations
- 44 Non-linear Hypotheses
- 45 Neurons and the Brain


### Neural Networks
- 46 Model Representation I
- 47 Model Representation II


### Applications
- 48 Examples and Intuitions I
- 49 Examples and Intuitions II
- 50 Multiclass Classification

### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%204/Lecture8.pdf



## WEEK 5

### 8 videos

## 5.1 Neural Networks: Learning
In this module, we introduce the backpropagation algorithm that is used to help learn parameters for a neural network. 

At the end of this module, you will be implementing your own neural network for digit recognition.



### 8 videos, 8 readings

### Cost Function and Backpropagation

- 51 Cost Function
- 52 Backpropagation Algorithm
- 53 Backpropagation Intuition

### Backpropagation in Practice


- 54 Implementation Note: Unrolling Parameters
- 55 Gradient Checking

- 56 Random Initialization
- 57 Putting It Together

### Application of Neural Networks
- 58 Autonomous Driving

### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%205/Lecture9.pdf



## WEEK 6

### 12 videos

## 6.1 Advice for Applying Machine Learning
Applying machine learning in practice is not always straightforward. 

In this module, we share best practices for applying machine learning in practice, and discuss the best ways to evaluate performance of the learned models.

### 7 videos, 7 readings

### Evaluating a Learning Algorithm
- 59 Deciding What to Try Next
- 60 Evaluating a Hypothesis
- 61 Model Selection and Train/Validation/Test Sets

### Bias vs. Variance
- 62 Diagnosing Bias vs. Variance
- 63 Regularization and Bias/Variance
- 64 Learning Curves
- 65 Deciding What to Do Next Revisited





### Slide:

https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%206/Lecture10.pdf

## 6.2 Machine Learning System Design
To optimize a machine learning algorithm, you’ll need to first understand where the biggest improvements can be made. 

In this module, we discuss how to understand the performance of a machine learning system with multiple parts, and also how to deal with skewed data.



### 5 videos, 3 readings


### Buildinf a Spam Classifier
- 66 Prioritizing What to Work On
- 67 Error Analysis



### Handling Skewed Data
- 68 Error Metrics for Skewed Classes
- 69 Trading Off Precision and Recall

### Using Large Data Sets
- 70 Data For Machine Learning


### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%206/Lecture11.pdf


## WEEK 7

### 6 videos

## 7.1 Support Vector Machines

Support vector machines, or SVMs, is a machine learning algorithm for classification. 

We introduce the idea and intuitions behind SVMs and discuss how to use it in practice.


### 6 videos, 1 reading


### Large Margin Classification
- 71 Optimization Objective
- 72 Large Margin Intuition
- 73 Mathematics Behind Large Margin Classification




### Kernels
- 74 Kernels I
- 75 Kernels II

### SVMs in Practice
- 76 Using An SVM


### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%207/Lecture12.pdf




## WEEK 8


### 12 videos

## 8.1 Unsupervised Learning

We use unsupervised learning to build models that help us understand our data better. 

We discuss the k-Means algorithm for clustering that enable us to learn groupings of unlabeled data points.

### 5 videos, 1 reading

### Clustering
- 77 Unsupervised Learning: Introduction
- 78 K-Means Algorithm
- 79 Optimization Objective
- 80 Random Initialization
- 81 Choosing the Number of Clusters

### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%208/Lecture13.pdf


## 8.2 Dimensionality Reduction
In this module, we introduce Principal Components Analysis, and show how it can be used for data compression to speed up learning algorithms as well as for visualizations of complex datasets.


### 7 videos, 1 reading

### Motivation
- 82 Motivation I: Data Compression
- 83 Motivation II: Visualization



### Principal Component Analysis

- 84 Principal Component Analysis Problem Formulation
- 85 Principal Component Analysis Algorithm



### Applying PCA
- 86 Reconstruction from Compressed Representation
- 87 Choosing the Number of Principal Components
- 88 Advice for Applying PCA






### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%208/Lecture14.pdf


## WEEK 9

### 14 videos

## 9.1 Anomaly Detection
Given a large number of data points, we may sometimes want to figure out which ones vary significantly from the average. 

For example, in manufacturing, we may want to detect defects or anomalies. 

We show how a dataset can be modeled using a Gaussian distribution, and how the model can be used for anomaly detection.

### 8 videos, 1 reading


### Density Estimation
- 89 Problem Motivation
- 90 Gaussian Distribution
- 91 Algorithm

### Building an Anomaly Detection System
- 92 Developing and Evaluating an Anomaly Detection System
- 93 Anomaly Detection vs. Supervised Learning
- 94 Choosing What Features to Use

### Multivariate Gaussian Distribution (Optional)
- 95 Multivariate Gaussian Distribution
- 96 Anomaly Detection using the Multivariate Gaussian Distribution

### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%209/Lecture15.pdf


## 9.2 Recommender Systems
When you buy a product online, most websites automatically recommend other products that you may like. 

Recommender systems look at patterns of activities between different users and different products to produce these recommendations. 

In this module, we introduce recommender algorithms such as the collaborative filtering algorithm and low-rank matrix factorization.

### 6 videos, 1 reading

### Predicting Movie Ratings
- 97 Problem Formulation
- 98 Content Based Recommendations

### Collaborative Filtering
- 99 Collaborative Filtering
- 100 Collaborative Filtering Algorithm


### Low Rank Matrix Factorization
- 101 Vectorization: Low Rank Matrix Factorization
- 102 Implementational Detail: Mean Normalization



### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%209/Lecture16.pdf



## WEEK 10

### 6 videos

## 10.1 Large Scale Machine Learning
Machine learning works best when there is an abundance of data to leverage for training. 

In this module, we discuss how to apply the machine learning algorithms with large datasets.


### 6 videos, 1 reading

### Gradient Descent with Large Datasets
- 103 Learning With Large Datasets
- 104 Stochastic Gradient Descent
- 105 Mini-Batch Gradient Descent
- 106 Stochastic Gradient Descent Convergence


### Advanced Topics

- 107 Online Learning
- 108 Map Reduce and Data Parallelism



### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%2010/Lecture17.pdf



## WEEK 11


### 5 videos

## 11.1 Application Example: Photo OCR
Identifying and recognizing objects, words, and digits in an image is a challenging task. 

We discuss how a pipeline can be built to tackle this problem and how to analyze and improve the performance of such a system.



### 5 videos, 1 reading

### Photo OCR
- 109 Problem Description and Pipeline
- 110 Sliding Windows
- 111 Getting Lots of Data and Artificial Data
- 112 Ceiling Analysis: What Part of the Pipeline to Work on Next

### Slide:
https://github.com/grapestone5321/Coursera-Machine-Learning-by-Stanford-University/blob/master/Slides/Week%2011/Lecture18.pdf


### Conclusion

- 113 Summary and Thank You



