# Thinking about machine learning

*Cheng Soon Ong, 29 Nov 2017*

Machine learning's popularity means that a user is tempted to use it as a magic wand in the hope
of solving their problem. Unfortunately there are many constraints and assumptions that are
built-in to the design and implementation of machine learning methods. Here we attempt to
elucidate the due diligence that a researcher should exercise before using the machine
learning hammer.

## What is machine learning?

We will focus our discussion on supervised learning, in particular binary classification and
regression. Supervised machine learning is the problem of building a predictor from data.
Data is split into **examples** (often denoted by x) and **labels**
(often denoted by y).
The machine learning task is to construct a **predictor**
(often denoted f(x)) that takes an example as input and produces a label as output.
The word "training" is used to denote the problem of finding the best predictor from
a given dataset.

Other words for examples are: features, attributes, covariates, independent variable,
explanatory variable.  
Other words for labels are: annotations, outcomes, dependent variable.

Machine learning is most useful when there is:

- no mechanistic model of the phenomenon we are trying to predict
- relatively large amounts of data (many examples)
- well defined outcomes (labels are binary or real numbers)

## The machine learning view of the world

- data are vectors
- predictors are functions or probabilistic models
- learning is finding good parameters

#### Data are vectors

Not all data is in numerical format, and care must be taken in the conversion.
Data is assumed to be represented by a table of numbers, where each column is a feature,
and each row is an example. It is often convenient to add an extra column on the left
for the label. Because we assume that data are vectors, this table of numbers can be
thought of as an array (for computational convenience) or a matrix
(for mathematical convenience).

- Be careful about the encoding of labels, for example binary classifiers assume either
  {0,1} or {-1,+1}.
- How to represent sequences?
- How to encode categorical features? How about ordinal features, such as {small, medium, large}?
- How to take domain structure into account, for example spatial or time information?

The stereotype for a data matrix is that the columns all have zero mean and unit variance.
The researcher should have good reasons to deviate from this, and to realise its effect
on the machine learning model.

Since we represent data as vectors, knowledge of linear algebra and analytic
geometry is needed to understand basic data manipulation.

#### Predictors are functions or probabilistic models

Once we have data in an appropriate vectorial representation, we can get to the
business of constructing a predictor. There are two major threads of research
on how to represent a
predictor: a predictor as a function, and a predictor as a probabilistic model.

- Difference between interpolation and extrapolation
- If uncertainty is important, use appropriate approaches
- Understand the difference between the score, the probability, and the binary prediction

We often use predictors whose parameters are represented by finite dimensional
vectors, and use the tools of linear algebra and analytic geometry to
manipulate them. Naturally if we choose to use probabilistic models,
knowledge of probability theory and statistics is important.

#### Learning is finding good parameters

Given the choice of data and predictor representation, we use training data to find
parameters of the predictor that best explain the data.

- We are interested in generalisation error (not training error)
- Finding good parameters either involve minimising the objective function or maximising
  the likelihood.
- Most numerical approaches usually require calculation of the gradient
- Note distinction between objective function of the learning method, and the performance
  metric used for evaluating predictions

In addition to the previous mathematical topics, when training a machine learning
algorithm, we use matrix decompositions, vector calculus and continuous
optimisation.

## Data lifecycle

1. Can I load your data using `pandas` or `numpy`?
2. Confounders, missing values, scale, units, encoding
3. Define the problem you want to answer:
    - The business/scientific problem
    - The performance metric
    - The model for the predictor

4. Run `sklearn` or `statsmodels` (**this is the machine learning part**)  
  Do not train on the test set.
5. Convert predictions into human friendly form for decision making

\newpage

## Extra challenges of life sciences

#### Deep data
- High dimensional low sample size
- Finite domain (we want predictions genome wide)
- Difficult to satisfy i.i.d. assumption

#### Prior knowledge
- Hard and soft constraints
- Causal systems with feedback and delays
- Anatomy is 3D (and images are currently 2D)

#### Different problems and communities
- Genomics, proteomics, medical imaging, health records, ecology
- Communication challenges

#### Problems do not fit naturally into supervised learning
- biomarker discovery (feature selection)
- gene finding (structured prediction, dependent labels)
- gene expression (positive and proportional values)
- forecasting future outcomes (extrapolation)
- causal discovery (in contrast to association)
- experimental design (unobserved confounders)
- biological/technical replicates (dealing with noise in examples and labels)
