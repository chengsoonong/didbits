# didbits

An unsorted bunch of ideas on practical aspects of machine
learning. This is aimed at the scientist (biologist, astronomer) who
is starting to use machine learning tools.

If you have a good description for any of the concepts in [Ideas](#ideas)
please send me a pull request.
Use the issue tracker for suggested topics.

The script ipynbhelper.py comes from [nbtidy](https://github.com/chengsoonong/nbtidy)

## Concepts with more detailed descriptions
Mostly in python.


- **Double loop cross validation** ```CrossVal``` When a classifier has hyperparameters that need to be set, a separate validation set of data needs to be set aside. The double loop refers to the need for an inner loop to find good hyperparameters and an outer loop to compute test error.

- **Normalising features** ```Normalisation/normalise_vector.ipynb```
  Various vector transformations to make learning work well. *Could
  still have bugs with transpose of matrix*

- **Normalising the kernel matrix** ```Normalisation/normalise_kernel.ipynb``` In contrast to
  normalising features, we can normalise the kernel matrix. This is
  particularly useful when combining kernels, e.g. with MKL.

- **Reshaping data with pandas** ```DataWrangling/process_abs.ipynb```
  shows an example of converting data into a convenient tabular format.

- **Performance measures.** ```Accuracy/evaluation.py``` It is worth
  repeating things like balanced error, area under the Receiver Operating Characteristic curve, etc.
  Many of the concepts are best explained by considering the confusion matrix.

## Ideas

- **No peeking.** It is not obvious what generalisation error is, and how to
not do things like report variance explained. At one level, this is
education about cross validation and bootstrap. At another level, in long
complicated pipelines, test data may be used in training. The famous
example is Vapnik using some feature selection method to choose two genes
on the whole dataset, and showing the SVM works well with only two
features.

- **Data wide predictions.** Related to above. Very often in biology, there
are finite spaces (e.g. the genome). So they want to make predictions
genome wide. Of course the question is then how to train since you are
predicting everywhere. One option is to do leave one chromosome out
prediction. But then it is hard to explain that you actually have 23
different classifiers, and not just one.

- **Visualisation of data and results.** Things like PCA and clustering. See also [t-SNE](http://lvdmaaten.github.io/tsne/).

- **How to subsample data.** Perhaps for computational reasons, you may want to
sanity check your pipeline on a small sample. Taking the first k items
from your file is generally not a good idea. There are also other good
reasons, e.g. cross validation, quick debugging, visualisation. Good ways
to do single pass sampling are not obvious.

- **There is no statistical test for equality.** Hard to convince an
experimentalist that there is no p-value for showing that two things are
the same.

- **Predictive densities.** For the more technically savvy crowd, getting a
distribution is much better than getting point estimates.

- **Normalisation.** It is worth repeating that normalisation changes results for classifiers such as the SVM.
How to deal with it? Who knows

- **Feature selection.** Biologists are very keen on finding predictive models
that they can understand. So feature selection is an end goal.

- **Linear correlation vs rank correlation**

- **correlation does not equal causality**

- **Dangers of thresholding.** Very often for quality control, people throw
away data, discretise values, or set something to zero. This introduces
complex changes due to the cliff in the distributions. There are other
reasons for thresholding, e.g. prior knowledge that values are positive,
or removing outliers from the data.

- **Dependent random variables.** Graphical models, etc.

- **Computing entropy and mutual information** practical tricks for counting, accounting for zeros, and binning.

- **Open science**
