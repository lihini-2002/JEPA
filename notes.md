-There are 2500 samples in I, T and Y. Both I and T are 512 dimensional decimal numbers while Y is one dimensional integers.

## Observations on the label data Y

-The labels run from 1 to 30, and then there is a 34 with nothing at 31–33.
-Label 34 is the largest class with 150.
-The scatter plot shows that the rows are shuffled hence classes are not clustered in rows. A plain train/test split of rows will not leave out a whole class.

## Comparing the row lengths of I and T

-The rows are not normalised to length 1. Need to normalise them before cosine-style comparisons.

## Value distribution

-most values sit in a bell-curve around zero. I has a cluster around -6 and T has a cluster around +5. Since both these form distinct clusters, most possibly not random noise.

## Analysis of dimensions

-In both I and T, almost all the columns have a mean of around 0. Only one or two columns have a deviation.
-The dimensions need to be standardised.

## Analyzing duplicate T rows

-173 T rows are repeated with the most repeated row appearing 30 times.
-Out of the 173 rows, 171 have all the repetitions in the same Y label.
