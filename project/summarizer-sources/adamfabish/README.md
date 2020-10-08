Reduction

Reduction is a python script which automatically summarizes a text by extracting the sentences which are deemed to be most important.

Example usage:

from reduction import *
reduction = Reduction()
text = open('filename.txt').read()
reduction_ratio = 0.5
reduced_text = reduction.reduce(text, reduction_ratio)