# jigsaw_unintendedbiasclassification_validation_model
"Jigsaw Unintended Bias in Toxicity Classification" Competitions in Kaggle.

Cross validation model.
`python3 cv_model.py -m0 -v`

Minimize test.
`python3 cv_model.py -m1 -v`

Set random seed.
`python3 cv_model.py -r 1234`

Training and saving model.
`python3 cv_model.py -d model.bz2 -o Adam`

Training use Nadam.
`python3 cv_model.py -d model.bz2 -o Nadam`

## Useful additions

* lemma_dict-simbols.json

List of Lemmas that can be converted to the alphabet. A dictionary for correcting the fluctuation of notations expressed by Cyrillic letters and mathematical symbols.

* tokenvectorizer.py

Tokenizer with the function and multi-process to pick up the vocabulary contained in the word vector dictionary. If the word vector dictionary contains upper / lower case variations, map to the most likely word without lowering everything.
