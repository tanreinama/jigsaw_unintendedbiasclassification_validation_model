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

