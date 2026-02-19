# SproutCEX

## What is SproutCEX?

This project is an appendix to the Master's thesis *Learning Deterministic
$\omega$-Automata from Smallest Counterexamples* by Jan Lohse.

It provides an exemplary implementation of the **SproutCEX** algorithm for learning
deterministic $\omega$-automata with *informative right congruence* from smallest
counterexamples. **SproutCEX** is built upon the passive learner **Sprout** presented by
Bohn and Löding in *Constructing Deterministic $\omega$-Automata from Examples by an
Extension of the RPNI Algorithm*.

This implementation is to been seen as a proof-of-concept and used for visualization and
testing purposes. It was not written with the intent of being highly performant or being
used as a library for much further extension. Instead, the focus was on writing simple
pythonic code and keeping the dependencies minimal.

## Usage

As supplementary material for the thesis we provide the jupyter notebook
`thesis_examples.ipynb`. Simply clone or download the repository, install the
requirements using `pip install -r requirements.txt`, and then launch the notebook
using `jupyter notebook thesis_examples.ipynb`. It includes a selection of examples
for using **SproutCEX**, both from the thesis and additional ones.

We also provide the notebook `random_test_runner.ipynb`. Running it will generate a
sample of random *weak deterministic Büchi automata*, try to learn them using
**SproutCEX**, and perform a basic statistical analysis of the results. The statistical
analysis in Section TODO has been performed with the default parameters specified in the
notebook, and thus can be verified easily.

## Statistical results for SproutCEX

Here we give some basic statistical results for SproutCEX.

TODO
