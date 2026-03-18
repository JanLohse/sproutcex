# SproutCEX

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://janlohse.github.io/sproutcex/)
[![CI](https://github.com/JanLohse/sproutcex/actions/workflows/main.yml/badge.svg)](https://github.com/JanLohse/sproutcex/actions/workflows/main.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/JanLohse/sproutcex?tab=MIT-1-ov-file#readme)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13%20|%203.14-blue)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19101725.svg)](https://doi.org/10.5281/zenodo.19101725)

## What is SproutCEX?

This project is an appendix to the Master's thesis *Learning Deterministic*
$\omega$*-Automata from Smallest Counterexamples* by Jan Lohse.

It provides an exemplary implementation of the **SproutCEX** algorithm for learning
deterministic $\omega$-automata with *informative right congruence* from smallest
counterexamples. **SproutCEX** is built upon the passive learner **Sprout** presented by
Bohn and Löding in *Constructing Deterministic* $\omega$*-Automata from Examples
by an Extension of the RPNI Algorithm*.

This implementation is to been seen as a proof-of-concept and used for visualization and
testing purposes. It was not written with the intent of being highly performant or being
used as a library for much further extension. Instead, the focus was on writing simple
pythonic code and keeping the dependencies minimal.

## Usage

An installation of [Python](https://www.python.org/) and a package manager like [pip](https://pip.pypa.io/en/stable/installation/) are required to run the code. To visualize the graphs installing [graphviz](https://graphviz.org/download/) is also recommended.

As supplementary material for the thesis we provide the jupyter notebook
`thesis_examples.ipynb`. Simply clone the repository:
```shell
git clone https://github.com/JanLohse/sproutcex
```
Navigate to the folder:
```shell
cd sproutcex
```
Install the package and its requirements:
```shell
pip install .
```
Then launch the notebook using:
```shell
jupyter lab thesis_examples.ipynb
```
The notebook includes a selection of examples for using **SproutCEX**, both from the
thesis and additional ones.

### Sample Testing

We also provide the notebook `random_test_runner.ipynb`. Running it will generate a
sample of random *weak deterministic Büchi automata*, try to learn them using
**SproutCEX**, and perform a basic statistical analysis of the results. The statistical
analysis in Section 4.3 has been performed with the default parameters specified in the
notebook, and thus can be verified easily. The computation can be quite taxing, because
of which we included precomputed results in the `data` folder. It can be run like
before:
```shell
jupyter lab thesis_examples.ipynb
```

### Just the Package
Alternatively one can also just install the package directly using:
```shell
pip install git+https://github.com/JanLohse/sproutcex
```
