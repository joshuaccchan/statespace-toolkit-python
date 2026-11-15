# How-To Guides

Each guide shows how to use the toolkit's core functions on one small example with generated data.
A guide is a page, `README.md`, and a script, `guide.py`, that prints the numbers and draws the
figures on the page. The guides come in series, and within a series each guide builds on the one
before. They are the Python versions of the
[guides of the MATLAB toolkit](https://github.com/joshuaccchan/statespace-toolkit/tree/main/guides).

## Getting Started with the Precision Sampler

| Guide | What it shows | Core functions |
|---|---|---|
| [How to Draw the States of an Unobserved Components Model](uc_states/) | Draw the whole path of the states at once with the precision sampler | `ssm.simulate_states`, `ssm.diffmat` |
