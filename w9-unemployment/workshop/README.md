DMP
=========

This example computes a version of the Diamond-Mortensen-Pissarides labor search and matching model.
This version replicates Petrosky-Nadeau and Zhang's QE example of Hagedorn and Manovskii.

There is an ocassionally binding non-negativity constraint on vacancies :math:`V_{t}`.

Expect that this constraint tends to bind in states with low employment and low productivity.

Here we illustrate a time-iteration solution method.

This finds the fixed point (policy function) satisfying the system of nonlinear FOCs.

We solve this example using a version of a finite-element method on local hierarchical sparse grids.

Files:

* ``dmp.py`` is the class file

* ``main.py`` is the main script for executing an example

* ``Business-cycle Search and Matching.ipynb`` Example Jupyter Notebook

Things to do:

* Alternative version with Markov chain shocks instead of AR(1)

* Speedups comparisons using alternatives:

	* NUMBA
	
	* OpenMPI

Dependencies:

* TASMANIAN. See:

	* [website](https://tasmanian.ornl.gov/) 
	* [website for Python interface](https://pypi.org/project/Tasmanian/)
	* Install using ``pip install Tasmanian --user``

* STATSMODEL. See [website](https://www.statsmodels.org/)

(c) 2020++ T. Kam (tcy.kam@gmail.com)
