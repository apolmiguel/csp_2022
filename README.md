Files are for the 2022 course on Crystal Structure Prediction at SISSA.

Short descriptions of the files are the following:
  1. csp_main.py - Generation of supercell and spin configurations, calculations of energy contributions from cluster expansions, and automation of energy calculations via DFT using QE scripts.
  2. config_energies_ce.sh - Extracts energies from cluster expansion calculations to a data file to be read in csp_regression.py.
  3. config_energies_dft.sh - Extracts energies from DFT calcualtions to a data file to be read in csp_regression.py.
  4. csp_regression.py - Training of data from cluster energy calculations to find a functional form of the cluster expansion energy. Validation with DFT energies calculated in QE.

Thank you to Edoardo Alessandroni for the assistance with the functions to interface bash scripting, QE, and python.
For any questions, send an e-mail to atan@sissa.it.
