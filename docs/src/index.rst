PyMoDAQ-BeamShaping
===================

Welcome to the documentation of PyMoDAQ-BeamShaping, a toolbox to control Spatial Light Modulators in order
to produce spatially shaped optical beams!

Computational Spatial beam shaping is widely used to generate tailored optical fields for applications ranging
from optical trapping and microscopy to laser processing and quantum photonics. In practice, the design of a modulation
pattern must account for both the optical configuration and the physical constraints of the modulator.

PyMoDAQ-BeamShaping supports intensity-only and complex-field shaping and implements many algorithms such as
common iterative Fourier-transform algorithms. The toolbox is designed to bridge
computation and experiment by providing both numerical routines to solve the inverse beam-shaping problem
(phase retrieval and hologram optimization) and a complete set of tools to control spatial light modulators
and run the corresponding experimental implementations. It provides a unified interface for the implementation of a
large variety of propagation operators, modulator constraints, target constraints, evaluation metrics as well as
hardware equipments. Built on the open-source, Python-based PyMoDAQ framework, the toolbox enables reproducible
workflows across different experimental set-ups and is fully featured in a modular yet complete user interface as
shown on :numref:`main_gui`



  .. _main_gui:


.. figure:: /_images/main_app.png
   :alt: BeamShaping App

   The main Interface of the PyMoDAQ BeamShaping Toolbox


.. toctree::
   :numbered:
   :maxdepth: 1
   :caption: Documentation

   quick_start
   user
   tutorials
   about

