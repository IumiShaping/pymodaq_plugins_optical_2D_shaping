Installation
============

Prerequisites
-------------

You should follow the installation steps for PyMoDAQ (see
`PyMoDAQ documentation <https://pymodaq.cnrs.fr/en/latest/quick_start.html>`_.),
in particular the creation of a dedicated environment.


Install the Plugin
------------------

To install the BeamShaping extension, run (in the dedicated environment you just created):

.. code-block:: bash

   pip install pymodaq_plugins_beam_shaping

This command will automatically fetch and install PyMoDAQ and all the dependencies the plugin nee.
After installation, the plugin will automatically appear in PyMoDAQ’s Dashboard **Extensions** menu. A script
will also be created to directly launch the BeamShaping extension.