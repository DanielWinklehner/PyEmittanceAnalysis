# PyEmittanceAnalysis
Old GTK+3 based GUI for analysis of emittance scans

## installation
This being based on GTK3 only runs on Linux (including WSL) right now. 
I tried to make it work on Windows, but python 3.4 is no longer maintained and 
that's the latest python 3 that one can get an 'all-in-one' installer for
GTK3 on Windows for.

### Ubuntu 18.04 WSL
Follow the Ubuntu instructions on the 
[pygobject webpage](https://pygobject.readthedocs.io/en/latest/getting_started.html)
for 'system provided PyGObject' (his has to be done with the OS python3 
installation, not in Anaconda3). 
in addition, install pip3 with
```buildoutcfg
sudo apt install python3-pip
```
and the following packages using ``pip install``:

* numpy
* scipy
* matplotlib

You should be good to go.

# Some notes about the program:
* Plate length and gap need to be set for each new Allison Emittance Scanner.
