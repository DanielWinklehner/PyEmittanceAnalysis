# A simple setup script to create an executable for Allison emittance
# scanner analysis at Best Cyclotron Systems, Inc..
#
# Run the build process by running the command 'python setup.py build'
#
# If everything works well you should find a subdirectory in the build
# subdirectory that contains the files needed to run the application

import cx_Freeze
import sys
import os
import zipfile

base = None
if sys.platform == "win32":

    base = "Win32GUI"

build_exe_options = {"packages": ["os", "numpy", "scipy", "matplotlib"],
                     "include_files": ["GUI_v1.glade", "PlotSettingsDialog.glade", "fishfinder.png"]}

executables = [cx_Freeze.Executable("BCS_EmittanceAnalysis.py",
               base=base,
               targetName="BCS_EmittanceAnalysis.exe")]

cx_Freeze.setup(name="BCS_EmittanceAnalysis",
                version="1.0",
                description="Very simple analysis script for Allison emittance"
                            "scanner at Best Cyclotron Systems, Inc.",
                executables=executables,
                options={"build_exe": build_exe_options})

FILE = zipfile.ZipFile("gtktheme.zip")
FILE.extractall("build\exe.win32-2.7")
FILE.close()