# **DUU Python Package**


## **1. Goal**

Design Under Uncertainty (DUU for short) offers algorithms for solving design problems under uncertainty, such as:
- parameter estimation (PE)
- design space characterization (DS)
- parameter set membership (PSM) <-- NOT IMPLEMENTED YET
- experiment design (ED) <-- NOT IMPLEMENTED YET



## **2. Building the package**
As a package development collaborator you want to allow others to use the 'duu' package. More specifically, you would like to send to the users an archive file which they can easily use to install the package on their machines. Below you find how to do that.

## **2.1 Building a source distribution
This option leads to the creation of a duu-<version>.tar.gz archive which you can send to any user and (s)he can install the 'duu' package simply by running in CMD 'pip install duu-<version>.tar.gz'.

How to do that?
Step 1: Open CMD (or equivalent) and go to 'src' directory ('cd .../src').
Step 2: Run 'python setup.py sdist'. This command means that you want Python to run setup.py file in order to create a source distribution archive. A 'dist' folder will be created within 'src' directory. Also an egg-info folder will be created.
Voila! In 'dist' directory you will find the file 'duu-<version>.tar.gz' that can be used by any user that wants to install the package.

What if you want to delete the 'dist' and egg-info folders automatically?
Step 1: Open CMD (or equivalent) and go to 'src' directory ('cd .../src').
Step 2: Run 'python setup.py clean'. This command means that you want Python to run setup.py file in order to clean the 'dist' and egg-info folders.
Observation: It is recommended to first 'clean' and then to use sdist option when you want to build a package source distribution.



## **3. Installing the package**

## **3.1. Installing from a source distribution**
Step 1: Open CMD (or equivalent) and go to the directory containing the 'duu-<version>.tar.gz' archive.
Step 2: Run 'pip install duu-<version>.tar.gz'. pip will install 'duu' package in the standard place, i.e. <Python-distribution-home>/Lib/site-packages/.
That's it! Now you can import 'duu' in any project you want to use it.

Observation: If you go to <Python-distribution-home>/Lib/site-packages/ then you will find the folder 'duu' which contains the files of the 'duu' package.


## **4. Uninstalling the package**
Step 1: Open CMD (or equivalent) anywhere.
Step 2: Run 'pip uninstall duu'. pip will uninstall 'duu' package, so the <Python-distribution-home>/Lib/site-packages/ will no longer contain the 'duu' folder.


