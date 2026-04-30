No external packages are needed. This code will work with regular Python.

This code was tested with Python 3.9, 3.10,  3.11, and 3.12

--------------
*INSTALLATION*
--------------
Linux and MacOS come with a python interpreter installed. 
If you do not have Python on your computer, download python here[https://www.python.org/downloads/].

--------------
*RUNNING CODE*
--------------
You only need to run the 'simulation.py'-file. At the end of the file there is a 'if __name__="__main__" block.
This code-block is executed when you run the 'simulation.py'-file.
All files need to be in the same directory.

When python is installed simply do one of the following three statements in your terminal:

```
python3 /path/to/your/files/simulation.py
python /path/to/your/files/simulation.py
py /path/to/your/files/simulation.py
```
Getting your paths correct is something you should be able to do on your own. Google is your friend!

------------------------
*CONDUCTING EXPERIMENTS*
------------------------
Set the number of replications, number of weeks, path to the inputfile, and which rule to use to whatever you want in the if-block at the end of the "simulation.py"-file.
There are also some rules that you will need to code for yourself. This is done in the "setWeekSchedule"-method in the "simulation.py"-file.

You will also still need to write functionality that outputs the results to a file. Luckily, this is fairly easy in Python.
