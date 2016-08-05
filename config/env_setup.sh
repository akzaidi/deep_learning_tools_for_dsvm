#!/bin/bash

SESSION_HOME=$1

# Temporary fix for the new DSVMs
rm /usr/bin/python
ln -s /usr/bin/python2.7 /usr/bin/python
rm /usr/bin/R
ln -s /usr/lib64/MRO-3.2.5/R-3.2.5/lib64/R/bin/R /usr/bin/R
rm /usr/bin/Rscript
ln -s /usr/lib64/MRO-3.2.5/R-3.2.5/lib64/R/bin/Rscript /usr/bin/Rscript
echo "export PATH=/usr/local/bin:/usr/bin:/usr/lib64/MRO-3.2.5/R-3.2.5/lib64/R/bin/:$PATH" >> $SESSION_HOME/.bashrc
echo "export LD_LIBRARY_PATH=/usr/lib64/python2.7/site-packages:/usr/local/lib:/usr/lib64/MRO-3.2.5/R-3.2.5/lib64:$LD_LIBRARY_PATH" >> $SESSION_HOME/.bashrc
