# dato-native
a little repo for the "Truly Native?" kaggle competition

## data files
Raw html files can be downloaded from the kaggle competition website: (data files)[https://www.kaggle.com/c/dato-native/data].  Raw data files are going into the relative path: ```data/raw```. The zip files extract into separate folders (```{0, 1, 2, 3, 4}```). 

## processing the files
Processing the html files is easy!  Use ```process_html.py``` like so:

```python process_html.py data/raw/ data/processed/```

Takes a while, and will create the same separate folders in the ```data/processed/``` directory.

