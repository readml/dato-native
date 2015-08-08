# dato-native
a little repo for the "Truly Native?" kaggle competition

## data files
Raw html files can be downloaded from the kaggle competition website: (data files)[https://www.kaggle.com/c/dato-native/data].  Raw data files are going into the relative path: ```data/raw```. The zip files extract into separate folders (```{0, 1, 2, 3, 4}```). 

## processing the files
Processing the html files is not super easy. Use ```process_html.py``` like so:

```python process_html.py data/ processed_data/```

Takes a while, and will create a chunk of data for one of the five buckets (i.e. ```processed_data/chunk0.json```

## notes about GraphLab
The dataframe used in GraphLab is called the SFrame; the interesting part is it runs on disk, and boots up its own little database for each column in the SFrame.  While this is great for datasets that don't fit into memory (like our ~10g worth of processed scraped data), it's not so great if you are lacking on disk space, like my computer! So make sure you have sufficient disk space, as it will essentially duplicate the 10gb worth of json files into 10gb of SFrame database disk space.
