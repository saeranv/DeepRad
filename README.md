
To train the deeprad models first install the conda environment. Open a terminal in this directory and type this:

```
conda create --file deeprad_v2.yml
```

To train the source autoencoder run:
```
python run_traintest.py --run_hparam 0
```

To train the target autoencoder run:
```
python run_traintest2.py --run_hparam 0
```

 
