Challenge: https://www.kaggle.com/c/cassava-leaf-disease-classification/

Challenge anterior: https://www.kaggle.com/c/cassava-disease/
Solución ganadora challenge anterior: https://www.kaggle.com/c/cassava-disease/discussion/94114

# Día 1

- Intro competi
- Descarga datos
- [Exploración datos](./00_exploracion_datos.ipynb)
- [Baseline](./01_baseline.py) -> 0.722

# Día 2

- Descargamos datos extra
- [Integración](./03_extra_data.ipynb) datos extra
- [Baseline](./03_extra_data.py) con datos extra -> 0.762

# Día 3

- [Data Augmentation](./04_da.py)
- [TTA](./05_tta.ipynb) -> 0.815

# Día 4

- [Validación Cruzada](./06_cv.py) -> 0.825

# Día 5 

en0 < renset18 < wide resnet50 (66M) < se_resnet50 (26M) < swav(28M+15M) < resnext50 (22M) < resnet50 < resnest (25M) < se_resnext (25M) < en5 (28M) < en3 (10M)

