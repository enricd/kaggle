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

- [Data Augmentation](./04_da.py) -> 0.79
- [TTA](./05_tta.ipynb) -> 0.815

# Día 4

- [Validación Cruzada](./06_cv.py) -> 0.825

# Día 5

- Nuevos [modelos](./07_modelos.ipynb)
- [se_resnext50](./07_modelos.py) -> 0.81 (0.821 + tta)

Probar resnest, efficientnet
Con CV y TTA puede mejorar

# Día 6

- Transfer learning y [Learning Rate scheduling](./08._tl.py) ->
- Cómo encontrar el batch size y learning rate óptimos con Pytorch Lightning [aquí](./08_lr_find.ipynb)
