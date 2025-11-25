## Objetivo
Una empresa del sector de Ciencia de Datos y Big Data desea optimizar su proceso de 
reclutamiento. Cuenta con datos demográficos, educativos y laborales de candidatos que han 
tomado cursos de capacitación con ellos. El objetivo es predecir si un candidato buscará 
cambiar de trabajo o si se quedará trabajando en la empresa. 
Tu tarea consiste en construir y comparar modelos de clasificación utilizando tres enfoques 
distintos: SVM (Máquinas de Vectores de Soporte), Perceptrón (modelo lineal básico) y 
una Red Neuronal Multicapa (MLP) empleando Keras o TensorFlow. Deberás interpretar sus 
métricas para evaluar cuál modelo tiene mejor desempeño. 
● Usar el repositorio de Github Classroom: https://classroom.github.com/a/ET5E-kA0 
● Utiliza el dataset de Kaggle:   
○ https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists 
● Preprocesamiento: 
○ Realizar limpieza de datos 
○ Tratamiento de valores faltantes  
○ Codificación de variables categóricas. 
● Divide los datos en conjuntos de entrenamiento y prueba (70% - 30%) 
● Entrena un modelo con SVM utilizando scikit-learn. 
● Entrena un modelo con Perceptrón (sklearn.linear_model.Perceptron). 
● Entrena una Red Neuronal básica (al menos 1 capa oculta) usando Keras o 
TensorFlow. 
● Calcula las siguientes métricas en el conjunto de prueba: 
● Accuracy 
● Precision 
● Recall 
● F1-score 
● Muestra las métricas en una tabla comparativa 
○ Archivo csv en metrics/evaluation_report.csv 
# Ejercicio Comparación de Modelos de Clasificación

## Estructura

```
data/
    train.csv
metrics/
    evaluation_report.csv  # Debe ser generado por src/main.py
src/
    main.py             # Debe ser completado 
tests/
    *.py                # Tests automáticos
requirements.txt
```

## Calificacion Automática (70 pts)

| Criterio                                                      | Puntos |
|---------------------------------------------------------------|--------|
| main.py ejecuta sin errores y genera el archivo requerido     | 10     |
| El preprocesamiento trata los valores faltantes correctamente | 10     |
| Codificación de variables categóricas está presente           | 10     |
| Entrena y evalúa correctamente el modelo SVM                  | 10     |
| Entrena y evalúa correctamente el Perceptrón                  | 10     |
| Entrena y evalúa correctamente una red neuronal en Keras      | 10     |
| El archivo evaluation_report.csv contiene todas las métricas  | 10     |

## Ejecución

```bash
pip install -r requirements.txt
python src/main.py
pytest tests
```

