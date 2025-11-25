import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')
# Librerías-----------------------------------------------------------------------------
# Cargar Datos


def load_and_preprocess_data():
    print("Cargando y preprocesando dataset...")

    df = pd.read_csv("data/train.csv")  # Dataset Original

    # Esta columna no es útil
    if 'enrollee_id' in df.columns:
        df.drop(columns=["enrollee_id"], inplace=True)

    print(f"Forma del dataset original: {df.shape}")
    print(f"Valores nulos por columna:\n{df.isnull().sum()}")

    categorical_columns = ['gender', 'enrolled_university', 'education_level',
                           'major_discipline', 'experience', 'company_size',
                           'company_type', 'last_new_job']

    # Tratamiento de atos, Gender, usaré la MODA
    if 'gender' in df.columns:
        mode_gender = df['gender'].mode(
        )[0] if not df['gender'].mode().empty else 'Male'
        df['gender'].fillna(mode_gender, inplace=True)

    # Categoría especifica
    if 'enrolled_university' in df.columns:
        df['enrolled_university'].fillna('no_enrollment', inplace=True)

    # Graduado es el más común.
    if 'education_level' in df.columns:
        mode_education = df['education_level'].mode(
        )[0] if not df['education_level'].mode().empty else 'Graduate'
        df['education_level'].fillna(mode_education, inplace=True)

    if 'major_discipline' in df.columns:
        df['major_discipline'].fillna('Other', inplace=True)

    #  cero  para experiencia faltante
    if 'experience' in df.columns:
        df['experience'].fillna('0', inplace=True)

    # desconocido
    if 'company_size' in df.columns:
        df['company_size'].fillna('Unknown', inplace=True)

    # MODA
    if 'company_type' in df.columns:
        mode_company_type = df['company_type'].mode(
        )[0] if not df['company_type'].mode().empty else 'Pvt Ltd'
        df['company_type'].fillna(mode_company_type, inplace=True)

    # Nunca
    if 'last_new_job' in df.columns:
        df['last_new_job'].fillna('never', inplace=True)

    # Las numéricas con media o mediana servirán. intentaré hacer una gráfica en uncuaderno de Jup
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in numeric_columns:
        numeric_columns.remove('target')

    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    print("Valores faltantes tratados correctamente.")

    # Limpiando datos

    # city_development_index ajustando decimales que estaban fuera de lugar
    if 'city_development_index' in df.columns:
        df['city_development_index'] = df['city_development_index'].round(3)
        df['city_development_index'] = df['city_development_index'].clip(0, 1)

    # Outliers
    if 'training_hours' in df.columns:
        Q1 = df['training_hours'].quantile(0.25)
        Q3 = df['training_hours'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df['training_hours'] = df['training_hours'].clip(
            lower_bound, upper_bound)

    # Categóricas

    print("Iniciando codificacion de variables categoricas...")

    # Ver todas las categóricas y aplicar label encoding, probar luego One hot si me da tiemop
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Codificada columna: {col}")

    print(
        f"Total de columnas categoricas codificadas: {len(categorical_cols)}")

    # Guardar dataset procesado
    os.makedirs("processed", exist_ok=True)
    df.to_csv("processed/train.csv", index=False)
    print("Dataset procesado guardado en: processed/train.csv")

    # Separar características y variable objetivo
    X = df.drop("target", axis=1)
    y = df["target"]

    print(f"Forma final del dataset: X{X.shape}, y{y.shape}")

    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Entrenando Modelos, SVM, Red neu M.C, Perceptrón


def train_models(X_train, y_train):
    print("\n" + "="*50)
    print("ENTRENANDO MODELOS")
    print("="*50)

    models = {}

    # SVM
    print("Entrenando SVM...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale',
              random_state=42, class_weight='balanced')
    svm.fit(X_train, y_train)
    models['svm'] = svm
    print("SVM entrenado correctamente")

    #  Perceptrón
    print("Entrenando Perceptron...")
    perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    perceptron.fit(X_train, y_train)
    models['perceptron'] = perceptron
    print("Perceptron entrenado correctamente")

    # 3. MLP Keras
    print("Entrenando Red Neuronal...")
    nn = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    nn.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Entrenar la red neuronal
    nn.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    models['nn'] = nn
    print("Red Neuronal entrenada correctamente")

    return models

# Evaluar modeloos


def evaluate_models(models, X_test, y_test):
    print("\n" + "="*50)
    print("EVALUANDO MODELOS")
    print("="*50)

    results = []

    for name, model in models.items():
        print(f"Evaluando {name}...")

        # Realizar predicciones
        if name == "nn":
            predictions = (model.predict(X_test) >
                           0.5).astype("int32").flatten()
        else:
            predictions = model.predict(X_test)

        # Calcular métricas
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        # Guardar resultados
        results.append({
            'model': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,        # Para tests individuales
            'f1_score': f1   # Para test principal
        })

        # Mostrar métricas
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print()

    return results


# Métricas guardadas en metrics
def save_metrics(results):
    print("Guardando metricas...")

    # Crear directorio metrics si no existe
    os.makedirs("metrics", exist_ok=True)

    # Crear DataFrame con los resultados
    df_metrics = pd.DataFrame(results)

    # Guardar en CSV
    df_metrics.to_csv("metrics/evaluation_report.csv", index=False)

    print("Metricas guardadas en: metrics/evaluation_report.csv")

    return df_metrics

# ==========Esta parte la saqué de un video de Youtube que vi, no recuerdo el link pq era de un hindú, únicamente fue esta parte de cuadros para comparar. ==================


def analyze_results(df_metrics):
    """
    Análisis de resultados para identificar el mejor modelo
    """
    print("\n" + "="*50)
    print("ANALISIS DE RESULTADOS")
    print("="*50)

    # Mostrar tabla comparativa
    print("Tabla Comparativa de Metricas:")
    print(df_metrics.to_string(index=False, float_format='%.4f'))

    # Identificar mejor modelo según F1-Score - usar 'f1_score' para consistencia
    best_model_idx = df_metrics['f1_score'].idxmax()
    best_model = df_metrics.iloc[best_model_idx]

    print(f"\nMEJOR MODELO SEGUN F1-SCORE:")
    print(f"   Modelo: {best_model['model']}")
    print(f"   F1-Score: {best_model['f1_score']:.4f}")
    print(f"   Accuracy: {best_model['accuracy']:.4f}")

    # Análisis adicional - usar 'f1_score' para consistencia
    print(f"\nANALISIS COMPARATIVO:")
    for idx, row in df_metrics.iterrows():
        model_name = row['model']
        f1_score_val = row['f1_score']
        accuracy = row['accuracy']

        if idx == best_model_idx:
            status = "MEJOR"
        elif f1_score_val >= 0.7:
            status = "BUENO"
        elif f1_score_val >= 0.5:
            status = "REGULAR"
        else:
            status = "BAJO"

        print(
            f"   {model_name:15} | F1: {f1_score_val:.4f} | Acc: {accuracy:.4f} | {status}")

    return best_model


# ======================================hasta Aquí===========================

def main():
    """
    Función principal que ejecuta todo el pipeline del laboratorio
    """
    print("="*60)
    print("LABORATORIO FINAL - INTELIGENCIA ARTIFICIAL")
    print("Prediccion de Cambio de Trabajo - HR Analytics")
    print("="*60)

    try:
        # 1. Cargar y preprocesar datos
        X_train, X_test, y_train, y_test = load_and_preprocess_data()

        # 2. Entrenar modelos
        models = train_models(X_train, y_train)

        # 3. Evaluar modelos
        results = evaluate_models(models, X_test, y_test)

        # 4. Guardar métricas
        df_metrics = save_metrics(results)

        # 5. Analizar resultados
        best_model = analyze_results(df_metrics)

        print("\n" + "="*60)
        print("Main Compilado sin Problemas")
        print("="*60)
        print("Archivos generados:")
        print("  processed/train.csv - Dataset procesado")
        print("  metrics/evaluation_report.csv - Metricas de evaluacion")
        print("="*60)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
