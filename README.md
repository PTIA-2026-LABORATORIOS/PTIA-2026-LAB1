# PTIA-2026-LAB1

# ESCUELA COLOMBIANA DE INGENIERÍA
# PRINCIPIOS Y TECNOLOGÍAS IA 2026-1
## ARBOLES DE DECISIÓN
## LABORATORIO 1/4

**OBJETIVOS**

Desarrollar competencias básicas para:
1. Modelar y resolver problemas usando árboles de decisión
2. Implementar árboles de decisión
3. Apropiar un framework para árboles de decisión (*scikit-learn*)
4. Conocer una librería para procesamiento de datos simbólicos (*python pandas DataFrame*)

**ENTREGABLE**


*Reglas para el envío de los entregables*:

* **Forma de envío:**
  Este laboratorio se debe enviar únicamente a través de la plataforma Moodle en la actividad definida. Se tendrán dos entregas: inicial y final.

* **Formato de los archivos:**
  Incluyan en un archivo *.zip* los archivos correspondientes al laboratorio.

* **Nomenclatura para nombrar los archivos:**
  El archivo deberá ser renombrado, “DT-lab-” seguido por los usuarios institucionales de los autores ordenados alfabéticamente (por ejemplo, se debe adicionar pedroperez al nombre del archivo, si el correo electrónico de Pedro Pérez es pedro.perez@mail.escuelaing.edu.co)

# **PARTE I. IMPLEMENTACIÓN DE ÁRBOLES DE DECISIÓN**
Para este apartado se van a implementar un árbol de decisión, en este caso usando como medida la entropia.

*La idea de los árboles de decisión fue desarrollada paulatinamiente. El pionero más reconocido es Ross Quinlan, quien propuso en 1986 el algoritmo ID3 (Iterative Dichotomiser 3) en el artículo [Induction of decision trees](https://link.springer.com/article/10.1007/BF00116251). Este algoritmo marcó un hito en la construcción automática de árboles de decisión a partir de datos.*

## I.A. IMPLEMENTACIÓN DE UN ÁRBOL DE DECISIÓN

Implementar un árbol de decisión; calculando una salida (Yp) para unas entradas X.

**Propiedades:**

*   Tarea: **Clasificación binaria**
*   Características: **Categóricas**
*   Criterio de selección: **Ganancia de información**
*   Métrica para evaluación : **F1 SCORE**

<div>
<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2020/09/Precision-vs-Recall-in-Machine-Learning.webp" width="350"/>
</div>

> **Formulas**

*   Impureza : **Entropía: H**

*   Ganancia de información: **IG(D,A)**

*   Impureza de Arbol: **Impurity(V\*)**


## Paso 1. Criterios de selección de atributos
Las impurezas en árboles de decision miden cuán homogéneas o heterogéneas son las clases dentro de un conjunto de datos ***en un nodo del árbol***. La métrica de impureza determina cómo dividir los datos en cada nodo.
Los principales métodos de evaluación de impureza son: **entropía** e **indice gini**.

*Incluyan las formulas de los dos criterios de selección y comparelos considerando criterios como sensibilidad al desbalance de clases y eficiencia computacional*

---
**Entropia:**

![img.png](img.png)

---
**Gini**

![img_1.png](img_1.png)

Donde *Pi* es la probabilidad de que un ejemplo sea de la clase *i*.

---
**Comparación**

La Entropía y el Índice Gini son medidas utilizadas en árboles de decisión para evaluar la impureza de un conjunto de datos y determinar la mejor división en cada nodo.

La Entropía, basada en la teoría de la información, mide el nivel de desorden o incertidumbre en los datos. Su valor va de 0 (conjunto completamente puro) hasta 1 (máxima incertidumbre, en clasificación binaria). Utiliza logaritmos en su cálculo, lo que la hace ligeramente más costosa computacionalmente.

Por otro lado, el Índice Gini mide la probabilidad de clasificar incorrectamente un elemento si se asigna aleatoriamente según la distribución de clases. También toma valores entre 0 (nodo puro) y 0.5 en clasificación binaria (máxima impureza). Su cálculo es más simple porque no utiliza logaritmos, por lo que suele ser más rápido.

En la práctica, ambos criterios suelen producir resultados similares en los árboles de decisión. La principal diferencia radica en que la entropía tiende a penalizar más los conjuntos muy desbalanceados, mientras que Gini es más eficiente computacionalmente y es el criterio usado por defecto en muchos algoritmos como CART.

---

## Paso 2. Ganancia de una característica e impureza del árbol
Los otros dos conceptos de fundamentan los árboles de decisión son la **ganancia de información** y la **impureza *de un arbol***.

---
**Ganancia:**

Es una medida utilizada en los árboles de decisión para cuantificar cuánto disminuye la impureza del conjunto de datos cuando se realiza una división basada en una característica específica. Su objetivo es determinar cuál atributo resulta más adecuado para separar los datos en un nodo del árbol. Cuanto mayor sea la ganancia obtenida, más efectiva es la característica para diferenciar las clases y mayor será la pureza de los subconjuntos generados.

![img_2.png](img_2.png)



