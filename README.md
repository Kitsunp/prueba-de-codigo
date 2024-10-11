
Este proyecto es un ejemplo de implementación avanzada utilizando PyTorch para resolver problemas complejos de lenguaje y matemáticas. El modelo desarrollado cuenta con capacidades como evaluación de coherencia, atención mejorada, y embebidos líquidos, lo que le permite adaptarse dinámicamente a diferentes contextos, mientras mantiene la eficiencia y evita un crecimiento exponencial en tamaño y complejidad en comparación con otros modelos de gran escala. A continuación, se proporciona una explicación más detallada para entender cada componente del modelo y su funcionamiento, incluyendo fórmulas matemáticas donde sea relevante.

## Instalación

Para instalar las dependencias necesarias, primero asegúrese de tener `pip` y Python 3.8 o superior instalados. Luego, ejecute el siguiente comando:

```bash
pip install -r requirements.txt
```

Esto instalará todas las bibliotecas necesarias para ejecutar el proyecto, incluidas PyTorch, Transformers y otras dependencias utilizadas para el procesamiento de lenguaje natural y matemáticas avanzadas.

## Uso

Para entrenar el modelo, ejecute el script `main.py` con los parámetros deseados. A continuación se muestra un ejemplo básico de cómo entrenar el modelo:

```bash
python main.py
```

### Parámetros de Entrenamiento

El archivo `main.py` acepta varios parámetros que se pueden ajustar para optimizar el rendimiento del modelo según sus necesidades. Algunos de los parámetros importantes incluyen:

- `--batch_size`: Define el tamaño del lote utilizado durante el entrenamiento. Un tamaño de lote más grande puede mejorar la estabilidad del entrenamiento, pero requiere más memoria.
- `--learning_rate`: La tasa de aprendizaje utilizada por el optimizador. Ajuste este valor para encontrar un buen equilibrio entre la velocidad de convergencia y la estabilidad.
- `--num_epochs`: El número de épocas para entrenar el modelo. Un mayor número de épocas puede resultar en un mejor rendimiento, pero también aumenta el riesgo de sobreajuste.

Ejemplo de uso con parámetros personalizados:

```bash
python main.py --batch_size 16 --learning_rate 0.001 --num_epochs 10
```

### Evaluación del Modelo

Una vez entrenado el modelo, puede evaluar su rendimiento utilizando el script `evaluate.py`. Este script calculará métricas como precisión, coherencia y cobertura de conceptos para evaluar la calidad de las respuestas generadas por el modelo.

```bash
python evaluate.py --model_path path/to/model
```

## Arquitectura del Modelo

El proyecto utiliza una arquitectura de Transformer mejorada con varias innovaciones para mejorar la capacidad del modelo de manejar tareas complejas de lenguaje y matemáticas sin incrementar excesivamente su tamaño. A continuación, se describen algunos de los componentes clave de forma más detallada:

- **LiquidFoundationModelOptimized**: Este es el modelo principal que combina un codificador bidireccional y un decodificador, utilizando capas de Transformer mejoradas y embebidos líquidos. Está diseñado para manejar problemas complejos y adaptarse dinámicamente a diferentes entradas. El uso de especialistas (capas MoE) permite que el modelo evite un crecimiento excesivo en número de parámetros, manteniendo un tamaño optimizado que varía según la tarea específica.

- **LiquidEmbedding**: Los embeddings líquidos son una innovación que permite adaptar la cantidad de información que se retiene sobre la entrada dependiendo de su complejidad. Matemáticamente, esto se logra mediante una combinación de convoluciones y análisis de frecuencia utilizando la Transformada Rápida de Fourier (FFT). La compresión dinámica se controla mediante un coeficiente, que se ajusta de acuerdo a la magnitud de las frecuencias presentes en la entrada. Esto se expresa como:
  
  \[
  C_{ratio} = base_{compression} \times (1 - complexity)
  \]
  
  Donde \(C_{ratio}\) es el ratio de compresión y \(complexity\) es una medida de la variabilidad en las frecuencias de la entrada. Un valor más alto de \(complexity\) indica que la entrada tiene características más ricas y complejas, lo que disminuye el nivel de compresión aplicado. Este enfoque permite que el tamaño del modelo no crezca innecesariamente, sino que se ajuste dinámicamente según la necesidad.

- **MoELayer (Mixture of Experts)**: Esta capa tiene múltiples expertos, cada uno especializado en diferentes aspectos del problema. El enrutamiento dinámico se logra utilizando una función de gating que selecciona los expertos más adecuados para cada muestra de entrada. La función de gating se expresa como:
  
  \[
  G(x) = \text{softmax}(W_g x)
  \]
  
  Donde \(W_g\) son los pesos de la red de gating y \(x\) es la entrada. Esta función calcula probabilidades para cada experto, y los expertos con las probabilidades más altas son seleccionados para procesar la entrada. Esto permite que el modelo se adapte y aproveche expertos específicos para cada tipo de problema, lo cual mejora la eficiencia y la capacidad de generalización sin incrementar el número de parámetros global del modelo. Además, el ajuste dinámico del número de expertos (dinámico K) asegura que solo se utilicen los recursos necesarios para cada tarea, optimizando la capacidad y evitando el desperdicio de recursos.

- **EnhancedLocalAttention**: La atención local mejorada permite al modelo enfocarse en un subconjunto de la secuencia de entrada a la vez, lo cual es computacionalmente más eficiente que la atención completa. Se utiliza una ventana de atención deslizante de tamaño \(W\), y si se permiten ventanas superpuestas, se logra una mejor captura de las dependencias de largo alcance. La fórmula para calcular los puntajes de atención dentro de una ventana es:
  
  \[
  A_{ij} = \frac{(Q_i K_j^T)}{\sqrt{d_k}}
  \]
  
  Donde \(Q_i\) y \(K_j\) son las matrices de consultas y claves respectivamente, y \(d_k\) es la dimensión de la cabeza de atención, utilizada para normalizar los puntajes. Este enfoque de atención local reduce significativamente la complejidad computacional comparado con la atención global, permitiendo al modelo escalar mejor sin aumentar desproporcionadamente el uso de memoria.

### Otros Componentes Importantes

- **BidirectionalEncoder**: Este componente es un codificador bidireccional que tiene la capacidad de ver toda la secuencia de entrada antes de generar una salida. Esto es especialmente útil para tareas en las que el contexto completo es necesario para una comprensión adecuada. Utiliza convoluciones dilatadas para extender el campo receptivo, lo cual se expresa mediante la siguiente fórmula para una convolución de dilatación con factor \(r\):
  
  \[
  y(t) = \sum_{k=0}^{K-1} x(t - r \cdot k) w(k)
  \]
  
  Donde \(y(t)\) es la salida, \(x(t)\) es la entrada, \(w(k)\) son los pesos del kernel, y \(r\) es el factor de dilatación que controla la separación entre elementos de la entrada considerados por el kernel. Este uso de convoluciones dilatadas aumenta la capacidad de capturar dependencias de largo alcance sin añadir una gran cantidad de parámetros adicionales.

- **ActivationMonitor**: Esta herramienta es esencial para el monitoreo del proceso de entrenamiento del modelo. Permite visualizar las activaciones y los gradientes de las diferentes capas para identificar problemas como el desvanecimiento o explosión de gradientes, fenómenos comunes en redes profundas que pueden impedir el aprendizaje efectivo. Además, proporciona una visión detallada de cómo se comportan las diferentes capas del modelo, lo que es crucial para mantener una arquitectura eficiente y ajustada.

- **OptimizedFocalLoss**: La función de pérdida focal optimizada se utiliza para poner más peso en ejemplos difíciles y reducir la influencia de ejemplos fáciles que ya se clasifican correctamente. Esto se expresa matemáticamente como:
  
  \[
  L_{focal}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
  \]
  
  Donde \(p_t\) es la probabilidad predicha del modelo para la clase verdadera, \(\alpha_t\) es un factor de ponderación que controla el impacto de las clases, y \(\gamma\) es un parámetro de ajuste que controla el enfoque en ejemplos difíciles. Esta función de pérdida mejora la capacidad del modelo para manejar clases desbalanceadas sin aumentar significativamente la complejidad del modelo.

## Ejemplos de Uso

A continuación se presentan algunos ejemplos de cómo se puede utilizar el modelo para resolver problemas complejos:

1. **Resolución de Problemas Matemáticos**: El modelo puede resolver ecuaciones matemáticas y problemas de álgebra explicando cada paso intermedio.
   ```
   Instrucción: Resuelve la ecuación 3x + 5 = 20.
   Respuesta: Primero restamos 5 de ambos lados: 3x = 15. Luego dividimos por 3: x = 5.
   ```
   En este ejemplo, el modelo utiliza la propiedad distributiva para simplificar la ecuación y luego aplica una división para encontrar el valor de \(x\). La capacidad de descomponer el problema en pasos más pequeños evita que el modelo tenga que memorizar soluciones, promoviendo en su lugar un razonamiento más general.

2. **Generación de Respuestas Explicativas**: El modelo puede generar respuestas detalladas a preguntas complejas, proporcionando explicaciones paso a paso.
   ```
   Instrucción: ¿Cuál es la capital de Francia y por qué es importante?
   Respuesta: La capital de Francia es París. Es importante porque es el centro cultural, económico y político del país, además de ser un destino turístico mundialmente famoso.
   ```
   El modelo toma en cuenta aspectos históricos, económicos y culturales para proporcionar una respuesta completa que abarca diferentes dimensiones de la importancia de París.

3. **Análisis de Sentimientos**: El modelo puede analizar textos para determinar su sentimiento (positivo, negativo o neutro). Por ejemplo:
   ```
   Instrucción: Analiza el sentimiento del siguiente texto: "Hoy ha sido un día maravilloso, lleno de sorpresas positivas".
   Respuesta: El sentimiento del texto es positivo.
   ```
   Utiliza técnicas de clasificación supervisada para asignar una etiqueta de sentimiento al texto analizado.
##precacion
el modelo actual necesita entrenarse con otros datos, por el momento necesita mejorar y aumentar los parametros del modelo, con la configuracion actual el modelo solo tiene 50 millones de parametros, y faltan mejoras por implementar, los datos de entrenamiento necesitan mejorarse y el preprocesamiento y posterior analisis
## Contribuir

Las contribuciones a este proyecto son bienvenidas. Si desea contribuir, siga estos pasos:

1. Haga un fork del repositorio.
2. Cree una nueva rama para su funcionalidad (`git checkout -b nueva-funcionalidad`).
3. Haga commit de sus cambios (`git commit -m 'Añadir nueva funcionalidad'`).
4. Haga push a la rama (`git push origin nueva-funcionalidad`).
5. Abra un Pull Request.

Asegúrese de que sus contribuciones sigan las guías de estilo del proyecto y que todos los tests pasen antes de enviar su PR.
