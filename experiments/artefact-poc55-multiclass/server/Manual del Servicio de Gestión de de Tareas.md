# Servicio de gestión de tareas (CPU y GPU)

## 1. Características del servidor

| Componente | Especificación |
|---|---|
| CPU | 2 × Intel(R) Xeon(R) Gold 6240 @ 2.60GHz |
| GPU | 2 × NVIDIA Tesla V100S PCIe 32GB |
| RAM | 256 GB |

---

## 2. Arquitectura de la plataforma

La plataforma para la gestión de tareas se construyó utilizando como base los siguientes elementos:

### Clearlinux
Distribución de Linux de código abierto y lanzamiento continuo enfocada en rendimiento y seguridad. Orientada a profesionales de TI, DevOps y despliegues en nube/contenedores. Itera continuamente para optimizar rendimiento y entregar parches de seguridad varias veces por semana.

### Slurm
Sistema de programación de trabajos y gestión de clústeres de código abierto, tolerante a fallas y altamente escalable para entornos pequeños o grandes. Funciones clave:
- Acceso exclusivo o compartido a recursos (nodos).
- Marco para inicio/ejecución de trabajos.
- Gestión de la disputa por recursos mediante colas de trabajos pendientes.

### Environment Modules
Herramienta que permite administrar el entorno de *shell* configurando variables por usuario o grupo. El entorno se configura con archivos de módulo para aplicaciones como CUDA, Python, GCC, OpenMPI, etc. Los módulos se pueden cargar/descargar de forma dinámica y atómica.

**Figura 1. Arquitectura del servicio**  
![Arquitectura del servicio](./Screenshot%202025-10-29%20122031.png "Esquema lógico con 4 VMs y el servidor físico (Pakhus)")  

---

## 3. Acceso al servicio

El servicio se accede vía **SSH** desde la red interna ECCI o utilizando el **VPN ECCI**.

**Direcciones de acceso**
- **Login 01**: `10.1.70.22`  
- **Login 02**: `10.1.70.24`

Existen dos nodos para iniciar sesión (puede usarse cualquiera) y una máquina virtual controladora que gestiona las tareas enviadas por los usuarios. Dentro del servicio, el servidor HPC funge como nodo de trabajo utilizando almacenamiento externo.

---

## 4. Aplicaciones disponibles

### CUDA
Plataforma de NVIDIA para computación en paralelo (incluye compilador). Puede usarse con Python, Fortran, Java o C/C++.

Cargar **CUDA v11.4** con environment modules:
```bash
module add cuda11.4
```

### Miniconda
Instalador mínimo para **conda**, permite instalar distintas versiones de Python.

Cargar **Miniconda**:
```bash
module add miniconda3
```

Crear un entorno con la versión deseada de Python:
```bash
conda create --name <nombre_entorno> python=<version>
# Ejemplo:
conda create --name python3.9 python=3.9
```

Activar conda y el entorno:
```bash
conda init bash
conda activate <nombre_entorno>
# Ejemplo:
conda activate python3.9
```

### GCC
Compilador para C/C++ desde línea de comandos.

Cargar **GCC 11.2**:
```bash
module add gcc11.2
```

---

## 5. Slurm

### 5.1 Colas de trabajo

| Cola         | Máx. duración por trabajo | GPUs | Enfoque/Notas                      |
|---|---:|---:|---|
| `gpu_long`   | 24 horas                   | 1   | Trabajos largos en GPU             |
| `gpu-wide`   | 12 horas                   | 2   | **Cola por defecto**               |
| `gpu-students` | 1 hora                   | 1   | Uso estudiantil / pruebas cortas   |
| `cpu-only`   | 12 horas                   | 0   | Solo CPU                           |
| `gpu-debug`  | 30 minutos                 | 1   | Pruebas pequeñas / *debug*         |

### 5.2 Comandos esenciales

- **Información de particiones y nodos**
  ```bash
  sinfo
  ```

- **Ver configuración y estado vía `scontrol`**
  - Trabajos en ejecución/pendientes:
    ```bash
    scontrol show jobs
    ```
  - Configuración detallada de particiones (colas):
    ```bash
    scontrol show partitions
    ```

- **Enviar un trabajo (script) a la cola**
  ```bash
  sbatch <archivo.sh>
  ```

- **Cancelar un trabajo por ID**
  ```bash
  scancel <job_id>
  ```

> Documentación oficial de Slurm: https://slurm.schedmd.com

### 5.3 Estados de los trabajos

- **RUNNING**: el trabajo tiene asignación y se está ejecutando.  
- **PENDING**: está esperando asignación de recursos.  
- **FAILED**: terminó con código de salida distinto de cero u otra condición de falla.  
- **COMPLETED**: finalizó en todos los nodos con código de salida cero.

### 5.4 Ejemplos

**Archivo: `red-neuronal.sh`**
```bash
#!/bin/bash
#
#SBATCH --job-name=red-neuronal
#SBATCH --output=red-neuronal.out
#
#SBATCH --nodes=1
#SBATCH --mem=45gb
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1

module load miniconda3 cuda11.4
source ~/.bashrc
conda activate python3.9

srun python red_neuronal/project_cnn.py
```

**Archivo: `matrix-multi.sh`**
```bash
#!/bin/bash
#
#SBATCH --job-name=multimatrix
#SBATCH --output=multimatrix.out
#
#SBATCH --nodes=1
#SBATCH --mem=20gb
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1

module load gcc11.2 cuda11.4

nvcc -o multi-matrix.out multi-matrix.cu
./multi-matrix.out
```

> Notas de lectura rápida:
> - En ambos scripts se asignan **RAM**, **nodos**, **cola (partition)** y **cantidad de GPUs** con directivas `#SBATCH`.
> - Luego se realiza la **carga de módulos** requeridos (p. ej., `miniconda3`, `cuda11.4`, `gcc11.2`).
> - Finalmente, se ejecuta el **programa objetivo** (`srun python ...` o binarios compilados con `nvcc`).

> Más ejemplos: https://help.rc.ufl.edu/doc/Sample_SLURM_Scripts
