# Proyecto de Modelado Predictivo - Grupo 2


## Configuración del Entorno

Sigue estos pasos para configurar tu entorno virtual e instalar las dependencias necesarias.

### 1. Crear el Entorno Virtual (Virtual Environment)
Ejecuta el siguiente comando en tu terminal para crear un entorno virtual llamado `venv`:

```powershell
python -m venv venv
```

### 2. Activar el Entorno Virtual
Dependiendo de qué terminal estés usando, elige el comando adecuado:

- **En PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **En Windows Command Prompt (CMD):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```
- **En Linux o WSL (Ubuntu):**
  ```bash
  source venv/bin/activate
  ```

### 3. Instalar Dependencias
Con el entorno virtual activado, instala las librerías necesarias con:

```powershell
pip install -r requirements.txt
```

### Requisitos
El archivo `requirements.txt` incluye las librerias necesarias para el proyecto.