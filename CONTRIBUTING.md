# Cómo contribuir a este repositorio

Guía paso a paso para trabajar en este proyecto. Si sigues esto tal cual, no puedes romper nada de lo que ya está en `main`.

> **Nota:** esta guía aplica a todo el equipo. El dueño del repositorio ([@mar7inlancheros-maker](https://github.com/mar7inlancheros-maker)) sube sus propios cambios directo a `main`, sin rama ni pull request — ver la sección **"Flujo del dueño del repo"** al final.

## Antes de empezar (una sola vez)

### 1. Pide acceso

Necesitas ser colaborador del repo. Si aún no te agregaron, pide que te inviten con tu usuario o correo de GitHub, y acepta la invitación que te llega por correo/notificación.

### 2. Configura tu identidad en Git

Abre una terminal (en VS Code: `Terminal → New Terminal`) y corre, una sola vez en tu computadora:

```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu-correo@ejemplo.com"
```

Usa el mismo correo con el que tienes tu cuenta de GitHub.

### 3. Clona el repositorio

Elige una carpeta en tu computadora (fuera de OneDrive, para evitar conflictos de sincronización) y corre:

```bash
git clone https://github.com/mar7inlancheros-maker/Asset-Management-.git
cd Asset-Management-
```

Esto te descarga todo el proyecto. Ábrelo en VS Code (`File → Open Folder...`, selecciona la carpeta `Asset-Management-`).

### 4. Crea tu archivo de configuración local

Copia `.env.example` y renómbralo a `.env`. Rellena tus propios valores (por ejemplo tu correo real para el User-Agent que exige la SEC). Este archivo **nunca se sube a GitHub** — es solo tuyo, en tu computadora.

---

## La única regla de oro

**Si no eres el dueño del repositorio, nunca hagas `git push` estando parado en `main`.** Siempre trabaja desde tu propia rama. Este repositorio no tiene bloqueo automático que te lo impida (por ser privado en un plan gratuito), así que esta regla depende de que todos la sigamos por acuerdo.

---

## El flujo de trabajo, paso a paso

Repite este ciclo cada vez que vayas a trabajar en algo nuevo.

### Paso 1 — Trae lo último antes de empezar

```bash
git checkout main
git pull
```

Esto asegura que partes desde la versión más reciente del proyecto, no desde una copia vieja.

### Paso 2 — Crea tu propia rama

```bash
git checkout -b feature/nombre-descriptivo
```

Reemplaza `nombre-descriptivo` por algo claro y corto. Ejemplos reales para este proyecto:

```bash
git checkout -b feature/factor-momentum
git checkout -b fix/division-cero-fama-macbeth
git checkout -b experiment/nuevo-universo-sectores
```

Usa el prefijo que corresponda:
- `feature/` → algo nuevo
- `fix/` → corrección de un error
- `experiment/` → una prueba que puede no llegar a usarse

### Paso 3 — Trabaja normal

Edita tus archivos `.py` en VS Code, corre tus tests, prueba tu código. Nada distinto a como ya trabajas.

### Paso 4 — Revisa qué cambiaste

```bash
git status
```

Te muestra qué archivos modificaste. Si quieres ver el detalle línea por línea:

```bash
git diff
```

### Paso 5 — Guarda tus cambios (commit)

```bash
git add .
git commit -m "Descripción corta y clara del cambio"
```

Ejemplos de buenos mensajes:
- `"Agrega filtro de liquidez a quality_factor"`
- `"Corrige división por cero cuando el sector tiene menos de 3 empresas"`
- `"Agrega tests para el nuevo cálculo de score"`

Si trabajaste en varias cosas distintas, sepáralas en commits distintos (haz `add` y `commit` varias veces con archivos diferentes) en vez de mezclarlo todo en uno.

**Antes de este paso, revisa siempre** que `git status` no muestre tu archivo `.env` en la lista — si lo ves ahí, detente y avisa antes de continuar.

### Paso 6 — Sube tu rama a GitHub

La primera vez que subes esa rama específica:

```bash
git push -u origin feature/nombre-descriptivo
```

Las veces siguientes, si sigues trabajando en la misma rama, basta con:

```bash
git push
```

### Paso 7 — Abre el pull request

Después del `push`, la terminal normalmente te muestra un link como:

```
https://github.com/mar7inlancheros-maker/Asset-Management-/pull/new/feature/nombre-descriptivo
```

Cópialo y pégalo en tu navegador. Si no te salió el link, ve a la página del repositorio en GitHub — va a aparecer un aviso con un botón **"Compare & pull request"**.

Al abrirlo:
- Escribe un título claro.
- En la descripción, explica qué hiciste y por qué (2-3 líneas bastan).
- Click en **"Create pull request"**.

### Paso 8 — Avisa al equipo

Como no hay bloqueo automático, este paso reemplaza esa función: avisa por el chat del equipo que dejaste un pull request listo para revisión, con el link.

### Paso 9 — Espera revisión

Alguien del equipo entra al PR en GitHub, revisa el código, y:
- Si todo está bien: aprueba y hace **"Merge pull request"**.
- Si algo necesita ajuste: comenta directamente sobre las líneas en cuestión.

Si te piden un cambio, corrígelo en tu misma rama local, y repite el Paso 5 y 6 (`add`, `commit`, `push`) — el pull request se actualiza solo, no necesitas abrir uno nuevo.

### Paso 10 — Limpieza

Una vez que tu PR se fusionó, GitHub te ofrece un botón **"Delete branch"** justo ahí — bórrala, ya cumplió su función. En tu computadora:

```bash
git checkout main
git pull
git branch -d feature/nombre-descriptivo
```

Y listo — vuelves al Paso 1 para lo siguiente que vayas a hacer.

---

## Si te sale un conflicto al fusionar

Git te va a marcar el archivo así:

```python
<<<<<<< HEAD
    return df["roic"] * 0.6 + df["margin"] * 0.4
=======
    return df["roic"] * 0.5 + df["margin"] * 0.3 + df["growth"] * 0.2
>>>>>>> feature/tu-rama
```

Decide qué versión debe quedar (o combina ambas a mano), borra las líneas `<<<<<<<`, `=======` y `>>>>>>>`, guarda el archivo, y continúa:

```bash
git add nombre_del_archivo.py
git commit -m "Resuelve conflicto en nombre_del_archivo.py"
git push
```

Si no estás seguro de cómo resolverlo, pide ayuda antes de adivinar — un conflicto mal resuelto puede borrar el trabajo de otra persona sin querer.

---

## Preguntas frecuentes

**¿Puedo trabajar directo en `main` para algo rápido?**
No. Aunque parezca más lento, crear la rama toma 10 segundos y evita que un error tuyo afecte a todo el equipo de inmediato.

**¿Qué pasa si `git pull` me dice que hay cambios sin guardar?**
Guarda o descarta tus cambios locales primero (`git add` + `git commit`, o `git stash` si quieres guardarlos temporalmente sin comitear), y vuelve a intentar.

**¿Subí algo que no debía (como el `.env`)?**
Avisa de inmediato al dueño del repo. Borrarlo y subir un nuevo commit no es suficiente — el archivo sigue visible en el historial anterior y hay que tratarlo aparte.

**¿Cómo veo el historial de cambios?**

```bash
git log --oneline
```

---

## Flujo del dueño del repo

El dueño del repositorio escribe la mayoría del código y sube sus cambios **directo a `main`, sin rama ni pull request**. Su flujo es:

```bash
git checkout main
git pull
# edita lo que sea...
git add .
git commit -m "descripción del cambio"
git push
```

El único paso que no se salta es el `git pull` al inicio — así evita que su push choque con algo que el equipo subió mientras tanto.

Esta excepción aplica solo al dueño. El resto del equipo sigue el flujo de rama + pull request descrito arriba.
