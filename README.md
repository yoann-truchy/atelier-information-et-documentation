# Prédiction de de la dépression chez les étudiants
Le but est de, grâce a deux modèles, prédire si oui ou non un étudiant est en dépression ou de déterminer son niveau de stress potentiel, selon certaines de ses habitutes de vie.

# Prérequis
Python3 est requis.

Commencez par cloner le répertoire git:
```Bash
git clone "https://github.com/yoann-truchy/atelier-information-et-documentation.git"
cd atelier-information-et-documentation
```

Optionellement, vous pouvez créer un environnement virtuelle python:
``` sh
python -m venv .env
# Linux
source .env\bin\activate
# Windows
.\.env\Scripts\activate.ps1
```

Afin de pouvoir executer le modèle il est important de s'assurer que toutes les dépendances sont installées.
Vous pouvez les installer executant la commande suivante:
``` sh
pip install -r requirements.txt
```

# Explication des données
Affin de se familiariser avec les données du dataset, des datacards sont fournies. Pour les visualiser, executez la commande suivantes :
``` sh
python ./visualiez-datacards.py
```

# Explication du modèle
