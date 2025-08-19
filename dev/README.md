# Workspace de l'environnement de dev

## Exportation des Open WebUI functions

Après avoir suivi les instructions de [`../README.md`](../README.md), vous pouvez par exemple exporter les [Open WebUI Functions](https://docs.openwebui.com/features/plugin/functions/) de l'instance de <https://albert-dev.beta.numerique.gouv.fr/> vers `./functions/` avec la commande suivante :

```sh
$ ./scripts/pull-functions.py
Export https://albert-dev.beta.numerique.gouv.fr/ function to "/home/stephane/git/github.com/stephane-klein/dinum-assistant-ia-gitops-sklein-playground/dev/functions"

- "..." exported
- ...
```

## Upload de Pipelines

Vous pouvez utiliser le scripts suivant pour uploader une pipeline function, par exemple [`./pipelines/hello_world.py`](./pipelines/hello_world.py) vers l'instance de <https://albert-dev.beta.numerique.gouv.fr/> :

```sh
$ ./scripts/upload-pipelines-function.py pipelines/hello_world.py
{"status":true,"detail":"Pipeline uploaded successfully to ./pipelines/hello_world.py"}
```

Il est aussi possible d'uploader la configuration des variables *Valves* :

```sh
$ ./scripts/upload-pipelines-function-valves.py pipelines/hello_world_valves.json
{"status":true,"detail":"Pipeline uploaded successfully to ./pipelines/hello_world.py"}
```

Attention, les fichiers `*valves.json` peuvent contenir des informations sensibles et ne doivent pas être committés en clair dans le repository. 
