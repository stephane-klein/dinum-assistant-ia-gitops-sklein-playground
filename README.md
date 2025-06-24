# DINUM Assistant-IA GitOps Stéphane Klein playground

Ce [*playground*](https://notes.sklein.xyz/Playground/) me permet de manipuler as code ([GitOps](https://notes.sklein.xyz/GitOps/)) des éléments de configuration des instances <https://albert-dev.beta.numerique.gouv.fr/> et <https://albert.beta.numerique.gouv.fr/>.

## Conventions

La documentation (fichiers `*.md`) est rédigée en français, tandis que le code source des scripts (*bash* ou *python*), les commits messages et les noms de fichiers utilisent l'anglais.

## Configuration du workspace

Ce projet est compatible sour Linux et MacOS. Je ne l'ai personnellement testé uniquement sous Linux [Fedora](https://notes.sklein.xyz/Fedora/).

Prérequis:

- Installer Mise: https://mise.jdx.dev/installing-mise.html

```sh
$ mise install
$ pip install -r requirements.txt
```

Configurer `.secret` :

```sh
$ cp .secret.skel .secret
```

Modifiez les paramètres dans `.secret`.

```
$ source .envrc
```

Et pour finir, testez que vous avez bien accès à l'API:

```
$ ./scritps/check-api-access.py
Hello Stéphane Klein, your API Key secret token works successfully
```
