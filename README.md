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
$ git config core.hooksPath git-hooks
$ gitleaks --version
gitleaks version 8.25.1
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
$ ./prod/scritps/check-api-access.py
Hello Stéphane Klein, your API Key secret token works successfully

$ ./dev/scritps/check-api-access.py
Hello Stéphane Klein, your API Key secret token works successfully
```

## Instances de prod et dev

Pour le moment il existe deux instances de *Assistant IA* :

- https://albert.numerique.gouv.fr/ géré dans le dossier [`./prod/`](./prod/)
- https://albert-dev.beta.numerique.gouv.fr/ géré dans le dossier [`./dev/`](./dev/)

Je vous invite à explorer la suite de ce playground en vous rendant dans le dossier de l'instance de votre choix.

## Contributions

Ce projet utilise [Gitleaks](https://notes.sklein.xyz/Gitleaks/) pour éviter de publier par erreur des secrets.

Pour plus d'informations, je vous invite à consulter cette note : https://notes.sklein.xyz/2025-05-07_2353/

Voici quelques commandes utiles.

Tester si votre dossier contient des secrets en dehors du fichiers `.secret` :

```
$ gitleaks dir -v

    ○
    │╲
    │ ○
    ○ ░
    ░    gitleaks

4:37PM INF scanned ~101712 bytes (101.71 KB) in 18.3ms
4:37PM INF no leaks found
```
