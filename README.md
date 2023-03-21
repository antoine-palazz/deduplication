# Deduplication

## Web Intelligence - Deduplication Challenge

- https://statistics-awards.eu/competitions/4

- https://github.com/antoine-palazz/deduplication

- For more information, you can contact [Antoine Palazzolo](mailto:antoine.palazzolo@insee.fr).

## Overview

This is a Kedro project, which was generated using `Kedro 0.18.6`.

Take a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.

## How to start with the code

In the file ```setup.sh```, change the path to your dataset "wi_dataset.csv" and possible past submissions.

To install the dependencies and import your data, run:

```
./setup.sh
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## Project dependencies

To generate or update the dependency requirements for your project:

```
kedro build-reqs
```

This will `pip-compile` the contents of `src/requirements.txt` into a new file `src/requirements.lock`. You can see the output of the resolution by opening `src/requirements.lock`.

After this, if you'd like to update your project requirements, please update `src/requirements.txt` and re-run `kedro build-reqs`.

[Further information about project dependencies](https://kedro.readthedocs.io/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://kedro.readthedocs.io/en/stable/tutorial/package_a_project.html)
