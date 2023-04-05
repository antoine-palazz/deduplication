# Deduplication

## Web Intelligence - Deduplication Challenge

- Link of the challenge: https://statistics-awards.eu/competitions/4

- Link to the public repository: https://github.com/antoine-palazz/deduplication

- For more information, you can contact [Antoine Palazzolo](mailto:antoine.palazzolo@insee.fr).

## Overview

This is a Kedro project, which was generated using `Kedro 0.18.6`.

Take a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.

## How to start with the code

In the file ```setup.sh```, change the path to your dataset ```wi_dataset.csv``` and possible past approaches to the problem.

To install the dependencies and import your data, run:

```
./setup.sh
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

To run only the part that has been selected for the final submission, add ```--tags=final_models```.

If you have several CPUs at your disposition and want to make the execution faster, you can run the following lines:
```
kedro run --tags=final_models_parallel_part --runner=ParallelRunner
kedro run --tags=final_models_sequential_part --runner=SequentialRunner
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
