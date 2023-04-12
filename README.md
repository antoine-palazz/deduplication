# Deduplication

## Web Intelligence - Deduplication Challenge

- Link of the challenge: https://statistics-awards.eu/competitions/4

- Link to the public repository: https://github.com/antoine-palazz/deduplication

- For more information, you can contact [Antoine Palazzolo](mailto:antoine.palazzolo@insee.fr), that represents team **Nins**.

## Overview

This is a Kedro project, which was generated using `Kedro 0.18.6`.

Take a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.

## How to start with the code

Two possibilities:
- Load manually the initial dataset wi_dataset.csv into the folder ```data/01_raw/``` (and the possible past approaches or submissions in ```data/09_past_approaches/```)
- In the file ```setup.sh```, change the s3 path to your dataset ```wi_dataset.csv``` and possible past approaches to the problem, and uncomment the import.

To install the dependencies (and possibly import your data), run:

```
./setup.sh
```

To visualize the diverse elements of the code in a more interactive way, run:

```
kedro viz
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

To run only the part that has been selected for the final submission, add ```--tags=final_models```. To better visualize the parts of the pipelines that have been filtered, you can apply the same filters on the visual representation generated by ```kedro viz```.

If you have several CPUs at your disposition and want to make the execution faster, you can run the following lines:
```
kedro run --tags=final_models_parallel_part --runner=ParallelRunner
kedro run --tags=final_models_sequential_part --runner=SequentialRunner
```

The final output will be stored in ```data/07_model_output/best_duplicates.csv```, and a description of the output will be available in ```data/08_reporting/best_duplicates_description.csv```.

To further analyze and possibly improve the output by using past approaches, you can then use the notebook stored in ```notebooks/use_past_approaches.ipynb```.

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
