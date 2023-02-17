# Starting Kit

This starting kit is here to help you with preparing the submission.zip file.

It contains the following files:

- This `README.md` file

- A `duplicates.csv` file containing a sample of the duplicates.csv file that
  you will have to submit

- A `reproducibility_approach_description.docx` file, used to describe the
  approach you used to generate the duplicates.csv file

## Structure of the `submission.zip` file

The submission.zip file should only contain the following files:

```
submission.zip
├── duplicates.csv
├── code.zip
└── reproducibility_approach_description.docx (optional)
```

Where the `code.zip` file contains the code used to generate the `duplicates.csv`
file.

> NOTE: DO NOT CHANGE THE NAMES OF THE FILES OR THE STRUCTURE OF THE `submission.zip`
> FILE.

While the `duplicates.csv` and `code.zip` files are mandatory requirements for
the Accuracy and AccuracyPlus Awards, the `reproducibility_approach_description.docx`
file is only used for the Reproducibility Award.

## Structure of the `duplicates.csv` file

The `duplicates.csv` file should contain the following columns:

- `id1`: the id of the first document
- `id2`: the id of the second document
- `type`: the type of duplicate (see below)

> NOTE: The `duplicates.csv` file SHOULD NOT contain any header.

The `duplicates.csv` file should contain only the duplicates that you
have found. The `id1` and `id2` columns should not contain the same value,
and `id1` should always be smaller than `id2`.

In addition, each `id1`,`id2` pair should only appear once in the `duplicates.csv`
file. If multiple duplicates are found between the same two documents, then the
`type` column should contain the most specific type of duplicate.

### Types of duplicates

The `type` column can take the following values:

- `FULL`: Two job advertisements are considered as full duplicates if they are
  both exactly the same, i.e. they have the same job title and job description.
  They may have differing sources and retrieval dates;

- `SEMANTIC`: Two job advertisements are considered as semantic duplicates if they
  advertise the same job position and include the same content in terms of the job
  characteristics (e.g. the same occupation, the same education or qualification
  requirements, etc.), but are expressed differently in natural language or in
  different languages;

- `TEMPORAL`: Two job advertisements are considered as temporal duplicates if
  they are semantic duplicates with varying advertisement retrieval and vacancy
  expired dates;

- `PARTIAL`: Two job advertisements are considered as partial duplicates if they
  describe the same job position but do not necessarily contain the elements
  (e.g. one job advertisement contains elements that the other does not).

> If specific job advertisements are not duplicates of any of the above types
> (e.g. have a type `NON_DUPLICATE`), they should NOT BE INCLUDED in the
> `duplicates.csv` file.
