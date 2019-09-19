# This is an empty directory where you will download the training data, using the [/script/setup](/script/setup) script.

After downloading the data, the directory structure will look like this:

```
├──data
|    │
|    ├──`{javascript, java, python, ruby, php, go}_licenses.pkl`
|    ├──`{javascript, java, python, ruby, php, go}_dedupe_definitions_v2.pkl`
|    │
|    ├── javascript
|    │   └── final
|    │       └── jsonl
|    │           ├── test
|    │           ├── train
|    │           └── valid
|    ├── java
|    │   └── final
|    │       └── jsonl
|    │           ├── test
|    │           ├── train
|    │           └── valid
|    ├── python
|    │   └── final
|    │       └── jsonl
|    │           ├── test
|    │           ├── train
|    │           └── valid
|    ├── ruby
|    │   └── final
|    │       └── jsonl
|    │           ├── test
|    │           ├── train
|    │           └── valid
|    ├── ruby
|    │   └── final
|    │       └── jsonl
|    │           ├── test
|    │           ├── train
|    │           └── valid
|    ├── php
|    │   └── final
|    │       └── jsonl
|    │           ├── test
|    │           ├── train
|    │           └── valid
|    └── go
|        └── final
|            └── jsonl
|                ├── test
|                ├── train
|                └── valid
| 
└── saved_models
```

## Directory structure

- `{javascript, java, python, ruby, php, go}\final\jsonl{test,train,valid}`:  these directories will contain multi-part [jsonl](http://jsonlines.org/) files with the data partitioned into train, valid, and test sets.  The baseline training code uses TensorFlow, which expects data to be stored in this format, and will concatenate and shuffle these files appropriately.
- `{javascript, java, python, ruby, php, go}_dedupe_definitions_v2.pkl` these files are python dictionaries that contain a superset of all functions even those that do not have comments.  This is used for model evaluation.
- `{javascript, java, python, ruby, php, go}_licenses.pkl` these files are python dictionaries that contain the licenses found in the source code used as the dataset for CodeSearchNet.  The key is the owner/name and the value is a tuple of ( path,  license content).  For example:
```
In [6]: data['pandas-dev/pandas']
Out[6]:
('pandas-dev/pandas/LICENSE',
 'BSD 3-Clause License\n\nCopyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development
 Team\nAll rights reserved.\n\nRedistribution and use in source and binary forms, with or without\nmodification, are
 permitted provided that the following conditions are met:\n\n* Redistributions of source code must retain the above
 copyright notice, this\n  list of conditions and the following disclaimer.\n\n* Redistributions in binary form must
 reproduce the above copyright notice,\n  this list of conditions and the following disclaimer in the documentation\n
 and/or other materials provided with the distribution....')
````
- `saved_models`: default destination where your models will be saved if you do not supply a destination

## Data Format

See [this](docs/DATA_FORMAT.md) for documentation and an example of how the data is stored.
