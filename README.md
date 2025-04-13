[![ORCID: Monks](https://img.shields.io/badge/Tom_Monks_ORCID-0000--0003--2631--4481-brightgreen)](https://orcid.org/0000-0003-2631-4481)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Generating Python Unit-Tests using Claude 3.7 Sonnet.

🎓 This tutorial aims provides an example generic system prompt for Claude 3.7 to support unit-testing in python.  By following the tutorial you will:

* ✅ Gain an understanding of system prompting
* ✅ Understand different types of unit tests that can be generated
* ✅ Learn how to limit the number of tests generated to allow for human-in-the-loop checking


## License

The materials have been made available under an [MIT license](LICENCE).  The materials are as-is with no liability for the author. Please provide credit if you reuse the code in your own work.

## Citation

If you reuse any of the code, or the tutorial helps you work, please provide a citation.

```bibtex
@software{TheOpenScienceNerd_Template_
author = {Monks, Thomas},
license = {MIT},
title = {{TheOpenScienceNerd - [INSERT TITLE]}},
url = {https://github.com/TheOpenScienceNerd/tosn_python_template}
}
```

## Installation instructions

### Installing dependencies

All dependencies can be found in [`binder/environment.yml`]() and are pulled from conda-forge.  To run the code locally, we recommend installing [miniforge](https://github.com/conda-forge/miniforge);

> miniforge is Free and Open Source Software (FOSS) alternative to Anaconda and miniconda that uses conda-forge as the default channel for packages. It installs both conda and mamba (a drop in replacement for conda) package managers.  We recommend mamba for faster resolving of dependencies and installation of packages. 

navigating your terminal (or cmd prompt) to the directory containing the repo and issuing the following command:

```bash
mamba env create -f binder/environment.yml
```

Activate the mamba environment using the following command:

```bash
mamba activate unit_test
```


## Repo overview

```
.
├── binder
│   └── environment.yml
├── prompts
│   └── 01_prompt.md
│   └── ...
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── metrics.py
├── test_metrics.py
└── README.md
```

* `binder/environment.yml` - contains the conda environment if you wish to run the unit-tests
* `prompts` - directory containing prompts including example system prompt
* `metrics.py` - example code to test
* `test_metrics.py` - example unit tests for `metrics.py`
* `CHANGES.md` - changelog with record of notable changes to project between versions.
* `CITATION.cff` - citation information for the code.
* `LICENSE` - details of the MIT permissive license of this work.
