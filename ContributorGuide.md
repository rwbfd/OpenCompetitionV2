# Contributor Guide

Thank you very much for contributing to the project. We appreciate any form of contribution, including 
coding and testing.

## Contributing to testing
If you want to contribute to the testing of this project, please follow the following rules. 

1. Please use only the latest docker environment provided in the README.md file.
2. If possible, please perform the test based on the data file provided in the test/data folder to aim to reproduce.
3. If bugs are found, please mention it in the issues of the GitHub repositories. Please include complete code examples and outputs.
4. If additional functionalities are desired, please also describe it carefully in the issues.
5. If willing, please use the python unit test to perform the test in the test folder. 

## Contributing to the code
1. Please follow the general GitHub contribution pipeline. 
2. Please use pep-8 as the code format. 
3. All the class and functionality should come with docstrings. 
4. For functions, please specify its functionalities, its inputs, and its outputs (if any).
5. For classes, please specify the fields and a brief summarization of each method.
6. For class methods used only for implementation details, please start it with an underscore (such as `_train_impl`).
Such methods *DO NOT NEED TO BE DOCUMENTED OR TESTED*.
7. Except for all the implementation detail methods, please provide at least one unit test for each method. The unit test should use
real data as examples should cover all the branches and should be runnable as in the standard environment.
8. Please only use English (and correct ones at that)  
9. Please use dataclasses as options. One example is 
```
@dataclass
class TrainingArguments:

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
```
HAPPY CODING.
