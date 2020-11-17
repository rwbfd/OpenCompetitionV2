# coding = 'utf-8'

from dataclasses import dataclass, field
import logging


@dataclass
class TabDataOpt:
    label: str = field(
        default='label',
        metadata={'help':
                      """
                      The name for the `y` variable.
                      The default name is 'label'.
                      """}
    )

    dis_vars_entity: list = field(
        default=None,
        metadata={'help':
                      """
                      A list of variables that are meant to be fed into a normal entity embedding layer (without using vicinity information). 
                      Default is None.
                      """}
    )

    dis_vars_vic: list = field(
        default=None,
        metadata={'help':
                      """
                      A list of variables that are meant to be fed into a vicino entity embedding layer (using vicinity information). 
                      Default is None. 
                      """}
    )

    conti_vars: list = field(
        default=None,
        metadata={'help':
                      """
                      A list of continuous variables to be fed directly to densely connected layers. 
                      Default is None.
                      """}
    )

    num_dim: int = field(
        default=10,
        metadata={'help':
                      """
                      The default number of dimensions for entity embeddings. 
                      We assume the output dimension number is the same so that it can be fed into neural networks.
                      Default is 10.
                      Cannot be smaller than 1.
                      """}
    )

    def __post_init__(self):
        if int(self.num_dim) <= 1:
            logging.error("""The number of dimension specified is %f. 
            The constructor only accepts integer value larger than 0.""" % self.num_dim)
        self.num_dim = int(self.num_dim)

        if self.dis_vars_entity is None and self.dis_vars_vic is None and self.conti_vars is None:
            logging.error("""All the variables specify are None.
            Must specify at least one of the dis_vars_entity, dis_vars_vic or conti_vars""")

        if (self.dis_vars_vic is None and self.centroids is not None) or (
                self.dis_vars_vic is not None and self.centroids is not None):
            logging.error("""
            If specify to use vicino embedding both is_vars_vic and centroids should be non-empty.""")

        if len(self.dis_vars_vic) != len(self.centroids):
            logging.error("""The number of variables for vicino entity embeddings  is %d, while the number of centoirds is %d. 
            These two must be equal.  
            """ % (len(self.dis_vars_vic), len(self.centroids)))
