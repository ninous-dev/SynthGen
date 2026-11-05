import logging

from utils.logger import BLUE, RESET, configure_logging
logger = logging.getLogger(__name__)


class TransformPipeline:
    """A simple pipeline to apply a sequence of transforms to a subject."""
    def __init__(self, transforms_list, verbose_dev: bool = True):
        self.transforms = transforms_list
        configure_logging(dev_mode=verbose_dev, module_name=__name__)

    
    def __call__(self, subject):
        """Applies each transform in the pipeline sequentially to the subject."""
        for transform in self.transforms:
            logger.info("%s[TransformPipeline]%s Applying transform: %s", BLUE, RESET, transform.__class__.__name__)
            subject = transform(subject)
        return subject

    def __repr__(self):
        """Display the pipeline as a list of transform class names"""
        transforms_str = ", ".join([
            t.__class__.__name__ for t in self.transforms
        ])
        return f"TransformPipeline({transforms_str})"

    __str__ = __repr__
