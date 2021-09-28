from generator import ArtefactGenerator, ArtefactBlockGenerator, GlasGenerator, DataGenerator

class UnknownGeneratorException(Exception):
    def __init__(self, message):
        super().__init__(message)

class GeneratorFactory:
    """Retrieve generator classes from string name"""
    generators = {
        'artefact': ArtefactGenerator,
        'artifact': ArtefactGenerator,
        'artefact_block': ArtefactBlockGenerator,
        'glas': GlasGenerator
    }

    @classmethod
    def get_generator(cls, generator_name: str) -> DataGenerator:
        """Get generator from string name"""
        if generator_name not in cls.generators:
            raise UnknownGeneratorException(f"Unknown generator: {generator_name}.")

        return cls.generators[generator_name]