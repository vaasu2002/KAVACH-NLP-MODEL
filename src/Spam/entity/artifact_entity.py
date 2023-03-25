from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

@dataclass(frozen=True)
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str