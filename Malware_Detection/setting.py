# dataset
database_path = "dataset/data.csv"
processed_data_path = ["dataset/processed_data/non_embedding_data",
                       "dataset/processed_data/embedding_data"]
embedding_variable = [
    'SizeOfCode', 'SizeOfInitializedData', 'SizeOfUninitializedData', 'AddressOfEntryPoint', 'BaseOfCode', 'BaseOfData', 'ImageBase',
    'SizeOfImage', 'SectionsMeanEntropy', 'SectionsMinEntropy', 'SectionsMaxEntropy', 'SectionsMeanRawsize', 'SectionsMinRawsize', 'SectionMaxRawsize',
    'SectionsMeanVirtualsize', 'SectionsMinVirtualsize', 'SectionMaxVirtualsize', 'ImportsNb', 'ImportsNbOrdinal',
    'ExportNb', 'ResourcesNb', 'ResourcesMeanEntropy', 'ResourcesMinEntropy', 'ResourcesMaxEntropy', 'ResourcesMeanSize', 'ResourcesMinSize',
    'ResourcesMaxSize',
]
non_embedding_variable = [
    'Machine', 'SizeOfOptionalHeader', 'Characteristics',
    'MajorLinkerVersion', 'MinorLinkerVersion',
    'SectionAlignment', 'FileAlignment', 'MajorOperatingSystemVersion',
    'MinorOperatingSystemVersion', 'MajorImageVersion', 'MinorImageVersion',
    'MajorSubsystemVersion', 'MinorSubsystemVersion',
    'SizeOfHeaders', 'Subsystem', 'DllCharacteristics',
    'SizeOfStackReserve', 'SizeOfStackCommit', 'SizeOfHeapReserve',
    'SizeOfHeapCommit', 'LoaderFlags', 'NumberOfRvaAndSizes', 'SectionsNb',
    'ImportsNbDLL', 'LoadConfigurationSize',
    'VersionInformationSize', 'legitimate'
]
num_embedding_variable = 27
num_non_embedding_variable = 27
dataset_size = 138047
balance = {0: 96724, 1: 41323}
anormal_indexes_path = "dataset/processed_data/anormal_indexes"
normal_indexes_path = "dataset/processed_data/normal_indexes"

# overall tarining
batch_size = 50
class_weight = {0: balance[1] / dataset_size, 1: balance[0] / dataset_size} # not used because BinaryFocalCrossentropy

# auto encoder training
ae_train_dataset_ratio = 0.8
train_dataset_size = int(dataset_size * ae_train_dataset_ratio)
test_dataset_size = dataset_size - train_dataset_size
embedding_dimension = 16
encoder_path = "saved_model/autoencoder/encoder"
decoder_path = "saved_model/autoencoder/decoder"
discriminator_path = "saved_model/autoencoder/discriminator"
ae_classifier_path = "saved_model/autoencoder/ae_classifier"
discriminator_weight = 0.5
ae_classifier_weight = 0.5
loss_mean_std_path = "dataset/processed_data/loss_mean_std"
anormal_deviate = 0.1

# dense training
dense_classifier_path = "saved_model/dense_classifier"
dense_train_dataset_ratio = 0.9
