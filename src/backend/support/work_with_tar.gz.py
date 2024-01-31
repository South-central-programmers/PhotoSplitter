import tarfile

def tar_compress(file_ways, archive_way):
    with tarfile.open(f'{archive_way}.tar.gz', 'w:gz') as tar:
        for file_way in file_ways:
            tar.add(file_way)

def tar_decompress(directory, archive_way):
    with tarfile.open(f'{archive_way}', 'r:gz') as tar:
        tar.extractall(directory)

#tar_compress(["1", "2"], "new_archive")
#tar_decompress(".", "new_archive.tar.gz")