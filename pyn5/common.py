import enum


class StrEnum(str, enum.Enum):
    def __new__(cls, *args):
        for arg in args:
            if not isinstance(arg, (str, enum.auto)):
                raise TypeError(
                    "Values of StrEnums must be strings: {} is a {}".format(
                        repr(arg), type(arg)
                    )
                )
        return super().__new__(cls, *args)

    def __str__(self):
        return self.value

    # pylint: disable=no-self-argument
    # The first argument to this function is documented to be the name of the
    # enum member, not `self`:
    # https://docs.python.org/3.6/library/enum.html#using-automatic-values
    def _generate_next_value_(name, *_):
        return name


class CompressionType(StrEnum):
    RAW = "raw"
    BZIP2 = "bzip2"
    GZIP = "gzip"
    # LZ4 = "lz4"
    # XZ = "xz"


compression_args = {
    CompressionType.RAW: None,
    CompressionType.BZIP2: "blockSize",
    CompressionType.GZIP: "level",
    # CompressionType.LZ4: "blockSize",
    # CompressionType.XZ: "preset",
}
