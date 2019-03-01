import target.debug.libpyn5 as pyn5
from pathlib import Path
import shutil
from typing import NamedTuple, Optional


class Result(NamedTuple):
    passed: bool
    msg: Optional[str]

    def __str__(self):
        return "{}; {}".format("PASSED" if self.passed else "FAILED", self.msg)


def setup_dataset() -> pyn5.Dataset:
    dataset = Path("test.n5")
    if dataset.is_dir():
        shutil.rmtree(dataset)
    pyn5.create_dataset("test.n5", "test", [10, 10, 10], [2, 2, 2])
    return pyn5.Dataset("test.n5", "test", False)


def test_write_not_enough_data() -> Result:
    test = setup_dataset()
    try:
        test.write_block([0, 0, 0], [0, 1, 2, 3])
        return Result(False, "Allowed writing block with 4 values instead of 8!")
    except ValueError as e:
        return Result(True, str(e))


def test_write_too_much_data() -> Result:
    test = setup_dataset()
    try:
        test.write_block([0, 0, 0], [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0])
        return Result(False, "Allowed writing block with 16 values instead of 8!")
    except ValueError as e:
        return Result(True, str(e))


def test_write_to_negative_block_index() -> Result:
    test = setup_dataset()
    data = [0, 1, 2, 3, 4, 5, 6, 7]
    try:
        test.write_block([-1, -1, -1], data)
        try:
            test.read_ndarray([-2, -2, -2], [2, 2, 2])
            return Result(
                False, "Allowed writing and reading of blocks with negative indicies"
            )
        except Exception:
            return Result(
                False,
                "Allowed writing but not reading of blocks with negative indicies",
            )
    except Exception as e:
        return Result(True, str(e))


def test_write_above_block_bounds() -> Result:
    test = setup_dataset()
    try:
        data = [0, 1, 2, 3, 4, 5, 6, 7]
        test.write_block([11, 11, 11], data)
        try:
            test.read_ndarray([22, 22, 22], [2, 2, 2])
            return Result(
                False, "Allowed writing and reading of blocks outside boundaries"
            )
        except Exception:
            return Result(
                False, "Allowed writing but not reading of blocks outside boundaries"
            )
    except Exception as e:
        return Result(True, str(e))


def test_write_and_read() -> Result:
    test = setup_dataset()
    data = [0, 1, 2, 3, 4, 5, 6, 7]
    test.write_block([0, 0, 0], data)
    try:
        assert data == test.read_ndarray(
            [0, 0, 0], [2, 2, 2]
        ), "read data does not match expected output"
    except AssertionError as e:
        return Result(False, str(e))
    for i in range(8):
        # first coordinate first
        x = (i >> 0) % 2
        y = (i >> 1) % 2
        z = (i >> 2) % 2
        try:
            assert (
                data[i] == test.read_ndarray([x, y, z], [1, 1, 1])[0]
            ), "found {}, expected {} at ({},{},{})".format(
                test.read_ndarray([x, y, z], [1, 1, 1])[0], data[i], x, y, z
            )
        except AssertionError as e:
            return Result(False, str(e))
    return Result(True, None)


print(test_write_not_enough_data())
print(test_write_too_much_data())
print(test_write_to_negative_block_index())
print(test_write_above_block_bounds())
print(test_write_and_read())
