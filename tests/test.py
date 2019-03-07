from pathlib import Path
import shutil
import unittest
import numpy as np

import pyn5


class TestU8(unittest.TestCase):
    def setUp(self):
        self.root = "test.n5"
        self.dataset = "test_u8"
        self.dataset_size = [10, 10, 10]
        self.block_size = [2, 2, 2]
        self.dtype = "UINT8"

        pyn5.create_dataset(
            self.root, self.dataset, self.dataset_size, self.block_size, self.dtype
        )
        self.n5 = pyn5.open(self.root, self.dataset, self.dtype, False)

        big = 2 ** 8
        self.working_block = [0, 1, 2, 3, big - 4, big - 3, big - 2, big - 1]
        self.out_of_bounds_block = [-1, -2, -3, -4, big, big + 1, big + 2, big + 3]
        self.wrong_dtype = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    def tearDown(self):
        if Path(self.root, self.dataset).is_dir():
            shutil.rmtree(Path(self.root, self.dataset))

    def test_read_write(self):
        # Dataset should start empty
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.dataset_size),
            [0] * np.prod(self.dataset_size),
        )
        # Test writing a valid block
        self.n5.write_block([0, 0, 0], self.working_block)
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.block_size), self.working_block
        )

        # Test writing overflow block
        try:
            self.n5.write_block([1, 1, 1], self.out_of_bounds_block)
            raise AssertionError("Expected OverflowError")
        except OverflowError:
            self.assertEqual(
                self.n5.read_ndarray([2, 2, 2], self.block_size),
                [0] * np.prod(self.block_size),
            )
        # Test writing invalid type block
        try:
            self.n5.write_block([2, 2, 2], self.wrong_dtype)
            raise AssertionError("Expected TypeError")
        except TypeError:
            self.assertEqual(
                self.n5.read_ndarray([4, 4, 4], self.block_size),
                [0] * np.prod(self.block_size),
            )

    def test_writting_wrong_dtype(self):
        bad_n5 = pyn5.open(self.root, self.dataset, "FLOAT64")
        try:
            bad_n5.write_block([0, 0, 0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            self.fail("Expected TypeError")
        except TypeError:
            pass


class TestU16(unittest.TestCase):
    def setUp(self):
        self.root = "test.n5"
        self.dataset = "test_u16"
        self.dataset_size = [10, 10, 10]
        self.block_size = [2, 2, 2]
        self.dtype = "UINT16"

        pyn5.create_dataset(
            self.root, self.dataset, self.dataset_size, self.block_size, self.dtype
        )
        self.n5 = pyn5.open(self.root, self.dataset, self.dtype, False)

        big = 2 ** 16
        self.working_block = [0, 1, 2, 3, big - 4, big - 3, big - 2, big - 1]
        self.out_of_bounds_block = [-1, -2, -3, -4, big, big + 1, big + 2, big + 3]
        self.wrong_dtype = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    def tearDown(self):
        if Path(self.root, self.dataset).is_dir():
            shutil.rmtree(Path(self.root, self.dataset))

    def test_read_write(self):
        # Dataset should start empty
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.dataset_size),
            [0] * np.prod(self.dataset_size),
        )
        # Test writing a valid block
        self.n5.write_block([0, 0, 0], self.working_block)
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.block_size), self.working_block
        )

        # Test writing overflow block
        try:
            self.n5.write_block([1, 1, 1], self.out_of_bounds_block)
            raise AssertionError("Expected OverflowError")
        except OverflowError:
            self.assertEqual(
                self.n5.read_ndarray([2, 2, 2], self.block_size),
                [0] * np.prod(self.block_size),
            )
        # Test writing invalid type block
        try:
            self.n5.write_block([2, 2, 2], self.wrong_dtype)
            raise AssertionError("Expected TypeError")
        except TypeError:
            self.assertEqual(
                self.n5.read_ndarray([4, 4, 4], self.block_size),
                [0] * np.prod(self.block_size),
            )


class TestU32(unittest.TestCase):
    def setUp(self):
        self.root = "test.n5"
        self.dtype = "UINT32"
        self.dataset = "test_{}".format(self.dtype)
        self.dataset_size = [10, 10, 10]
        self.block_size = [2, 2, 2]

        pyn5.create_dataset(
            self.root, self.dataset, self.dataset_size, self.block_size, self.dtype
        )
        self.n5 = pyn5.open(self.root, self.dataset, self.dtype, False)

        big = 2 ** 32
        self.working_block = [0, 1, 2, 3, big - 4, big - 3, big - 2, big - 1]
        self.out_of_bounds_block = [-1, -2, -3, -4, big, big + 1, big + 2, big + 3]
        self.wrong_dtype = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    def tearDown(self):
        if Path(self.root, self.dataset).is_dir():
            shutil.rmtree(Path(self.root, self.dataset))

    def test_read_write(self):
        # Dataset should start empty
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.dataset_size),
            [0] * np.prod(self.dataset_size),
        )
        # Test writing a valid block
        self.n5.write_block([0, 0, 0], self.working_block)
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.block_size), self.working_block
        )

        # Test writing overflow block
        try:
            self.n5.write_block([1, 1, 1], self.out_of_bounds_block)
            raise AssertionError("Expected OverflowError")
        except OverflowError:
            self.assertEqual(
                self.n5.read_ndarray([2, 2, 2], self.block_size),
                [0] * np.prod(self.block_size),
            )
        # Test writing invalid type block
        try:
            self.n5.write_block([2, 2, 2], self.wrong_dtype)
            raise AssertionError("Expected TypeError")
        except TypeError:
            self.assertEqual(
                self.n5.read_ndarray([4, 4, 4], self.block_size),
                [0] * np.prod(self.block_size),
            )


class TestU64(unittest.TestCase):
    def setUp(self):
        self.root = "test.n5"
        self.dtype = "UINT64"
        self.dataset = "test_{}".format(self.dtype)
        self.dataset_size = [10, 10, 10]
        self.block_size = [2, 2, 2]

        pyn5.create_dataset(
            self.root, self.dataset, self.dataset_size, self.block_size, self.dtype
        )
        self.n5 = pyn5.open(self.root, self.dataset, self.dtype, False)

        big = 2 ** 64
        self.working_block = [0, 1, 2, 3, big - 4, big - 3, big - 2, big - 1]
        self.out_of_bounds_block = [-1, -2, -3, -4, big, big + 1, big + 2, big + 3]
        self.wrong_dtype = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    def tearDown(self):
        if Path(self.root, self.dataset).is_dir():
            shutil.rmtree(Path(self.root, self.dataset))

    def test_read_write(self):
        # Dataset should start empty
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.dataset_size),
            [0] * np.prod(self.dataset_size),
        )
        # Test writing a valid block
        self.n5.write_block([0, 0, 0], self.working_block)
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.block_size), self.working_block
        )

        # Test writing overflow block
        try:
            self.n5.write_block([1, 1, 1], self.out_of_bounds_block)
            raise AssertionError("Expected OverflowError")
        except OverflowError:
            self.assertEqual(
                self.n5.read_ndarray([2, 2, 2], self.block_size),
                [0] * np.prod(self.block_size),
            )
        # Test writing invalid type block
        try:
            self.n5.write_block([2, 2, 2], self.wrong_dtype)
            raise AssertionError("Expected TypeError")
        except TypeError:
            self.assertEqual(
                self.n5.read_ndarray([4, 4, 4], self.block_size),
                [0] * np.prod(self.block_size),
            )


class TestI8(unittest.TestCase):
    def setUp(self):
        self.root = "test.n5"
        self.dtype = "INT8"
        self.dataset = "test_{}".format(self.dtype)
        self.dataset_size = [10, 10, 10]
        self.block_size = [2, 2, 2]

        pyn5.create_dataset(
            self.root, self.dataset, self.dataset_size, self.block_size, self.dtype
        )
        self.n5 = pyn5.open(self.root, self.dataset, self.dtype, False)

        big = 2 ** 7
        self.working_block = [
            1 - big,
            2 - big,
            3 - big,
            4 - big,
            big - 4,
            big - 3,
            big - 2,
            big - 1,
        ]
        self.out_of_bounds_block = [
            -big,
            -1 - big,
            -2 - big,
            -3 - big,
            big,
            big + 1,
            big + 2,
            big + 3,
        ]
        self.wrong_dtype = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    def tearDown(self):
        if Path(self.root, self.dataset).is_dir():
            shutil.rmtree(Path(self.root, self.dataset))

    def test_read_write(self):
        # Dataset should start empty
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.dataset_size),
            [0] * np.prod(self.dataset_size),
        )
        # Test writing a valid block
        self.n5.write_block([0, 0, 0], self.working_block)
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.block_size), self.working_block
        )

        # Test writing overflow block
        try:
            self.n5.write_block([1, 1, 1], self.out_of_bounds_block)
            raise AssertionError("Expected OverflowError")
        except OverflowError:
            self.assertEqual(
                self.n5.read_ndarray([2, 2, 2], self.block_size),
                [0] * np.prod(self.block_size),
            )
        # Test writing invalid type block
        try:
            self.n5.write_block([2, 2, 2], self.wrong_dtype)
            raise AssertionError("Expected TypeError")
        except TypeError:
            self.assertEqual(
                self.n5.read_ndarray([4, 4, 4], self.block_size),
                [0] * np.prod(self.block_size),
            )


class TestI16(unittest.TestCase):
    def setUp(self):
        self.root = "test.n5"
        self.dtype = "INT16"
        self.dataset = "test_{}".format(self.dtype)
        self.dataset_size = [10, 10, 10]
        self.block_size = [2, 2, 2]

        pyn5.create_dataset(
            self.root, self.dataset, self.dataset_size, self.block_size, self.dtype
        )
        self.n5 = pyn5.open(self.root, self.dataset, self.dtype, False)

        big = 2 ** 15
        self.working_block = [
            1 - big,
            2 - big,
            3 - big,
            4 - big,
            big - 4,
            big - 3,
            big - 2,
            big - 1,
        ]
        self.out_of_bounds_block = [
            -big,
            -1 - big,
            -2 - big,
            -3 - big,
            big,
            big + 1,
            big + 2,
            big + 3,
        ]
        self.wrong_dtype = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    def tearDown(self):
        if Path(self.root, self.dataset).is_dir():
            shutil.rmtree(Path(self.root, self.dataset))

    def test_read_write(self):
        # Dataset should start empty
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.dataset_size),
            [0] * np.prod(self.dataset_size),
        )
        # Test writing a valid block
        self.n5.write_block([0, 0, 0], self.working_block)
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.block_size), self.working_block
        )

        # Test writing overflow block
        try:
            self.n5.write_block([1, 1, 1], self.out_of_bounds_block)
            raise AssertionError("Expected OverflowError")
        except OverflowError:
            self.assertEqual(
                self.n5.read_ndarray([2, 2, 2], self.block_size),
                [0] * np.prod(self.block_size),
            )
        # Test writing invalid type block
        try:
            self.n5.write_block([2, 2, 2], self.wrong_dtype)
            raise AssertionError("Expected TypeError")
        except TypeError:
            self.assertEqual(
                self.n5.read_ndarray([4, 4, 4], self.block_size),
                [0] * np.prod(self.block_size),
            )


class TestI32(unittest.TestCase):
    def setUp(self):
        self.root = "test.n5"
        self.dtype = "INT32"
        self.dataset = "test_{}".format(self.dtype)
        self.dataset_size = [10, 10, 10]
        self.block_size = [2, 2, 2]

        pyn5.create_dataset(
            self.root, self.dataset, self.dataset_size, self.block_size, self.dtype
        )
        self.n5 = pyn5.open(self.root, self.dataset, self.dtype, False)

        big = 2 ** 31
        self.working_block = [
            1 - big,
            2 - big,
            3 - big,
            4 - big,
            big - 4,
            big - 3,
            big - 2,
            big - 1,
        ]
        self.out_of_bounds_block = [
            -big,
            -1 - big,
            -2 - big,
            -3 - big,
            big,
            big + 1,
            big + 2,
            big + 3,
        ]
        self.wrong_dtype = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    def tearDown(self):
        if Path(self.root, self.dataset).is_dir():
            shutil.rmtree(Path(self.root, self.dataset))

    def test_read_write(self):
        # Dataset should start empty
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.dataset_size),
            [0] * np.prod(self.dataset_size),
        )
        # Test writing a valid block
        self.n5.write_block([0, 0, 0], self.working_block)
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.block_size), self.working_block
        )

        # Test writing overflow block
        try:
            self.n5.write_block([1, 1, 1], self.out_of_bounds_block)
            raise AssertionError("Expected OverflowError")
        except OverflowError:
            self.assertEqual(
                self.n5.read_ndarray([2, 2, 2], self.block_size),
                [0] * np.prod(self.block_size),
            )
        # Test writing invalid type block
        try:
            self.n5.write_block([2, 2, 2], self.wrong_dtype)
            raise AssertionError("Expected TypeError")
        except TypeError:
            self.assertEqual(
                self.n5.read_ndarray([4, 4, 4], self.block_size),
                [0] * np.prod(self.block_size),
            )


class TestI64(unittest.TestCase):
    def setUp(self):
        self.root = "test.n5"
        self.dtype = "INT64"
        self.dataset = "test_{}".format(self.dtype)
        self.dataset_size = [10, 10, 10]
        self.block_size = [2, 2, 2]

        pyn5.create_dataset(
            self.root, self.dataset, self.dataset_size, self.block_size, self.dtype
        )
        self.n5 = pyn5.open(self.root, self.dataset, self.dtype, False)

        big = 2 ** 63
        self.working_block = [
            1 - big,
            2 - big,
            3 - big,
            4 - big,
            big - 4,
            big - 3,
            big - 2,
            big - 1,
        ]
        self.out_of_bounds_block = [
            -big,
            -1 - big,
            -2 - big,
            -3 - big,
            big,
            big + 1,
            big + 2,
            big + 3,
        ]
        self.wrong_dtype = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    def tearDown(self):
        if Path(self.root, self.dataset).is_dir():
            shutil.rmtree(Path(self.root, self.dataset))

    def test_read_write(self):
        # Dataset should start empty
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.dataset_size),
            [0] * np.prod(self.dataset_size),
        )
        # Test writing a valid block
        self.n5.write_block([0, 0, 0], self.working_block)
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.block_size), self.working_block
        )

        # Test writing overflow block
        try:
            self.n5.write_block([1, 1, 1], self.out_of_bounds_block)
            raise AssertionError("Expected OverflowError")
        except OverflowError:
            self.assertEqual(
                self.n5.read_ndarray([2, 2, 2], self.block_size),
                [0] * np.prod(self.block_size),
            )
        # Test writing invalid type block
        try:
            self.n5.write_block([2, 2, 2], self.wrong_dtype)
            raise AssertionError("Expected TypeError")
        except TypeError:
            self.assertEqual(
                self.n5.read_ndarray([4, 4, 4], self.block_size),
                [0] * np.prod(self.block_size),
            )


class TestF32(unittest.TestCase):
    def setUp(self):
        self.root = "test.n5"
        self.dtype = "FLOAT32"
        self.dataset = "test_{}".format(self.dtype)
        self.dataset_size = [10, 10, 10]
        self.block_size = [2, 2, 2]

        pyn5.create_dataset(
            self.root, self.dataset, self.dataset_size, self.block_size, self.dtype
        )
        self.n5 = pyn5.open(self.root, self.dataset, self.dtype, False)

        self.working_block = [
            -3.3999999521443642e38,
            -1.199999978106707e-38,
            1.199999978106707e-38,
            3.3999999521443642e38,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        big = 1.5e40
        self.out_of_bounds_block = [
            -big,
            -1 - big,
            -2 - big,
            -3 - big,
            big,
            big + 1,
            big + 2,
            big + 3,
        ]
        self.wrong_dtype = [1, 2, 3, 4, 5, 6, 7, 8]

    def tearDown(self):
        if Path(self.root, self.dataset).is_dir():
            shutil.rmtree(Path(self.root, self.dataset))

    def test_read_write_valid(self):
        # Dataset should start empty
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.dataset_size),
            [0] * np.prod(self.dataset_size),
        )
        # Test writing a valid block
        self.n5.write_block([0, 0, 0], list(self.working_block))
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.block_size), self.working_block
        )

    def test_read_write_expected_failures(self):
        self.fail("Not yet implemented")


class TestF64(unittest.TestCase):
    def setUp(self):
        self.root = "test.n5"
        self.dtype = "FLOAT64"
        self.dataset = "test_{}".format(self.dtype)
        self.dataset_size = [10, 10, 10]
        self.block_size = [2, 2, 2]

        pyn5.create_dataset(
            self.root, self.dataset, self.dataset_size, self.block_size, self.dtype
        )
        self.n5 = pyn5.open(self.root, self.dataset, self.dtype, False)

        self.working_block = [
            -2.3e-308,
            -1.7e308,
            1.7e308,
            2.3e-308,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        big = 1.5e600
        self.out_of_bounds_block = [
            -big,
            -1 - big,
            -2 - big,
            -3 - big,
            big,
            big + 1,
            big + 2,
            big + 3,
        ]
        self.wrong_dtype = [1, 2, 3, 4, 5, 6, 7, 8]

    def tearDown(self):
        if Path(self.root, self.dataset).is_dir():
            shutil.rmtree(Path(self.root, self.dataset))

    def test_read_write(self):
        # Dataset should start empty
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.dataset_size),
            [0] * np.prod(self.dataset_size),
        )
        # Test writing a valid block
        self.n5.write_block([0, 0, 0], self.working_block)
        self.assertEqual(
            self.n5.read_ndarray([0, 0, 0], self.block_size), self.working_block
        )

    def test_read_write_expected_failures(self):
        self.fail("Not yet implemented")
