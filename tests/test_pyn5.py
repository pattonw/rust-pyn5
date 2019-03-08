#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyn5` package."""


import unittest
from pathlib import Path
import shutil
import numpy as np


import pyn5


class BaseTestCase:
    class BaseTest(unittest.TestCase):
        def setUp(self):
            self.root = "test.n5"
            self.dataset = "test_{}".format(self.dtype)
            self.dataset_size = [10, 10, 10]
            self.block_size = [2, 2, 2]
            if Path(self.root).is_dir():
                shutil.rmtree(str(Path(self.root).absolute()))
            pyn5.create_dataset(
                self.root, self.dataset, self.dataset_size, self.block_size, self.dtype
            )
            self.n5 = pyn5.open(self.root, self.dataset, self.dtype, False)

        def tearDown(self):
            if Path(self.root).is_dir():
                shutil.rmtree(str(Path(self.root).absolute()))

        def test_read_write_valid(self):
            self.n5.write_block([0, 0, 0], self.valid_block)
            self.assertEqual(
                self.n5.read_ndarray([0, 0, 0], self.block_size), self.valid_block
            )

        def test_read_write_overflow(self):
            """
            Doesn't work properly for float types since floats that are too
            large get converted to inf
            """
            try:
                self.n5.write_block([1, 1, 1], self.overflow_block)
                raise AssertionError("Expected OverflowError")
            except OverflowError:
                self.assertEqual(
                    self.n5.read_ndarray([2, 2, 2], self.block_size),
                    [0] * np.prod(self.block_size),
                )

        def test_read_write_wrong_dtype(self):
            """
            Doesn't work properly for float types since I'm guessing ints
            get converted to float types somewhere between python and rust
            """
            try:
                self.n5.write_block([2, 2, 2], self.wrong_dtype_block)
                raise AssertionError("Expected TypeError")
            except TypeError:
                self.assertEqual(
                    self.n5.read_ndarray([4, 4, 4], self.block_size),
                    [0] * np.prod(self.block_size),
                )


class TestU8(BaseTestCase.BaseTest):
    def setUp(self):
        self.dtype = "UINT8"

        big = 2 ** 8
        self.valid_block = [0, 1, 2, 3, big - 4, big - 3, big - 2, big - 1]
        self.overflow_block = [-1, -2, -3, -4, big, big + 1, big + 2, big + 3]
        self.wrong_dtype_block = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        super().setUp()

    @unittest.expectedFailure
    def test_writting_wrong_dtype(self):
        bad_n5 = pyn5.open(self.root, self.dataset, "FLOAT64")
        try:
            bad_n5.write_block([0, 0, 0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            self.fail("Expected TypeError")
        except TypeError:
            pass


class TestU16(BaseTestCase.BaseTest):
    def setUp(self):
        self.dtype = "UINT16"

        big = 2 ** 16
        self.valid_block = [0, 1, 2, 3, big - 4, big - 3, big - 2, big - 1]
        self.overflow_block = [-1, -2, -3, -4, big, big + 1, big + 2, big + 3]
        self.wrong_dtype_block = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        super().setUp()


class TestU32(BaseTestCase.BaseTest):
    def setUp(self):
        self.dtype = "UINT32"

        big = 2 ** 32
        self.valid_block = [0, 1, 2, 3, big - 4, big - 3, big - 2, big - 1]
        self.overflow_block = [-1, -2, -3, -4, big, big + 1, big + 2, big + 3]
        self.wrong_dtype_block = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        super().setUp()


class TestU64(BaseTestCase.BaseTest):
    def setUp(self):
        self.dtype = "UINT64"

        big = 2 ** 64
        self.valid_block = [0, 1, 2, 3, big - 4, big - 3, big - 2, big - 1]
        self.overflow_block = [-1, -2, -3, -4, big, big + 1, big + 2, big + 3]
        self.wrong_dtype_block = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        super().setUp()


class TestI8(BaseTestCase.BaseTest):
    def setUp(self):
        self.dtype = "INT8"

        big = 2 ** 7
        self.valid_block = [
            1 - big,
            2 - big,
            3 - big,
            4 - big,
            big - 4,
            big - 3,
            big - 2,
            big - 1,
        ]
        self.overflow_block = [
            -big,
            -1 - big,
            -2 - big,
            -3 - big,
            big,
            big + 1,
            big + 2,
            big + 3,
        ]
        self.wrong_dtype_block = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        super().setUp()


class TestI16(BaseTestCase.BaseTest):
    def setUp(self):
        self.dtype = "INT16"

        big = 2 ** 15
        self.valid_block = [
            1 - big,
            2 - big,
            3 - big,
            4 - big,
            big - 4,
            big - 3,
            big - 2,
            big - 1,
        ]
        self.overflow_block = [
            -big,
            -1 - big,
            -2 - big,
            -3 - big,
            big,
            big + 1,
            big + 2,
            big + 3,
        ]
        self.wrong_dtype_block = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        super().setUp()


class TestI32(BaseTestCase.BaseTest):
    def setUp(self):
        self.dtype = "INT32"

        big = 2 ** 31
        self.valid_block = [
            1 - big,
            2 - big,
            3 - big,
            4 - big,
            big - 4,
            big - 3,
            big - 2,
            big - 1,
        ]
        self.overflow_block = [
            -big,
            -1 - big,
            -2 - big,
            -3 - big,
            big,
            big + 1,
            big + 2,
            big + 3,
        ]
        self.wrong_dtype_block = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        super().setUp()


class TestI64(BaseTestCase.BaseTest):
    def setUp(self):
        self.dtype = "INT64"

        big = 2 ** 63
        self.valid_block = [
            1 - big,
            2 - big,
            3 - big,
            4 - big,
            big - 4,
            big - 3,
            big - 2,
            big - 1,
        ]
        self.overflow_block = [
            -big,
            -1 - big,
            -2 - big,
            -3 - big,
            big,
            big + 1,
            big + 2,
            big + 3,
        ]
        self.wrong_dtype_block = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        super().setUp()


class TestF32(BaseTestCase.BaseTest):
    def setUp(self):
        self.dtype = "FLOAT32"

        self.valid_block = [
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
        self.overflow_block = [
            -big,
            -1 - big,
            -2 - big,
            -3 - big,
            big,
            big + 1,
            big + 2,
            big + 3,
        ]
        self.wrong_dtype_block = [1, 2, 3, 4, 5, 6, 7, 8]

        super().setUp()

    @unittest.expectedFailure
    def test_read_write_overflow(self):
        super().test_read_write_overflow(self)

    @unittest.expectedFailure
    def test_read_write_wrong_dtype(self):
        super().test_read_write_wrong_dtype(self)


class TestF64(BaseTestCase.BaseTest):
    def setUp(self):
        self.dtype = "FLOAT64"

        self.valid_block = [-2.3e-308, -1.7e308, 1.7e308, 2.3e-308, 0.0, 0.0, 0.0, 0.0]

        big = 1.5e600
        self.overflow_block = [
            -big,
            -1 - big,
            -2 - big,
            -3 - big,
            big,
            big + 1,
            big + 2,
            big + 3,
        ]
        self.wrong_dtype_block = [1, 2, 3, 4, 5, 6, 7, 8]

        super().setUp()

    @unittest.expectedFailure
    def test_read_write_overflow(self):
        super().test_read_write_overflow(self)

    @unittest.expectedFailure
    def test_read_write_wrong_dtype(self):
        super().test_read_write_wrong_dtype(self)


class TestPythonReadWrite(unittest.TestCase):
    def setUp(self):
        self.root = "test.n5"
        self.dataset = "test"
        self.dtype = "UINT8"
        self.dataset_size = [10, 10, 10]
        self.block_size = [2, 2, 2]

        pyn5.create_dataset(
            self.root, self.dataset, self.dataset_size, self.block_size, self.dtype
        )
        self.n5 = pyn5.open(self.root, self.dataset, self.dtype, False)

    def tearDown(self):
        if Path(self.root).is_dir():
            shutil.rmtree(str(Path(self.root).absolute()))

    def test_read_write(self):
        # make sure n5 is initialized to all zeros
        self.assertTrue(
            np.array_equal(
                pyn5.read(self.n5, (np.array([0, 0, 0]), np.array(self.dataset_size))),
                np.zeros([10, 10, 10]),
            )
        )

        # write ones to whole dataset, and then
        # write on partial blocks to make sure data isn't overwritten
        pyn5.write(
            self.n5,
            (np.array([0, 0, 0]), np.array(self.dataset_size)),
            np.ones(self.dataset_size),
        )
        self.assertTrue(
            np.array_equal(
                pyn5.read(self.n5, (np.array([0, 0, 0]), np.array(self.dataset_size))),
                np.ones([10, 10, 10]),
            )
        )

        pyn5.write(
            self.n5, (np.array([1, 1, 1]), np.array([3, 3, 3])), np.ones([2, 2, 2]) * 2
        )
        self.assertTrue(
            np.array_equal(
                pyn5.read(self.n5, (np.array([1, 1, 1]), np.array([3, 3, 3]))),
                np.ones([2, 2, 2]) * 2,
            )
        )

        # test writting non-uniform block to make sure axis orderings are correct
        pyn5.write(
            self.n5,
            (np.array([1, 1, 1]), np.array([5, 5, 5])),
            np.array(range(64), dtype=int).reshape([4, 4, 4]),
        )
        self.assertTrue(
            np.array_equal(
                pyn5.read(self.n5, (np.array([1, 1, 1]), np.array([5, 5, 5]))),
                np.array(range(64), dtype=int).reshape([4, 4, 4]),
            )
        )

