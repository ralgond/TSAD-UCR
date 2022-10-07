import unittest

from utils import moving_average
from utils import has_intersection

class TestMovingAverage(unittest.TestCase):
    def test1(self):
        l = [1,2,3,4,5,6,7,8]
        l_ma = moving_average(l, 3)
        self.assertTrue(all([a == b for a,b in zip(l_ma, [2., 3., 4., 5., 6., 7.])]))

class TestHasIntersection(unittest.TestCase):
    def test1(self):
        r1 = (1,2)
        r2 = (3,4)
        self.assertFalse(has_intersection(r1, r2))
    
    def test2(self):
        r1 = (0,1)
        r2 = (3,4)
        self.assertFalse(has_intersection(r1, r2))

    def test3(self):
        r1 = (3,4)
        r2 = (5,6)
        self.assertFalse(has_intersection(r1, r2))

    def test4(self):
        r1 = (3,4)
        r2 = (6,7)
        self.assertFalse(has_intersection(r1, r2))

    def test5(self):
        r1 = (3,5)
        r2 = (5,7)
        self.assertTrue(has_intersection(r1, r2))

    def test6(self):
        r1 = (3,5)
        r2 = (4,7)
        self.assertTrue(has_intersection(r1, r2))

    def test7(self):
        r1 = (3,5)
        r2 = (2,8)
        self.assertTrue(has_intersection(r1, r2))

if __name__ == "__main__":
    unittest.main()
