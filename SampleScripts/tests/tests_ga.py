# test_ga.py
#    This file tests ga.py for errors.
#    Author: Kristina Yancey Spencer

import unittest
from mock import Mock
import ga
import numpy as np
from random import choice, random, sample

class XOverTests(unittest.TestCase):

    def setUp(self):
        self.tfilloptions = [np.array([14.0, 16.0, 23.0, 51.0, 55.0, 62.0, 68.0,
                                       87.0, 97.0, 98.0]),
                             np.array([13.0, 20.0, 21.0, 22.0, 57.0, 60.0, 84.0,
                                       84.0, 88.0, 0]),
                             np.array([16.0, 18.0, 24.0, 54.0, 56.0, 65.0, 70.0,
                                       87.0, 97.0, 98.0]),
                             np.array([13.0, 20.0, 21.0, 22.0, 57.0, 60.0, 75.0,
                                       84.0, 88.0, 0])]
        self.mocka = Mock()
        self.mocka.getopenbins.return_value = 10
        self.mocka.gettfill.return_value = self.tfilloptions[0].copy()
        self.mockb = Mock()
        self.mockb.getopenbins.return_value = 9
        self.mockb.gettfill.return_value = self.tfilloptions[1].copy()

    def test_xover_mv(self):
        chromoptions = [range(10), sample(range(10), 10),
                        sample(range(10), 10), sample(range(10), 10)]
        mockc = Mock()
        mockd = Mock()
        self.mocka.getgenes.return_value = chromoptions[0]
        self.mockb.getgenes.return_value = chromoptions[1]
        mockc.getgenes.return_value = chromoptions[2]
        mockc.getopenbins.return_value = 10
        mockc.gettfill.return_value = self.tfilloptions[2].copy()
        mockd.getgenes.return_value = chromoptions[3]
        mockd.getopenbins.return_value = 9
        mockd.gettfill.return_value = self.tfilloptions[3].copy()
        mocklist = [self.mocka, self.mockb, mockc, mockd]
        q, newgenes = ga.xover_mv(list(mocklist), 4, 0.9)
        # Check that it returns the correct number of solutions
        self.assertEqual(len(q) + len(newgenes), 4)
        # Check that newgenes didn't change old solutions
        for m in range(len(mocklist)):
            if mocklist[m] in q:
                tfillafter = mocklist[m].gettfill()
                for i in range(10):
                    self.assertEqual(self.tfilloptions[m][i], tfillafter[i])
                self.assertEqual(mocklist[m].getgenes(), chromoptions[m])

    def test_xover_realgene(self):
        chroma = range(10)
        chromb = sample(chroma, len(chroma))
        chroma, chromb = ga.xover_realgene(chroma, chromb)
        self.assertEqual(len(chroma), 10)
        self.assertEqual(len(chromb), 10)
        self.assertTrue(len(chroma) == len(set(chroma)))
        self.assertTrue(len(chromb) == len(set(chromb)))

    def test_xover_tfill(self):
        tfilla, tfillb = ga.xover_tfill(self.mocka, self.mockb)
        self.assertEqual(tfilla[-1], 98.0)
        self.assertEqual(tfillb[-1], 0)
        # Check that newgenes didn't change old solutions
        tfillafter = self.mocka.gettfill()
        for i in range(10):
            self.assertEqual(self.tfilloptions[0][i], tfillafter[i])
        # Check sorting capability
        mocke = Mock()
        mocke.getopenbins.return_value = 8
        tfille = np.zeros(10)
        for i in range(8):
            bini = choice(range(10))
            tfille[i] = self.tfilloptions[0][bini]
        mocke.gettfill.return_value = tfille
        tfillb, tfille = ga.xover_tfill(self.mockb, mocke)
        self.assertEqual(tfille[-1], 0)


class MutatTests(unittest.TestCase):

    def setUp(self):
        self.tfilla = np.array([14.0, 16.0, 23.0, 51.0, 55.0, 62.0, 68.0, 87.0,
                                97.0, 98.0])
        self.tfillb = np.array([13.0, 20.0, 21.0, 22.0, 57.0, 60.0, 84.0, 84.0,
                                88.0, 0])
        self.tfillc = np.array([16.0, 18.0, 24.0, 54.0, 56.0, 65.0, 70.0, 87.0,
                                97.0, 98.0])
        self.tfilld = np.array([13.0, 20.0, 21.0, 22.0, 57.0, 60.0, 75.0, 84.0,
                                88.0, 0])

    def test_mutat_mv(self):
        mocka = Mock()
        mockb = Mock()
        mocka.gettfill.return_value = self.tfilla
        mockb.gettfill.return_value = self.tfillb
        mocka.getgenes.return_value = range(10)
        mockb.getgenes.return_value = sample(range(10), 10)
        q = [mocka, mockb]
        newgenes = [(sample(range(10), 10), self.tfillc),
                    (sample(range(10), 10), self.tfilld)]
        q, newgenes = ga.mutat_mv(q, newgenes, 4, 0.3)
        self.assertEqual(len(q) + len(newgenes), 4)

    def test_mutat_realgene(self):
        chrom = range(10)
        newchrom = ga.mutat_realgene(chrom)
        self.assertFalse(newchrom is range(10))

    def test_mutat_tfill(self):
        tfill = np.array([1372.77308083, 1381.75457038, 1390.73605993,
                          1399.71754949, 1408.69903904, 2550.0, 4350.0,
                          6150.0, 4350.0])
        variability = [[600, 1200, 134.900449290686], [1200, 1800, 143.32316278231067],
                       [1800, 2400, 152.20572431951882], [2400, 3000, 151.89364239043209],
                       [3000, 3600, 162.25968073355423], [3600, 4200, 160.07688178223492],
                       [4200, 4800, 150.91100591911388], [4800, 5400, 107.49740305690803],
                       [5400, 6000, 68.271071373239749], [6000, 6600, 53.324598127813857],
                       [6600, 7200, 1.7503233696504618e-13]]
        tfilla = ga.mutat_tfill(tfill, variability)
        self.assertEqual(len(tfilla), 9)
        self.assertTrue(2400 <= tfilla[5] < 3000)

    def test_calcstdevs(self):
        # Test case
        samplestored = np.load('tests/sample.npz')
        sample2 = samplestored[samplestored.files[0]]
        variability = ga.calcstdevs(sample2)
        self.assertEqual(variability[0][0], 600)
        self.assertEqual(variability[0][1] - variability[0][0], 600)

    def test_sampletfills(self):
        mocka = Mock()
        mockb = Mock()
        mocka.gettfill.return_value = self.tfilla
        mockb.gettfill.return_value = self.tfillb
        mocka.getgenes.return_value = range(10)
        mockb.getgenes.return_value = sample(range(10), 10)
        q = [mocka, mockb]
        newgenes = [(sample(range(10), 10), self.tfillc),
                    (sample(range(10), 10), self.tfilld)]
        sample2 = ga.sampletfills(q, newgenes)
        self.assertEqual(len(sample2), 38)


if __name__ == '__main__':
    unittest.main()
